# yolo11.cu

A YOLO11 inference engine written entirely in hand-rolled CUDA kernels — **no cuDNN, no cuBLAS, no TensorRT**.

Supports YOLO11 n/s/m scales (the exporter derives the graph from the actual model, so other scales work unmodified), 640×640 input, COCO 80 classes. The whole pipeline — preprocessing, network, DFL decode, NMS — runs on the GPU and is captured as a single CUDA graph.

Reference hardware: RTX 3060 Ti (Ampere, SM 8.6, 8 GB), CUDA 13.3.

## Results

`bench` = network + decode + NMS, one CUDA-graph launch per frame.
`pipeline` = end-to-end: raw image H2D → GPU letterbox → network → decode → NMS → results in pinned host memory.

| | yolo11n | yolo11m |
|---|---|---|
| ultralytics `predict` (full pipeline) | 14.9 ms | 11.3 ms |
| PyTorch fp16 eager, net only (cuDNN) | 6.26 ms | 9.57 ms |
| **this engine, `bench`** | **0.90 ms** (1113 fps) | **3.87 ms** (258 fps) |
| **this engine, `pipeline` (end-to-end)** | **1.21 ms** (828 fps) | **4.07 ms** (246 fps) |

Accuracy (max per-op relative error vs a PyTorch fp32 reference): **0.99%** (n) / **1.3%** (m) in the default accumulation mode — on the order of fp16 storage error itself. Detections on the test images match ultralytics in boxes, classes, and scores across all modes.

## Quick start

```bash
pip install ultralytics          # export-time only; the engine has no Python runtime deps
make export MODEL=yolo11n        # download weights, build graph + numeric references
make                             # builds ./yolo11cuda (nvcc, sm_86 by default)
make test MODEL=yolo11n          # per-op numeric verification + detection check

./yolo11cuda detect build/yolo11n --image photo.jpg   # any jpg/png; boxes in original coords
./yolo11cuda bench build/yolo11n 300
./yolo11cuda pipeline build/yolo11n 300
```

Other scales: `make export MODEL=yolo11m && make test MODEL=yolo11m`.

### CLI

```
yolo11cuda <mode> [model-dir] [iters] [--image path]

  detect     one inference, print detections (--image loads any jpg/png via stb_image)
  bench      net-only benchmark, stream launches vs CUDA graph
  pipeline   end-to-end latency benchmark
  profile    per-op timing breakdown
  dump       write every op's output for test/compare.py

env:
  ACC16=1    pure fp16 mma accumulation   (fastest: n 0.94 / m 3.80 ms, ~2–3% max op error)
  ACC32=1    pure fp32 mma accumulation   (most exact: n 0.98 / m 4.11 ms)
  NOTUNE=1   skip the startup autotuner
  NOFUSE=1   disable residual-add fusion (debugging)
```

## How it works

**Graph export** (`export/export_yolo11.py`) — loads the ultralytics model, folds BatchNorm into conv weights (`fuse()`), and decomposes C3k2 / C3k / SPPF / C2PSA / Detect into seven primitives: `CONV, ADD, MAXPOOL5, UPSAMPLE2, COPYC, ATTN, DECODE` (n: 112 ops, m: 144 ops). Channel widths and attention head counts are read from the modules, so nothing is hardcoded per scale. Weights are stored fp16 in `[Cout][kh][kw][Cin]` order — which is exactly the k-contiguous B matrix the GEMM kernel wants.

**Memory** — NHWC fp16 with *view tensors* (buffer + channel offset + row stride). Every `split`/`chunk`/`concat` in the network is zero-copy: convs write directly into slices of pre-allocated concat buffers.

**Kernels** (`engine/engine.cu`, single file):

- `k_conv_mma` — implicit GEMM on tensor cores: `mma.sync.m16n8k16`, `cp.async` multistage pipeline with a single `__syncthreads` per k-step, XOR-swizzled shared memory + `ldmatrix`. The GEMM K axis is the flat `(ky, kx, ci)` index (*dense-K packing*): per-vec8 im2col coordinates are tracked incrementally, so thin-channel layers waste nothing and the pipeline loop contains zero integer divisions. Epilogue fuses bias + SiLU + residual add (all 14–19 `ADD` ops in the graph are absorbed). Tile variants (64×64, 64×32, 32×32, 64×128 @256 threads; 4-warp and 8-warp grids) are chosen per op — see autotuning below. 1×1 convs take a `KONE` template path that skips im2col entirely.
- **Attention (C2PSA) reuses the same GEMM kernel**: Q·Kᵀ reads the K matrix in place from the qkv buffer via a weight-stride trick, then a row softmax kernel, then P·V through a small Vᵀ transpose. ~8× faster than the naive attention kernel it replaced.
- Special-purpose kernels: `k_conv0` (Cin=3 first conv, 16-output-channel chunks per block), `k_dwconv8` (depthwise, weights repacked `[K²][C]` for uint4 channel loads), `k_maxpool5` (`__hmax2`), `k_decode` (DFL softmax-expectation + sigmoid + confidence filter, vectorized), `k_preprocess` (BGR u8 → letterbox bilinear → fp16, matches cv2), `k_nms` (single block: bitonic sort sized to the candidate count + greedy suppression; results match ultralytics NMS).
- The full forward + decode + NMS + result D2H (into pinned memory) is captured as **one CUDA graph**; per frame the host does one graph launch and one sync.

**Accumulation modes** — GA10x runs fp32-accumulate HMMA at half the fp16-accumulate rate, and ncu shows the tensor pipe is the top stall (`math_pipe_throttle` 33%). The default is therefore a *hybrid*: fp16 mma over a bounded K=64 window (2 k-steps), flushed into fp32 registers. This captures most of the pure-fp16 speed while the error stays bounded by the window. `ACC16`/`ACC32` select the pure modes.

**Startup autotuning** — for every conv on the 64×64 tile, the engine times the 4-warp and 8-warp grid variants at load (tens of ms, `NOTUNE=1` to skip) and keeps the winner. The 8-warp variant doubles occupancy (33% → 67%) and hides barrier/wait stalls, but loses on some shapes — measuring beats heuristics, and the choice adapts automatically to other GPUs and model scales.

**Verification** (`make test`) — two levels: (1) at export, the decomposed graph is re-executed with PyTorch fp32 and checked against ultralytics layer hooks (~1e-5); (2) the CUDA engine dumps every op's output and `test/compare.py` checks each against fp16-weight references. End-to-end, detections are compared against ultralytics on real images.

## Performance engineering notes

Every optimization here was accepted or rejected on ncu/nsys measurements (RTX 3060 Ti):

- Initial ncu SpeedOfLight on the big convs: compute 37% / memory 78–83% — L2-bandwidth bound on im2col A-tile re-reads (one per n-block). Wide-N tiles (64×128, 256 threads) fixed this.
- Warp-stall profile after that: `math_pipe_throttle` 33%, `barrier` 20%, `wait` 20%, `long_scoreboard` (global memory) **1.7%** — tensor-pipe bound, not memory. This motivated the hybrid accumulation default and the 8-warp autotuned grids.
- Tried and rejected, with numbers:
  - **Two-pass explicit im2col** for stride-2 convs — the dense A matrix (29 MB) exceeds L2 (4 MB), turning L2-resident re-reads into DRAM round-trips (m: 4.08 → 4.31 ms).
  - **(64,256) tiles with 2 stages** — shallow pipeline made 1×1 convs 2.3× slower.
  - **`cp.async.ca`** (L1-cached loads) — shared memory occupies the L1 partition; no effect.
  - **Space-to-depth for stride-2 convs** — mooted by the stall data above (memory stall is 1.7%).
  - **mbarrier `cuda::pipeline`** (arrive/wait split instead of `__syncthreads`) — per-stage mbarrier overhead exceeded the barrier savings (n 0.99 → 1.11 ms, m 4.10 → 4.75 ms).

**Not implemented: TMA + `wgmma`.** These are Hopper-class (SM 9.0+) features — hardware async tensor-memory copies and warpgroup-level async MMA would replace this engine's `cp.async` loaders and per-k-step barrier synchronization outright, and are the expected next structural win. This repo targets SM 8.6 and has only been tuned and verified on Ampere; a Hopper port is untested future work.

## Layout

```
engine/engine.cu          the entire engine: graph executor, kernels, CLI
export/export_yolo11.py   graph/weight exporter + reference generator
test/compare.py           per-op numeric comparison
third_party/stb_image.h   vendored jpg/png loader
build/<model>/            export artifacts (not tracked)
```

## Requirements

- NVIDIA GPU, SM 8.0+ (uses `cp.async`, `ldmatrix`, `mma.sync`; tuned on SM 8.6 — pass `ARCH=-arch=sm_XX` to make for other targets)
- CUDA toolkit 12+
- Python + `ultralytics` for the export step only
