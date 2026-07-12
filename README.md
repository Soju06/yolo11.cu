# yolo11.cu

> **1,100+ FPS object detection on a $400 gaming GPU** — YOLO11, every kernel written by hand in one CUDA file.

![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?logo=nvidia&logoColor=white)
![yolo11n](https://img.shields.io/badge/yolo11n-0.90_ms_%2F_1113_fps-e91e63)
![deps](https://img.shields.io/badge/runtime_deps-zero-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

No cuDNN. No cuBLAS. No TensorRT. No Python at runtime. The entire engine — tensor-core GEMM, attention, preprocessing, decode, NMS — is ~1,200 lines of hand-rolled CUDA in a single `.cu` file, and it beats PyTorch's cuDNN backend by **7×** on the exact same GPU, producing the exact same detections.

## How fast?

RTX 3060 Ti, 640×640, fp16 — same weights, same images, matching boxes/classes/scores:

```
yolo11n, end-to-end (raw image in → boxes out)

ultralytics predict   ██████████████████████████████████████  14.9 ms
PyTorch fp16 (cuDNN)  ████████████████                         6.3 ms   (net only!)
yolo11.cu             ███                                      1.21 ms  ← 12× end-to-end
```

| | yolo11n | yolo11m |
|---|---|---|
| ultralytics `predict` (full pipeline) | 14.9 ms | 11.3 ms |
| PyTorch fp16 eager, **net only** (cuDNN) | 6.26 ms | 9.57 ms |
| **yolo11.cu, net + decode + NMS** | **0.90 ms** · 1113 fps | **3.87 ms** · 258 fps |
| **yolo11.cu, end-to-end** | **1.21 ms** · 828 fps | **4.07 ms** · 246 fps |

*End-to-end* means everything: H2D copy of the raw image, GPU letterboxing, the network, DFL decode, NMS — all captured in **one CUDA graph**, one launch per frame. The host's entire job is `cudaGraphLaunch` + one sync, then reading boxes out of pinned memory.

And it's not a lossy trick: max per-op deviation from a PyTorch fp32 reference is **0.99%** (n) / **1.3%** (m) — the same order as fp16 rounding itself. Detections match ultralytics on real images in boxes, classes, and scores.

## Try it (60 seconds)

```bash
pip install ultralytics          # export-time only — the binary needs nothing but a GPU
make export MODEL=yolo11n        # download weights → build graph + numeric references
make                             # nvcc → ./yolo11cuda
./yolo11cuda detect build/yolo11n --image your_photo.jpg
```

```
$ ./yolo11cuda detect build/yolo11n --image bus.jpg
kept=5  (boxes in original image coords)
cls= 5 score=0.9397 box=(12.1, 228.5, 799.3, 735.1)     # bus
cls= 0 score=0.9021 box=(48.6, 398.0, 243.3, 904.4)     # person
cls= 0 score=0.8486 box=(670.6, 392.6, 810.0, 879.7)    # person
cls= 0 score=0.8336 box=(223.1, 405.6, 345.3, 859.8)    # person
cls= 0 score=0.3989 box=(-0.1, 550.4, 66.2, 871.7)      # person

$ ./yolo11cuda pipeline build/yolo11n
end-to-end pipeline (H2D + preprocess + net + decode + NMS): 1.21 ms/frame (828 fps)
```

Want a different scale? `make export MODEL=yolo11m && make test MODEL=yolo11m`. The exporter reads the graph out of the actual ultralytics model — channel widths, attention heads, block layouts — so n/s/m all work from the same code.

## Why is it this fast?

Because nothing is generic. Every kernel knows exactly what network it's running:

- **Implicit GEMM on tensor cores, written in raw PTX** — `mma.sync.m16n8k16` + `cp.async` multistage pipelines, XOR-swizzled shared memory, `ldmatrix`. The GEMM K axis is the flat `(ky, kx, ci)` index (*dense-K packing*): im2col coordinates advance incrementally, thin-channel layers waste zero lanes, and the hot loop contains **zero integer divisions**.
- **The graph is fused flat.** BatchNorm is folded into weights at export. Every residual `ADD` in the network is absorbed into a conv epilogue (bias + SiLU + residual in registers). Every `split`/`concat` is zero-copy — convs write straight into slices of the concat buffers through strided views.
- **Attention runs on the conv kernel.** C2PSA's Q·Kᵀ reads the K matrix *in place* from the qkv buffer by treating it as GEMM weights with a stride trick; P·V goes through the same kernel. 8× faster than a dedicated attention kernel we wrote first.
- **The engine autotunes itself at startup** — for each conv it times 4-warp vs 8-warp tile variants (~tens of ms, once) and keeps the winner, so the dispatch adapts to your GPU instead of trusting a heuristic.
- **Hybrid fp16-window accumulation** — Ampere runs fp32-accumulate HMMA at half rate, so the default accumulates in fp16 over a bounded K=64 window and flushes to fp32 registers: pure-fp16 speed, near-fp32 accuracy. `ACC16=1` / `ACC32=1` give you the pure modes.
- **GPU-resident NMS** — bitonic sort sized to the live candidate count + greedy suppression, inside the graph. Nothing crosses PCIe except the final boxes (7 KB, into pinned memory, also inside the graph).

Every claim above is verifiable: `make test` re-derives references from PyTorch fp32 and checks **every single op's output** (109 ops for n, 141 for m), then checks final detections against ultralytics.

## CLI

```
yolo11cuda <mode> [model-dir] [iters] [--image path]

  detect     one inference, print detections (--image loads any jpg/png, boxes in original coords)
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

## Engineering notes — including the failures

Everything was accepted or rejected on Nsight Compute / Systems measurements, and the rejects are documented so nobody walks the same dead ends:

- First ncu pass on the big convs: compute 37% / **memory 78–83%** — L2-bandwidth bound on im2col A-tile re-reads. Fixed with wide-N tiles (64×128, 256 threads).
- Stall profile after that: `math_pipe_throttle` 33%, `barrier` 20%, `wait` 20%, global memory **1.7%** — tensor-pipe bound. That data drove the hybrid accumulation default and the autotuned 8-warp grids (occupancy 33% → 67%).
- Measured and rejected:
  - **Two-pass explicit im2col** for stride-2 convs — dense A (29 MB) blows out L2 (4 MB); DRAM round-trips made it *slower* (m: 4.08 → 4.31 ms).
  - **(64,256) tiles, 2 stages** — shallow pipeline, 1×1 convs 2.3× slower.
  - **`cp.async.ca`** (L1-cached loads) — shared memory eats the L1 partition; no effect.
  - **Space-to-depth for stride-2 convs** — mooted by the 1.7% memory-stall measurement before a line was written.
  - **mbarrier `cuda::pipeline`** — per-stage overhead exceeded the barrier savings (m: 4.10 → 4.75 ms). Reverted.

**Not implemented: TMA + `wgmma`.** Those are Hopper-class (SM 9.0+) features; they would replace the `cp.async` loaders and the per-k-step barrier outright and are the obvious next structural win. This repo is tuned and verified **on SM 8.6 (Ampere) only** — a Hopper port is untested future work.

## Layout

```
engine/engine.cu          the entire engine: graph executor, kernels, CLI  (~1,200 lines)
export/export_yolo11.py   graph/weight exporter + reference generator
test/compare.py           per-op numeric comparison
third_party/stb_image.h   vendored jpg/png loader
build/<model>/            export artifacts (not tracked)
```

## Requirements

- NVIDIA GPU, SM 8.0+ (`cp.async`, `ldmatrix`, `mma.sync`; tuned on SM 8.6 — `make ARCH=-arch=sm_XX` for other targets)
- CUDA toolkit 12+
- Python + `ultralytics` for the export step only

## License

MIT © 2026 Soju06
