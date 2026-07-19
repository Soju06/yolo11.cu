# yolo.cu

> **1,100+ FPS object detection on a $400 gaming GPU** — YOLOv8 and YOLO11, every kernel written by hand in one CUDA file.

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
yolo.cu               ███                                      1.13 ms  ← 13× end-to-end
```

| | yolov8n | yolo11n | yolo11m |
|---|---|---|---|
| ultralytics `predict` (full pipeline) | 9.2 ms | 14.9 ms | 11.3 ms |
| PyTorch fp16 eager, **net only** (cuDNN) | 5.08 ms | 6.26 ms | 9.57 ms |
| **yolo.cu, net + decode + NMS** | **0.94 ms** · 1064 fps | **0.90 ms** · 1113 fps | **3.87 ms** · 258 fps |
| **yolo.cu, end-to-end** | **1.16 ms** · 864 fps | **1.13 ms** · 888 fps | **4.07 ms** · 246 fps |

*End-to-end* means everything: H2D copy of the raw image, GPU letterboxing, the network, DFL decode, NMS — all captured in **one CUDA graph**, one launch per frame. The host's entire job is `cudaGraphLaunch` + one sync, then reading boxes out of pinned memory.

The big scales run too, with identical boxes — both families:
**yolo11l 5.1 ms** (PyTorch fp16: 13.7 ms, 2.7×), **yolo11x 9.5 ms** (14.7 ms, 1.5×),
**yolov8s 1.85 / v8m 4.2 / v8l 7.3 / v8x 11.4 ms** (v8l/x PyTorch: 10.9 / 16.9 ms, 1.5×).
On the large scales the startup autotuner also chooses between tile widths per conv
(64×64 vs 64×128), not just warp grids — measured, not heuristic.

And it's not a lossy trick: max per-op deviation from a PyTorch fp32 reference is **0.99%** (n) / **1.3%** (m) — the same order as fp16 rounding itself. Detections match ultralytics on real images in boxes, classes, and scores.

## Try it (60 seconds)

```bash
pip install ultralytics          # export-time only — the binary needs nothing but a GPU
make export MODEL=yolo11n        # download weights → build graph + numeric references
make                             # nvcc → ./yolocuda
./yolocuda detect build/yolo11n --image your_photo.jpg
```

```
$ ./yolocuda detect build/yolo11n --image bus.jpg
kept=5  (boxes in original image coords)
cls= 5 score=0.9397 box=(12.1, 228.5, 799.3, 735.1)     # bus
cls= 0 score=0.9021 box=(48.6, 398.0, 243.3, 904.4)     # person
cls= 0 score=0.8486 box=(670.6, 392.6, 810.0, 879.7)    # person
cls= 0 score=0.8336 box=(223.1, 405.6, 345.3, 859.8)    # person
cls= 0 score=0.3989 box=(-0.1, 550.4, 66.2, 871.7)      # person

$ ./yolocuda pipeline build/yolo11n
end-to-end pipeline (H2D + preprocess + net + decode + NMS): 1.13 ms/frame (888 fps)
```

Any ultralytics YOLOv8/YOLO11 checkpoint works the same way: `make export MODEL=yolov8s`,
`MODEL=yolo11l`, `MODEL=yolov8n-seg`… The exporter is a generic topology walker over the model's
own layer graph — channel widths, block repeats, attention heads, task heads are all read off the
checkpoint, so scales and task variants compose with zero per-model code. Fine-tuned checkpoints
with custom class counts work too: the cls branch — whose width tracks nc in the ultralytics
heads — is zero-padded to 8-aligned channels at every stage (the conv kernels need %8 widths)
and the decode kernels ignore the pad channels.

## Not just detection

All four YOLO11 task heads run on the same hand-written kernels, each verified per-op against a
PyTorch fp32 reference and end-to-end against ultralytics:

Every task, both families, each verified end-to-end against ultralytics:

| task | models | verified against ultralytics | speed (RTX 3060 Ti) |
|---|---|---|---|
| detect | `yolov8n…x`, `yolo11n…x` | boxes/classes/scores match, all 10 checkpoints | 0.90–11.4 ms |
| classify | `yolov8n-cls`, `yolo11n-cls` (224²) | top-1/2 ids match; Δp ≤ 6e-3 (v8) / 1.3e-3 (v11) | **0.53 ms** e2e |
| OBB | `yolov8n-obb`, `yolo11n-obb` (1024²) | **171/171** (v8) and **169/169** (v11) rotated boxes | 1.6 ms |
| segment | `yolov8n-seg`, `yolo11n-seg` | per-instance mask IoU min **0.997** (v8) / **0.999** (v11) | 1.21 ms |

```bash
make export MODEL=yolov8n-seg && make test MODEL=yolov8n-seg && make test-seg MODEL=yolov8n-seg
./yolocuda detect build/yolo11n-obb --image aerial.jpg    # rotated cx,cy,w,h,angle output
./yolocuda detect build/yolov8n-cls --image photo.jpg     # top-5 classes
```

Highlights under the hood: classification's linear layer runs as an M=1 GEMM through the same
tensor-core kernel; OBB replicates ultralytics' probiou fast-NMS on device to 3.6e-7; segmentation
rewrites Proto's ConvTranspose as a 1×1 conv + pixel-shuffle so it also rides the GEMM kernel, and
assembles masks on GPU for NMS survivors only. Input size is fully generic (`make export
MODEL=yolo11n IMGSZ=1024`). The gRPC server serves all four tasks — the response carries plain
boxes, rotated boxes, top-5 classes, or RLE-encoded instance masks depending on the model dir it
was started with.

## Two ways to use it

YOLO deployments split into two very different shapes. This repo ships a dedicated path for each —
and they are guarded against each other: every server/batching change is re-benchmarked at
batch=1, and the single-frame numbers above are from *after* all of it landed.

### 1 · Ultra-low latency — embed the engine

For robotics, video pipelines, or anything that lives frame-to-frame: run batch=1 and embed the
engine directly. The whole thing is one `.cu` file with a tiny C API (`engine/yolo.h`) — no
server, no IPC, no Python. Per frame the host does one CUDA-graph launch and one sync; boxes come
back in pinned memory **1.13 ms** after the raw frame goes in.

```c
#include "engine/yolo.h"

void* h = yolo_create("build/yolo11n", /*max_batch=*/1);   // load + autotune + build graph

// per frame: dev_bgr = your BGR u8 HWC frame, already on the GPU
yolo_preprocess(h, dev_bgr, height, width, /*slot=*/0);    // letterbox+normalize, async
yolo_run(h, /*B=*/1);                                      // one graph launch + sync
YoloDet dets[300];
int n = yolo_get(h, /*slot=*/0, dets, 300);                // boxes in original image coords
```

```bash
nvcc -O3 -arch=sm_86 -DYOLO11_LIB -c engine/engine.cu -o engine_lib.o   # link engine_lib.o + cudart
```

Latency guardrails for this path: input already on-GPU skips the H2D+decode entirely (0.90 ms),
`yolo_run_async`/`yolo_sync` let you overlap your own work, and `make bench` is the regression
check — batch support added zero overhead at B=1 (verified: 0.90 ms before and after).

### 2 · High throughput — the labeling server

For dataset labeling and offline processing: `yoloserve` is a
containerized gRPC server that GPU-decodes JPEGs (nvJPEG, parallel decoder pool), dynamic-batches
requests, and runs the batched engine. It serves whichever task the model dir was exported for —
detect boxes, rotated boxes (obb), top-5 classes, or boxes + RLE instance masks (segment) — over
one RPC. Measured on the same RTX 3060 Ti, raw JPEG bytes in → results out over gRPC:

```
64 concurrent clients   1,123 img/s   p50  55 ms          (67,000+ labeled images/minute)
128 concurrent clients  1,200 img/s   p50 100 ms
8 concurrent clients      168 img/s   p99  51 ms  <- holds the 50 ms SLO exactly
```

**The scheduler** is deadline-aware dynamic batching: every request gets `deadline = arrival +
target latency`. The batcher keeps admitting requests while the *oldest* one could still meet its
deadline after a full max-batch run — computed from the engine's startup-calibrated per-batch-size
latency model plus a decode-time EMA — and fires the moment waiting would risk the SLO or the
batch fills. Throughput is maximized subject to the latency target; under light load it degrades
gracefully to small, fast batches. Decode of batch *k+1* overlaps GPU execution of batch *k*
(2-stage pipeline, ping-pong scratch buffers).

One process can serve several models at once — requests route by the `model` field and the
scheduler deadline-prioritizes across the per-model queues. Measured mixed load on one GPU:
detect 827 img/s + segment 424 img/s concurrently, both holding p50 ≈ 37 ms.

```bash
sudo apt install libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev
make serve
./yoloserve --dir build/yolo11n --dir build/yolo11n-seg:8 --dir build/yolo11n-cls:4
# or containerized:
docker build -t yoloserve . && docker run --gpus all -p 50051:50051 -v $(pwd)/build:/app/build yoloserve

pip install grpcio grpcio-tools
python3 client/label.py photos/*.jpg --out labels.jsonl -c 64      # mass labeling -> JSONL
python3 client/label.py photos/*.jpg --model yolo11n-seg \
        --coco annotations.json -c 64                              # COCO JSON (mask polygons!)
python3 client/label.py bus.jpg --bench 8000 -c 64                 # throughput/latency benchmark
```

Batched engine (the foundation — `--batch N` on the CLI too):

| | B=1 | B=4 | B=16 |
|---|---|---|---|
| yolo11n | 1,115 img/s | 1,823 img/s | **2,121 img/s** |

That B=16 number is 2.3× what cuDNN reaches at its saturated batch-32 (927 img/s).

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
yolocuda <mode> [model-dir] [iters] [--image path]

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
export/export.py          graph/weight exporter (generic topology walker) + references
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
