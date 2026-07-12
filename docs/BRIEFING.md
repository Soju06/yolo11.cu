# Contributor briefing (for humans and agents)

Architecture facts you must know before touching this repo. Read this fully.

## The pipeline

1. `export/export_yolo11.py <model>` loads the ultralytics model, calls `fuse()` (BN folded into
   conv weights), and *decomposes* every composite module into a flat primitive-op graph:
   `CONV, ADD, MAXPOOL5, UPSAMPLE2, COPYC, ATTN, DECODE` — written to `build/<model>/model.graph`
   (text), `weights.f16` (fp16, OHWI = [Cout][kh][kw][Cin]), `bias.f32`.
   It then re-executes the decomposed graph in torch fp32 and asserts <1e-4 vs ultralytics layer
   hooks (level-1 verification), and dumps every op's output (fp16 weights) to `ref/opNNN.npy`.
2. `engine/engine.cu` (single file, ~1500 lines) parses the graph, allocates NHWC fp16 buffers
   with **view tensors** (buffer id + channel offset + row stride; all split/concat is zero-copy),
   and dispatches hand-written kernels. `dump` mode writes every op output; `test/compare.py`
   asserts <3% max-rel vs the refs (level-2 verification).
3. Batch dim: buffers sized `net.B ×`; every kernel takes `Bn`; per-B CUDA graphs share buffers.
   `--batch N` on the CLI. Attention loops images (`ATTN` dispatch).
4. `yolo11serve` (server/serve.cpp) embeds the engine via `engine/yolo11.h` (`-DYOLO11_LIB`).

## Key engine internals

- `k_conv_mma<BM,BN,STAGES,KONE,AMODE,WARPS_M,WARPS_N>`: implicit GEMM, mma.m16n8k16 + cp.async.
  GEMM K axis = flat (ky,kx,ci). Weights are already the B matrix ([N][K] k-contig).
  `wstride` param lets you GEMM over non-weight tensors (attention does QK^T this way).
  Epilogue: bias + optional SiLU (`act=1`) + optional residual (`Res`). 1×1 convs: `KONE=true`.
  Requires Cin % 8 == 0 (except the dedicated first-conv kernel `k_conv0`, Cin=3).
- Startup **autotuner** times 4-warp vs 8-warp variants per conv — new layer shapes need NO manual
  tuning. Do not hand-tune tiles; extend the dispatch only if a genuinely new op class appears.
- `zerobias` (1024 floats of 0) exists for GEMM-as-matmul calls. `attnP`/`vt` are attention scratch.
- Op struct fields are overloaded (e.g. CONV uses `b` as fused-residual tensor). Read `loadGraph`.
- Weight offsets must stay 8-halves aligned (all channel counts % 8 == 0 guarantees this).

## Verification gates (all must pass before you are done)

```bash
make export MODEL=<model>            # includes the level-1 decomposition assert
make && make test MODEL=<model>      # per-op compare + detect
# task-specific end-to-end check vs ultralytics predict outputs (see your spec)
# regression: the detect path must not change
make test MODEL=yolo11n
./yolo11cuda bench build/yolo11n 300   # must stay ~0.90 ms ±3% (cuda graph line)
```

## Rules

- Single-file engine stays single-file. No new runtime dependencies.
- Exporter: derive EVERYTHING from the actual model modules (channels, repeats, heads) — no
  per-scale hardcoding. Scales n/s/m must all export.
- New primitives get: exporter emission + torch reference in `run_reference` + engine parse +
  kernel + dispatch. The per-op dump/compare must cover them.
- fp16 storage everywhere; fp32 accumulation in reference; engine default AMODE=2 (hybrid).
- Do NOT git commit; the maintainer reviews and commits.
- Comments in English, match existing style (sparse, explain constraints not mechanics).
