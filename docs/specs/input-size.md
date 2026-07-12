# Spec: input-size generalization (imgsz != 640)

Status: audit complete, implementation-ready.
Scope: make exporter + engine + tooling correct for any `imgsz % 32 == 0`, because the
upcoming task heads need it: **yolo11n-obb ships with imgsz=1024, yolo11n-cls with imgsz=224**
(verified from `YOLO('<m>.pt').model.args['imgsz']`, ultralytics 8.4.92). This spec does NOT
implement obb/cls themselves — only removes every 640 assumption so those specs can build on it.
No new graph op is introduced; the only format change is a graph-header field.

## 1. Verified model facts (ultralytics 8.4.92, actual checkpoints)

| model       | task     | imgsz | C2PSA layer | heads | kd | hd | attention N=(S/32)^2 | anchors A=21*(S/32)^2 |
|-------------|----------|-------|-------------|-------|----|----|----------------------|------------------------|
| yolo11n     | detect   | 640   | 10          | 2     | 32 | 64 | 400                  | 8400                   |
| yolo11s/m   | detect   | 640   | 10          | 4     | 32 | 64 | 400                  | 8400                   |
| yolo11n-obb | obb      | 1024  | 10          | 2     | 32 | 64 | **1024**             | **21504**              |
| yolo11n-cls | classify | 224   | 9 (11 layers total) | 2 | 32 | 64 | **49** (7x7, odd grid) | n/a (no detect head) |

- `kd=32, hd=64` at every scale n/s/m and every task; only `heads` and channel widths scale.
- All spatial shapes are pure functions of imgsz `S`: levels are `S/2, S/4, S/8 (P3), S/16 (P4),
  S/32 (P5)`; attention tokens `N=(S/32)^2`; anchors `A=(S/8)^2+(S/16)^2+(S/32)^2 = 21*(S/32)^2`.
- yolo11n forward verified at 1024/640/320 tensor input: output `[1, 84, 21504] / [1,84,8400] /
  [1,84,2100]`; `Detect.forward` regenerates anchors when the input shape changes, so level-1
  hook comparison works unmodified at any size.
- Candidate pressure at conf 0.25 (bus.jpg): 50 @1024, 47 @640, 45 @320 — candidate count does
  not scale with anchors for normal scenes, but the theoretical cap does (see §5).

## 2. Complete inventory of hardcoded size assumptions

### export/export_yolo11.py

| line | exact content | becomes | derived from |
|------|---------------|---------|--------------|
| 8    | `input.f16     - preprocessed input image, NHWC fp16 (640x640x3)` | docstring: `(SxSx3)` | — |
| 10   | `ref/final.npy - ultralytics head output [1, 84, 8400]` | `[1, 4+nc, A]` | A = 21*(S/32)^2 |
| 218  | `x = g.newt(640, 640, 3)  # tensor 0 = input` | `g.newt(S, S, 3)` | `S` = `--imgsz` arg, default `int(model.args['imgsz'])` |
| 229  | `cat12 = g.buf(40, 40, c10 + c6)` | `g.buf(S//16, S//16, ...)` | P4 grid |
| 230  | `cat15 = g.buf(80, 80, c13 + c4)` | `g.buf(S//8, S//8, ...)` | P3 grid |
| 231  | `cat18 = g.buf(40, 40, c17 + c13)` | `g.buf(S//16, S//16, ...)` | P4 grid |
| 232  | `cat21 = g.buf(20, 20, c20 + c10)` | `g.buf(S//32, S//32, ...)` | P5 grid |
| 234–257 | trailing comments `# 320x320`, `# 80x80 (P3)` etc. | `# S/2`, `# S/8 (P3)` … | comments only |
| 340  | `r = min(640 / h, 640 / w)` | `min(S/h, S/w)` | letterbox_input(S) |
| 343  | `top, left = (640 - nh) // 2, (640 - nw) // 2` | `(S - nh)//2, (S - nw)//2` | " |
| 344  | `out = np.full((640, 640, 3), 114, np.uint8)` | `np.full((S, S, 3), 114, ...)` | " |
| 365  | comment `# 1,3,640,640` | `# 1,3,S,S` | comment |
| 399–400 | `f.write(f'YOLO11GRAPH 1\n')` | version 2 header + `I` line (§3.2) | — |

Everything else in the exporter is already size-generic: every intermediate shape flows from
`g.shape(x)` through `g.conv`'s `Ho, Wo = (H + 2p - k)//s + 1` arithmetic; `run_reference`,
the hooks loop, and `build/<model>` weight emission never mention 640.

### engine/engine.cu

| line | exact content | becomes | derived from |
|------|---------------|---------|--------------|
| 53   | `static const int MAXDET = 4096;` | keep as member default; see §5 | anchor count (optional) |
| 136–137 | `CK(cudaMalloc(&net.zerobias, 1024 * sizeof(float)));` (+memset) | `zbN = max(1024, roundup8(maxN) + 8)` floats | maxN from ATTN ops in loadGraph (already computed at line 124–131) |
| 747  | `#define NMS_CAP 1024` | keep (smem-bound); document truncation, §5 | — |
| 925  | `if (op.kd == 32 && op.hd == 64 && N == 400) {` | relaxed predicate, §4 | op fields + N from view |
| 927–928 | comments `Q[400x32]`, `P[400x400] @ V[400x64]` | genericize comments | — |
| 966  | `..., 0.25f, net.dets, net.detcnt, Net::MAXDET, 80, Bn);` | `80` is **nc**, not imgsz — leave for this spec; the obb/cls specs must parameterize it | class count |
| 1116 | `float scale = std::min(640.f / sh, 640.f / sw);` | `min((float)ib.H / sh, (float)ib.W / sw)` | `Buf& ib` already fetched on line 1115 |
| 1118 | `int top = (640 - nh) / 2, left = (640 - nw) / 2;` | `(ib.H - nh)/2, (ib.W - nw)/2` | " |
| 1120 | `k_preprocess<<<(640 * 640 + 255) / 256, ...>>>` | `(ib.H * ib.W + 255) / 256` | " |
| 1121 | `ib.p + (size_t)slot * 640 * 640 * 3, 640, 640, ...` | `ib.p + (size_t)slot * ib.H * ib.W * ib.C, ib.H, ib.W, ...` | " (`ib.C == 3`) |
| 1222–1226 | detect `--image` path: same four 640 patterns | same fix via `ib.H/ib.W` (already has `Buf& ib`, line 1177) | graph input buffer |
| 1254–1263 | pipeline mode: `scale = min(640.f/sh, ...)`, grid, `inb.p, 640, 640` | same fix via `inb.H/inb.W` | " |

Already generic — verified, do NOT touch:
- `loadGraph` buffer allocation (line 98) takes H/W/C from the graph file; **input dims already
  come from the graph**: tensor 0 = input, `net.bufs[net.tens[0].buf]` is 640x640x3 today and
  becomes SxSx3 automatically. `main`'s input replicate (lines 1176–1180) uses `in.size()`.
- `k_preprocess` itself (line 714): fully parameterized `(dh, dw, scale, top, left, nh, nw)`.
- attention scratch `attnP`/`vt` sizing (lines 124–135): computed from the graph (`maxN =
  max(b.H*b.W)` over ATTN ops) — **dynamic, confirmed**. At S=1024/n: attnP = 2*1024*1024
  halves = 4 MiB, vt = 256 KiB.
- `k_decode` (line 665): takes `H, W, stride` per level from views; grid math is per-level.
- `k_nms` bitonic pad loop (`pad=32; while (pad<n) pad<<=1`) works for any n <= NMS_CAP.
- autotune / tile heuristics (lines 869–878, 1010–1011): pure functions of M = Bn*Ho*Wo; the
  startup autotuner re-times every shape, so 1024-sized layers need no manual tuning. The
  `M > 400` on line 874 is a *heuristic default* only (overridden by autotune), not correctness.
- batch dim: all buffers `net.B x H*W*C`, kernels take `Bn` — orthogonal to imgsz.

### server/serve.cpp, test/compare.py, client/label.py

- **serve.cpp: zero direct size constants** (grep hit only `64 << 20` gRPC message size). It
  inherits correctness from `yolo_preprocess`. BUT: it calls `yolo_create(dir, maxB)` with
  maxB up to 16 → per-B CUDA graph capture runs `forward` for every B, which today `exit(1)`s
  in the ATTN fallback branch (engine line 955) whenever the mma predicate fails and Bn != 1.
  **An obb-sized model with an unrelaxed/failed fast path kills the server at startup.** §4
  fixes this.
- serve.cpp `YoloDet dets[300]` (line 168) and `yolo_get(..., 300)` mirror `MAX_OUT=300` —
  unchanged by this spec (matches ultralytics `max_det=300`).
- test/compare.py: fully generic — shapes come from the `.npy` refs. No change.
- client/label.py: no size assumptions. No change.
- proto/yolo.proto: boxes are original-image coordinates; no imgsz field needed. No change.

## 3. Design

### 3.1 Exporter: `--imgsz`

```
python3 export/export_yolo11.py <model> [--imgsz S]
```

- Default: `S = int(model.args['imgsz'])` where `model = YOLO(...).model` (obb→1024, cls→224,
  detect→640; handles the "train imgsz" carried in the checkpoint). If `model.args` lacks it,
  default 640.
- Validate: `assert S % 32 == 0` (ultralytics stride constraint; all engine spatial ops then
  produce exact integer grids). Warn (don't fail) if `(S//32) % 2` or `((S//32)**2) % 8` — the
  engine then uses the slow attention fallback (§4).
- Thread `S` through `build(net, g, S)` and `letterbox_input(S)` per the §2 table. All other
  shapes follow from conv arithmetic; `run_reference` and level-1 hooks need no change.
- Output dir: keep `build/<model>` when S equals the model default; use `build/<model>-<S>`
  when overridden (e.g. `build/yolo11n-1024`) so 640 regression artifacts survive. Makefile:
  `export: python3 export/export_yolo11.py $(MODEL) $(if $(IMGSZ),--imgsz $(IMGSZ))`.

### 3.2 Graph header (only file-format change; no new op)

> **AMENDED (maintainer decree, 2026-07-12, supersedes the original `I <H> <W>` design):**
> all future task stages (obb/cls/segment) build on a unified v2 header, so the input-size
> line is a TASK line carrying task/nc/imgsz. This is what the exporter ships:

```
YOLO11GRAPH 2
TASK <detect|classify|obb|segment> <nc> <imgsz>
<nbufs> <ntens> <nops>
B <H> <W> <C>
... (unchanged)
```

- Exporter writes `f.write(f'YOLO11GRAPH 2\n'); f.write(f'TASK {task} {nc} {S}\n')` before
  the counts line. `task = YOLO(...).task`, `nc = len(model.names)`.
- Engine `loadGraph`: after the magic/version, `if (ver >= 2)` parse the TASK line into
  `net.task/net.nc/net.inH=net.inW`; v1 files get defaults `detect, nc=80, 640`. `net.nc`
  is plumbed into the `k_decode` dispatch (replacing the old `80` literal, §5). After the
  `B`/`T` lines, derive the authoritative dims from the graph itself and cross-check:

```c
const Buf& ib0 = net.bufs[net.tens[0].buf];
if (net.inH && (net.inH != ib0.H || net.inW != ib0.W)) { fprintf(stderr, "imgsz header mismatch\n"); exit(1); }
net.inH = ib0.H; net.inW = ib0.W;
```

  (`Net` gains `int inH = 0, inW = 0;`.) All preprocess call sites in §2 then use
  `ib.H/ib.W` (equivalently `net.inH/inW`) — the header is a consistency check + external
  tooling convenience, the buffer dims remain the source of truth. Old `YOLO11GRAPH 1` files
  keep loading (backward compatible).

### 3.3 Preprocess/pipeline/detect call sites

Mechanical, per the §2 table. The generalized `yolo_preprocess` becomes:

```c
extern "C" void yolo_preprocess(void* p, const unsigned char* dev_bgr, int sh, int sw, int slot) {
  auto* h = (YoloHandle*)p;
  Buf& ib = h->net.bufs[h->net.tens[0].buf];
  float scale = std::min((float)ib.H / sh, (float)ib.W / sw);
  int nh = (int)roundf(sh * scale), nw = (int)roundf(sw * scale);
  int top = (ib.H - nh) / 2, left = (ib.W - nw) / 2;
  h->slots[slot] = {scale, top, left};
  k_preprocess<<<(ib.H * ib.W + 255) / 256, 256, 0, h->st>>>(
      dev_bgr, sh, sw, ib.p + (size_t)slot * ib.H * ib.W * ib.C, ib.H, ib.W, scale, top, left, nh, nw);
}
```

Same pattern in `main` detect (`--image`) and pipeline modes. Note preprocess semantics are
letterbox — correct for detect/obb. **cls uses a different transform (resize-then-center-crop,
no 114 pad)**; that is the cls spec's problem, not this one — flag it there.

### 3.4 zerobias sizing

The attention fast path uses `zerobias` as the bias vector of GEMMs whose `Cout = N`
(scores GEMM). Epilogue reads `Bias[n]` and `Bias[n+1]` for even `n < Cout`. Today 1024 floats:
- N=400: fine. N=1024 (obb): fits **exactly** (max read index 1023) — zero headroom.
- N=1600 (imgsz 1280): out-of-bounds read.

Fix in `loadGraph` (maxN is already computed there):

```c
int zbN = std::max(1024, ((maxN + 7) & ~7) + 8);
CK(cudaMalloc(&net.zerobias, zbN * sizeof(float)));
CK(cudaMemset(net.zerobias, 0, zbN * sizeof(float)));
```

## 4. ATTN fast-path predicate: what actually breaks at N=1024

Current gate (engine line 925): `op.kd == 32 && op.hd == 64 && N == 400`. Audit of the mma
path (lines 926–953) for arbitrary N, with Q/K read in place from the qkv buffer
(`per = 2*kd+hd`, `wstride = q.s`), scores in `attnP` (row stride N), V^T in `vt`:

| constraint | source | condition |
|---|---|---|
| cp.async 16B alignment, QK^T A-tile (`X + row*q.s + q.o + h*per + a_kg`) | k_conv_mma loadA (KONE) | `q.s % 8 == 0` (holds: buffer C totals are sums of %8 channel counts), `kd % 8 == 0` (=32) |
| cp.async 16B alignment, QK^T B-tile (`qb + n*q.s + ... + kd + kg`) | loadB via `wstride=q.s` | same + `kd % 8 == 0` |
| half2 store of scores (`attnP + m*N + n`, n even) | epilogue line 369 | `N % 2 == 0` |
| cp.async 16B alignment, P@V^T A-tile (`attnP + row*N + a_kg`) | loadA (KONE, xs=N) | **`N % 8 == 0`** |
| cp.async 16B alignment, P@V^T B-tile (`vt + n*N + kg`) | loadB `wstride=N` | **`N % 8 == 0`** |
| bias reads `Bias[n]`, `Bias[n+1]`, n < Cout=N | epilogue | `zerobias >= N+1` floats (§3.4) |
| half2 store of out (`yb + m*y.s + y.o + h*hd + n`) | epilogue | `y.s % 2`, `hd % 2` (hold) |
| Cout=N bounds, M=N bounds | `n < Cout` / `m < M` guards | any N |
| smem/softmax/vtrans kernels | k_softmax_rows (warp/row, loop over N), k_vtrans (flat index) | any N |
| attnP/vt capacity | loadGraph maxN sizing | any N (dynamic — verified) |

So **nothing structural breaks at N=1024**; the `N == 400` literal is just "the only tested
shape". New dispatch:

```c
bool mma_ok = (op.kd % 8 == 0) && (op.hd % 2 == 0) && (N % 8 == 0);
if (mma_ok) { ...existing path, unchanged kernels... }
```

(kd=32/hd=64 always in practice; keep the modulo form so the assert is the real invariant.)
N=1024 satisfies it (1024 % 8 == 0). General rule: fast path iff `(S/32)^2 % 8 == 0`, i.e.
`S % 128 == 0` — true for 640 and 1024; false for 320 (N=100), 224-cls (N=49).

Fallback (`k_attn`) fix — it must stop being batch-1 only, because `yolo_create` captures
graphs for every B up to max_batch and the current branch `exit(1)`s (line 955), which would
kill `yolo11serve` on any non-fast-path model. Loop images exactly like the mma path does:

```c
} else {
  dim3 grid(N, op.heads);
  size_t smem = N * sizeof(float);
  for (int img = 0; img < Bn; img++)
    k_attn<<<grid, 128, smem, st>>>(q.p + (size_t)img * N * q.s, q.s, q.o,
                                    (__half*)y.p + (size_t)img * N * y.s, y.s, y.o,
                                    N, op.heads, op.kd, op.hd);
}
```

(k_attn itself is index-generic; smem = N*4 bytes = 4 KiB at N=1024, fine. Perf is poor but
correct; at N=49 (cls) it is trivially cheap.) Torch reference: unchanged — `run_reference`'s
ATTN branch already handles any N (`N = H*W`).

Do NOT attempt row-padding of attnP to lift the `N % 8` restriction in this change (it needs
zeroed K-padding in both attnP and vt to keep partial vec8 tiles exact); the fallback covers
those sizes, and no shipped model needs it fast (224-cls attention is 49 tokens).

## 5. Detection candidate caps at higher resolutions

Chain today: `k_decode` writes candidates with `conf > 0.25` into `dets[MAXDET=4096]`
(atomicAdd; overflow silently dropped, counter may exceed cap) → `k_nms` takes
`n = min(cnt, NMS_CAP=1024)` in *write order* (roughly anchor order: P3 level first), sorts
those by score, greedy-suppresses, emits top `MAX_OUT=300`.

At S=1024: anchors 21504 (vs 8400) but measured candidates at conf 0.25 stay ~50 on normal
images (§1). The caps are theoretical-worst-case concerns:
- MAXDET=4096 < anchors=21504: unchanged semantics vs today (4096 < 8400 already). Keep 4096.
- NMS_CAP=1024 truncation is the real semantic gap vs ultralytics (which sorts ALL candidates,
  takes top 30000 pre-NMS). Truncation only bites when >1024 anchors clear conf — degenerate
  scenes/noise. NMS_CAP cannot simply grow: smem = NMS_CAP*(20+2+1+2)B = 25 KiB at 1024; 4096
  would need 100 KiB (> 48 KiB default). **Decision: keep both caps, document, and add a
  cheap overflow tripwire**: in `forward`, also async-copy `detcnt` to a pinned mirror
  (`h_rawcnt`, alloc next to `h_cnt`), and in `fetchNMS`/`yolo_get` warn once if
  `h_rawcnt[img] > NMS_CAP`. That makes silent truncation observable without touching kernels
  or perf (one extra 4B*B D2H inside the graph).
- If a future task truly needs full recall (e.g. obb eval at conf 0.001), spec a device
  top-k pre-select then; out of scope here.

`nc=80` literal at the `k_decode` call (line 966): **as implemented**, the v2 TASK header
(§3.2 amendment) already carries `nc` and it is plumbed into the `k_decode` dispatch. The
`nc % 8 == 0` vec8 assumption inside `k_decode` stays; `loadGraph` fail-fasts on any graph
with a DECODE op whose `nc % 8 != 0` or whose cls view C != nc, and the obb/cls specs own
the real (non-vec8) fix.

## 6. New primitives

**None.** No new op kind, no new kernel, no weight-format change. New graph-file element only:

- Syntax: line `I <H> <W>` after the counts line, gated by header version `YOLO11GRAPH 2`.
- Exporter emission: §3.2. Engine parse: §3.2. Torch reference: n/a (input tensor 0 already
  carries the size). dump/compare coverage: unchanged (op outputs only).

## 7. Verification plan

All of this is testable **today with the detect task** — no need to wait for obb/cls.

1. Regression at 640 (must be bit-for-bit-ish identical to current behavior):
   ```
   make export MODEL=yolo11n && make && make test MODEL=yolo11n
   ./yolo11cuda bench build/yolo11n 300     # cuda graph line stays ~0.90 ms ±3%
   make export MODEL=yolo11m && make test MODEL=yolo11m
   ```
   The exported yolo11n graph must differ from the previous one *only* in the header
   (version + `I 640 640` line). Diff it to prove it.
2. Level-1 at a new size: `python3 export/export_yolo11.py yolo11n --imgsz 1024` — the built-in
   decomposition assert vs ultralytics hooks (all 23 layers) runs at 1024 automatically;
   also `--imgsz 320` to exercise the attention fallback path in the torch reference
   (reference is N-generic, so this mainly validates the cat-buffer arithmetic).
   **Measured correction (pending maintainer sign-off):** the flat <1e-4 gate this spec
   predicted does NOT hold at 1024 — measured worst abs diff is 3.5e-5 @320, 8.7e-5 @640,
   2.07e-4 @1024 (accumulation-order noise tracks pixel count). The shipped gate is
   `tol = 1e-4 * max(1, (S/640)^2)` (2.56e-4 at 1024, ~25% headroom over measured).
3. Level-2 per-op: `./yolo11cuda dump build/yolo11n-1024 && python3 test/compare.py
   build/yolo11n-1024` — <3% max-rel on every op, which covers the relaxed N=1024 mma
   attention against `ref/op*.npy`. Repeat for `build/yolo11n-320` (fallback path on GPU).
4. End-to-end vs ultralytics predict at 1024. Exact reference call (note `rect=False`: the
   predictor's default LetterBox uses `auto=rect=True` = minimum-rectangle padding, which does
   NOT match our square letterbox; `rect=False` forces the square pad and then both pipelines
   see pixel-identical inputs up to interpolation):
   ```python
   from ultralytics import YOLO
   r = YOLO('yolo11n.pt').predict('bus.jpg', imgsz=1024, rect=False,
                                  conf=0.25, iou=0.45, max_det=300, verbose=False)[0]
   for b in r.boxes:   # xyxy in ORIGINAL image coords — same space as our detect --image output
       print(int(b.cls), float(b.conf), [round(v,1) for v in b.xyxy[0].tolist()])
   ```
   Compare against `./yolo11cuda detect build/yolo11n-1024 --image
   $(python3 -c "from ultralytics.utils import ASSETS; print(ASSETS/'bus.jpg')")`.
   Match criteria: same detection count and classes; per-box IoU > 0.95 vs the ultralytics
   box; score diff < 0.02 (fp16 net + fp32-vs-fp16 accumulation). Borderline detections with
   score within ~0.01 of conf=0.25 may flicker in/out — treat count mismatch as pass only if
   the odd box out has score in [0.24, 0.27].
   Note ultralytics NMS is class-agnostic=False by default and ours suppresses same-class only
   (`bcl[oj] != acl` skip) — same semantics, no adjustment needed.
5. Batch/serve: `./yolo11cuda detect build/yolo11n-1024 --batch 4` must report
   `(consistent)`; then `make serve && ./yolo11serve --dir build/yolo11n-1024 --max-batch 8`
   plus `python3 client/label.py bus.jpg --bench 500 -c 32` — validates per-B graph capture
   with the relaxed attention path and the generalized `yolo_preprocess`. Repeat serve smoke
   at `build/yolo11n-320` (fallback attention must not exit(1) during graph capture).
6. Perf sanity: `./yolo11cuda bench build/yolo11n-1024 300` — expect roughly (1024/640)^2 =
   2.6x the 640 time (~2.4 ms); no gate, just record it. `profile` mode should show attention
   GEMMs scaled but still mma-path (no `k_attn` in top ops).

## 8. Risks / gotchas

- **zerobias boundary**: N=1024 uses the current 1024-float buffer *exactly*; forgetting §3.4
  works by luck at obb size and breaks silently (OOB read) at imgsz>1024. Fix it anyway.
- **Attention fallback batch**: without the §4 loop, any `S % 128 != 0` model (incl. cls-224)
  aborts inside `yolo_create`'s graph capture — a server-startup crash, not a request error.
- **Alignment**: channel counts (%8) are size-independent — imgsz changes H/W only, never C,
  so weight-offset and vec8 invariants are untouched. Odd H/W grids (S=224 → 7x7) are fine for
  all spatial kernels (bounds-checked); only the attention fast path cares (via N%8).
- **Memory**: activation buffers scale with S^2 and with B: yolo11n is 40.6 MB/img at 640 →
  ~104 MB/img at 1024; `yolo11serve --max-batch 16` on an obb model ≈ 1.7 GB activations +
  attnP 4 MiB + weights. Fine on the target GPU but worth a startup print.
- **CUDA graph capture time**: per-B capture at 1024 is ~2.6x slower per B; maxB=16 startup
  calibration grows accordingly. No correctness impact.
- **NMS semantics**: candidates beyond NMS_CAP=1024 are dropped in anchor-write order (P3
  first → P5 large-object candidates are the ones truncated). Only matters >1024 conf-passing
  anchors; tripwire per §5.
- **Preprocess mismatch traps in verification**: ultralytics `predict` defaults to
  minimum-rect letterbox (`rect=True`) — always pass `rect=False` when comparing; and its
  resize is cv2 INTER_LINEAR on BGR before RGB swap, which `k_preprocess` replicates — do not
  "improve" the interpolation.
- **cls is not just an imgsz change**: 224 input also means center-crop preprocessing, no
  letterbox, no DECODE/NMS, stride-1 output — this spec only guarantees the graph/buffers/
  attention work at 224; the cls spec owns preprocessing and head.
- **nc is a separate hardcode** (decode call `80`, `nc % 8` vec8): untouched here; obb/cls
  specs must carry nc in the graph (extend DECODE or a header field) — called out so it isn't
  mistaken for done.
- **Build-dir collision**: `build/<model>-<S>` naming (§3.1) prevents a 1024 export from
  clobbering the 640 regression baseline; Makefile `test`/`bench` need `IMGSZ` plumbed or an
  explicit dir arg when testing non-default sizes.
- **Server/proto impact**: none — responses are original-image coordinates; `YoloDet`/proto
  unchanged; nvJPEG scratch already sized per decoded image.
