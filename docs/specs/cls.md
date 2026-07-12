# yolo11n-cls (ImageNet classification) — implementation spec

Target: `yolo11n-cls.pt` (already in repo root), 224×224 input, 1000 ImageNet classes.
This spec is self-contained: every fact below was extracted from the actual model
(`ultralytics 8.x`, torch fp32, `model.fuse()` applied) and from the ultralytics source in
site-packages. The proposed decomposition was **prototype-verified end to end**: worst
level-1 abs diff vs ultralytics hooks = 3.2e-5 (< 1e-4 gate), top-5 identical.

Verified reference numbers on `ultralytics/assets/bus.jpg` (CPU, fp32):

```
top5 classes: [654, 734, 874, 757, 829]
top5 probs:   [0.571011, 0.337454, 0.042098, 0.014150, 0.005880]
names:        minibus, police_van, trolleybus, recreational_vehicle, streetcar
```

Note: ranks 5 and 6 are nearly tied (829 @ 0.005880 vs 408 @ ~0.00588); torch-on-GPU flips
them. Never gate on rank 5.

---

## 1. Model topology (yolo11n-cls, after fuse())

`ClassificationModel`, 11 top-level layers — **the detect backbone layers 0–9 verbatim
(same module classes the exporter already decomposes), no SPPF, no FPN neck, no Detect
head.** Layer 9 is C2PSA (in detect it is layer 10; there is no SPPF before it here).
Input 224×224 → spatial 112, 56, 56, 28, 28, 14, 14, 7, 7, 7.

| i | module | exact config (n scale) | out shape (hook) |
|---|--------|------------------------|------------------|
| 0 | Conv | 3→16 k3 s2 p1 SiLU | 16×112×112 |
| 1 | Conv | 16→32 k3 s2 p1 SiLU | 32×56×56 |
| 2 | C3k2 | cv1 32→32, c=16, m=[Bottleneck(16→8 k3, 8→16 k3, add)], cat 48, cv2 48→64 | 64×56×56 |
| 3 | Conv | 64→64 k3 s2 p1 SiLU | 64×28×28 |
| 4 | C3k2 | cv1 64→64, c=32, m=[Bottleneck(32→16, 16→32, add)], cat 96, cv2 96→128 | 128×28×28 |
| 5 | Conv | 128→128 k3 s2 p1 SiLU | 128×14×14 |
| 6 | C3k2 | cv1 128→128, c=64, m=[C3k(cv1 64→32, cv2 64→32, 2×Bottleneck(32,k3,k3,add), cv3 64→64)], cat 192, cv2 192→128 | 128×14×14 |
| 7 | Conv | 128→256 k3 s2 p1 SiLU | 256×7×7 |
| 8 | C3k2 | cv1 256→256, c=128, m=[C3k(cv1 128→64, cv2 128→64, 2×Bottleneck(64), cv3 128→128)], cat 384, cv2 384→256 | 256×7×7 |
| 9 | C2PSA | cv1 256→256, c=128, n=1 PSABlock: attn(qkv 128→256 1×1, **heads=2, kd=32, hd=64, N=7·7=49**, pe dw3×3 128, proj 128→128 1×1), ffn 128→256→128; cv2 256→256 | 256×7×7 |
| 10 | Classify | see §2 | (1,1000)×2 tuple |

All existing exporter decomposers (`c3k2`, `c3k`, `bottleneck`, `c2psa`, `psablock`)
handle layers 0–9 **unchanged** — verified numerically.

### Scale behavior (from yolo11{s,m}-cls.yaml builds; derive from modules, never hardcode)

| | n | s | m |
|---|---|---|---|
| layer 0/1 out | 16/32 | 32/64 | 64/128 |
| layer 2 (type, cv2 out) | Bottleneck, 64 | Bottleneck, 128 | **C3k**, 256 |
| layer 4 / 6 / 8 out | 128/128/256 | 256/256/512 | 512/512/512 |
| C2PSA c, heads (kd=32 hd=64 always) | 128, 2 | 256, 4 | 256, 4 |
| Classify conv in→out | 256→1280 | 512→1280 | 512→1280 |
| linear | 1280→1000 | 1280→1000 | 1280→1000 |

`Classify` internal width is **always 1280** (hardcoded `c_ = 1280` in ultralytics),
linear always 1280→nc. Attention token count is always N=49 at 224 input.

---

## 2. Classify head — exact math (ultralytics/nn/modules/head.py)

```python
class Classify(nn.Module):
    export = False
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)        # 1x1 s1 conv + BN + SiLU (BN folds on fuse())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)             # has bias
    def forward(self, x):
        if isinstance(x, list): x = torch.cat(x, 1) # never taken (single input, f=-1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training: return x
        y = x.softmax(1)
        return y if self.export else (y, x)         # eval: (probs, logits) tuple
```

Step-by-step for n at 224 (all verified, diff exactly 0.0 vs hook):

1. `conv`: 1×1 s1 p0, 256→1280, fused bias, **SiLU** → [B,1280,7,7]
2. `pool`: AdaptiveAvgPool2d(1) == mean over dims (2,3) → [B,1280,1,1]; `flatten(1)` → [B,1280]
3. `drop`: p=0.0 → exact no-op at eval; **omit entirely**
4. `linear`: `y = x @ W.T + b`, W [1000,1280], b [1000] → logits [B,1000]
5. `softmax(dim=1)` (plain, no temperature) → probs [B,1000]

Predict-time output consumed by ultralytics postprocess is `preds[0]` = **softmax probs**.
`nc` must be read from `head.linear.out_features` (=1000). **`model.yaml['nc']` says 80
here — it is a stale detect default, do not use it.**

---

## 3. Input preprocessing — exactly as ultralytics does it (NOT letterbox!)

`ClassificationPredictor.preprocess` (models/yolo/classify/predict.py) applies, per image:

```python
img = cv2.imread(path)                                    # BGR u8 HWC
t   = classify_transforms(imgsz)                          # imgsz = 224 from ckpt train_args
x   = t(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))   # PIL RGB in
```

`classify_transforms(224)` (data/augment.py) = torchvision Compose of:

1. `T.Resize(224, InterpolationMode.BILINEAR)` — **shortest edge** to 224, aspect kept,
   PIL bilinear which is **always antialiased** (triangle filter). Output long edge:
   `new_long = int(224 * long / short)` — **int() truncation, not round** (torchvision
   `_compute_resized_output_size`). bus.jpg 1080×810 → 298×224 (h×w).
2. `T.CenterCrop(224)`: `crop_top = int(round((rh-224)/2.0))`, same for left — Python
   `round()` = **half-to-even** (bus: top=37, left=0). If the resized image were smaller
   than the crop it pads with 0 — impossible for a square target (shortest edge == 224),
   so ignore.
3. `T.ToTensor()`: u8 HWC RGB → fp32 CHW, **/255**.
4. `T.Normalize(mean=(0,0,0), std=(1,1,1))` — `DEFAULT_MEAN/STD` are 0/1: **exact no-op**.
   There is NO ImageNet mean/std normalization. Final range [0,1], RGB.

So vs detect: no 114-pad letterbox, no aspect-pad; resize-shortest + center-crop instead,
same /255 + RGB channel order. The `.pt` checkpoint carries this exact Compose in
`model.transforms`; the predictor reuses it when `imgsz==224`.

**Exporter must generate `input.f16` through torchvision** (bit-exact path):

```python
from ultralytics.data.augment import classify_transforms
import cv2; from PIL import Image
img = cv2.imread(str(ASSETS / 'bus.jpg'))
S = int(model.args.get('imgsz', 224))
x = classify_transforms(S)(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))[None]  # 1,3,224,224 fp32
x[0].permute(1, 2, 0).numpy().astype(np.float16).tofile(BUILD + '/input.f16')            # NHWC fp16
```

Also keep writing `bus_raw.u8` / `bus_raw.txt` (raw BGR) for the engine's own-preprocess
paths.

---

## 4. Exporter: `build_cls()` second topology path

### 4.1 Task detection & main() branching

```python
from ultralytics.nn.modules import head as uhead
task = 'classify' if isinstance(net[-1], uhead.Classify) else 'detect'
```

Branch `build(net, g)` vs `build_cls(net, g)`, the input generation (§3 vs letterbox),
the hook range, and the graph-file header (§4.4). Detect path stays byte-identical.

### 4.2 build_cls (prototype-verified; ops=54, convs=40, bufs=39, tens=61,
weights=2,800,944 fp16 elems ≈ 5.6 MB, bias=6,080 fp32, activations 2.80 MB @B=1)

```python
def build_cls(net, g):
    S = int(getattr(net[-1], 'imgsz', 0) or 224)   # or thread model.args['imgsz'] in
    x = g.newt(S, S, 3)                            # tensor 0 = input, 224x224x3
    for i in range(len(net) - 1):                  # layers 0..9, all f=-1 sequential
        mod = net[i]
        if isinstance(mod, uconv.Conv):      x = g.conv(x, mod)
        elif isinstance(mod, ublock.C3k2):   x = c3k2(g, x, mod)
        elif isinstance(mod, ublock.C2PSA):  x = c2psa(g, x, mod)
        else: raise RuntimeError(type(mod))
        g.layer_out[i] = x
    head = net[-1]
    h = g.conv(x, head.conv)                       # 1x1 Cin->1280 SiLU
    H, W, C = g.shape(h)
    gp = g.gap(h, g.newt(1, 1, C))                 # NEW op
    logits = g.linear(gp, head.linear)             # emits CONV k=1 on the 1x1 tensor
    probs = g.softmax(logits, g.newt(1, 1, head.linear.out_features))   # NEW op
    g.layer_out[len(net) - 1] = probs
    return g
```

New `Graph` emitters (match existing style):

```python
def gap(self, x, out):
    H, W, C = self.shape(x)
    assert self.shape(out) == (1, 1, C)
    self.ops.append(('GAP', x, out)); return out

def softmax(self, x, out):
    assert self.shape(x) == self.shape(out)
    self.ops.append(('SOFTMAX', x, out)); return out

def linear(self, x, mod, out=None):
    """nn.Linear as a 1x1 CONV on a [1,1,I] tensor (M = B GEMM, reuses k_conv_mma KONE)."""
    O, I = mod.weight.shape
    assert self.shape(x) == (1, 1, I)
    if out is None: out = self.newt(1, 1, O)
    w = mod.weight.detach().numpy().reshape(O, 1, 1, I)     # OHWI, k=1 => no permute needed
    woff, boff = self.add_weight(w, mod.bias.detach().numpy())
    self.ops.append(('CONV', x, out, 1, 1, 0, 1, ACT_NONE, I, O, woff, boff))
    return out
```

`gp`, `logits`, `probs` must be **standalone buffers** (coff=0, tensor C == buffer C):
the engine D2H copy and the softmax kernel index images at stride `buf.C` (`g.newt` already
guarantees this).

### 4.3 run_reference additions (torch fp32, buffers are [C,H,W])

```python
elif kind == 'GAP':
    _, x, out = op; wr(out, rd(x).mean(dim=(1, 2), keepdim=True))
elif kind == 'SOFTMAX':
    _, x, out = op; wr(out, rd(x).softmax(dim=0))          # channel dim of [C,1,1]
```

The dump block already writes `op[2]` for non-ADD ops — GAP/SOFTMAX outs are `op[2]`, no
change needed. `ref/final.npy` for cls = ultralytics **softmax probs** `y[0]`, shape
[1,1000] (model eval output is the tuple `(probs, logits)`).

### 4.4 Graph file: task flag

Detect keeps emitting `YOLO11GRAPH 1` (zero regression risk). Classification emits:

```
YOLO11GRAPH 2
TASK classify 1000
39 61 54
B 224 224 3
...
T 0 0 3
...
GAP 58 59
CONV 59 60 1 1 0 1 0 1280 1000 <woff> <boff>
SOFTMAX 60 61
```

Engine `loadGraph`: after reading magic+ver, `if (ver >= 2) fscanf("%31s %63s %d", tag,
word, &net.nc)` expecting tag `TASK`; `word == "classify"` → `net.task = 1`, else 0.
Version-1 files parse exactly as before (task defaults to detect).

### 4.5 Level-1 verification & names

- Hooks on **all** layers `range(len(net))` (0..10). For i<10 compare as today; for the
  Classify layer the hook output is a tuple — compare `caps[10][0]` (probs, [1,1000])
  against the SOFTMAX out view reshaped [1000,1,1]. Measured worst diff: **3.2e-5**.
- Write `build/<model>/names.txt` (one name per line, index order), replicating
  AutoBackend's ImageNet remap so the CLI prints words, not synsets:

```python
names = dict(model.names)
if isinstance(names[0], str) and names[0].startswith('n0'):
    from ultralytics.utils import ROOT, YAML
    nm = YAML.load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']
    names = {k: nm[v] for k, v in names.items()}
open(BUILD + '/names.txt', 'w').write('\n'.join(names[i] for i in range(len(names))) + '\n')
```

---

## 5. Engine: new primitives and dispatch

### 5.1 Op enum / struct / parse

```cpp
enum OpKind { CONV, ADD, MAXPOOL5, UPSAMPLE2, COPYC, ATTN, DECODE, GAP, SOFTMAX };
```

(append — keeps profile `names[]` indices; extend that array with "GAP", "SOFTMAX").
No new Op fields needed: both ops are `a -> out`. Parse like COPYC:

```cpp
} else if (!strcmp(tag, "GAP")) {
  op.kind = GAP;     if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
} else if (!strcmp(tag, "SOFTMAX")) {
  op.kind = SOFTMAX; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
```

`Net` additions: `int task = 0, nc = 0, probsTen = -1; __half* h_probs = nullptr;`.
End of `loadGraph`: if task==1, find the last SOFTMAX op, set `probsTen = op.out`, assert
its view is a full standalone buffer with H==W==1 and C==nc, and
`cudaMallocHost(&net.h_probs, (size_t)net.B * net.nc * sizeof(__half))`.

### 5.2 GAP kernel (global average pool over H·W, per image/channel, fp32 acc)

Batched-image indexing convention (same as k_copyc/k_add): pixel p of image img lives at
`(img*HW + p) * stride + off`.

```cpp
// global average pool: [H,W,C] view -> [1,1,C] view, mean over H*W. fp32 accumulate.
__global__ void k_gap(const __half* __restrict__ X, int xs, int xo,
                      __half* __restrict__ Y, int ys, int yo, int HW, int C, int Bn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // Bn*C threads
  if (idx >= Bn * C) return;
  int c = idx % C, img = idx / C;
  const __half* x = X + (size_t)img * HW * xs + xo + c;
  float s = 0.f;
  for (int p = 0; p < HW; p++) s += __half2float(x[(size_t)p * xs]);
  Y[(size_t)img * ys + yo + c] = __float2half(s / HW);
}
```

(HW=49, C=1280 → 1280·B threads × 49 reads: sub-microsecond; no vectorization needed.
If desired later, a uint4 8-channel variant à la k_maxpool5 is trivial since C%8==0.)

Dispatch:

```cpp
case GAP: {
  View x = view(net, op.a), y = view(net, op.out);
  int total = Bn * x.C;
  k_gap<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o,
                                              x.H * x.W, x.C, Bn);
  break;
}
```

### 5.3 Linear = existing CONV (no new kernel)

The exporter emits the linear as `CONV k=1 s=1 p=0 g=1 act=0 Cin=1280 Cout=1000` on the
[1,1,1280] GAP output. Dispatch falls into the existing mma path (`g==1 && Cin>=8`,
`kone=true`, M=Bn): `b64 = 1*16 = 16 < 24` → `LAUNCH_TILE(32,32,3,2,2)`, grid (1, 32),
Ktot=1280 (40 BK-steps, exact). Verified against the kernel source: OOB M rows are
guarded (`mok[i] = m < M` → zero-fill) and OOB N (1000 % 32) is guarded in loadB and the
epilogue, so M=1 is safe. The autotuner skips it (`b64 < 24` branch) — nothing to tune.
Weight layout [1000][1280] is already the GEMM B matrix; 1000·1280 elements keep the
8-half weight-offset alignment for subsequent entries (there are none).

### 5.4 SOFTMAX kernel (channel softmax of a [1,1,C] view, one block per image)

fp32 max/sum block reduction (reuse the `red[32]` two-level shuffle pattern from k_attn).
Not in-place (dump-friendly; ref compares both logits CONV out and probs).

```cpp
// channel softmax on a [1,1,C] view; one block per image; fp32 math, fp16 out.
__global__ void k_softmax_ch(const __half* __restrict__ X, int xs, int xo,
                             __half* __restrict__ Y, int ys, int yo, int C) {
  const __half* x = X + (size_t)blockIdx.x * xs + xo;   // H*W==1 -> image stride == xs
  __half* y = (__half*)Y + (size_t)blockIdx.x * ys + yo;
  __shared__ float red[32];
  // pass 1: block max over C   (strided threadIdx.x loop + warp/level reduction into red)
  // pass 2: block sum of expf(x - gmax)
  // pass 3: y[c] = expf(x[c] - gmax) / gsum
}
```

Dispatch:

```cpp
case SOFTMAX: {
  View x = view(net, op.a), y = view(net, op.out);
  k_softmax_ch<<<Bn, 256, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.C);
  break;
}
```

(`k_softmax_rows` is *not* reusable as-is: it is in-place, takes a scale inside exp, and
assumes contiguous rows; a dedicated 20-line kernel is cleaner.)

### 5.5 ATTN at N=49: MUST use the fallback kernel; extend it for batch

The tensor-core ATTN path is gated on `kd==32 && hd==64 && N==400`. **Do not relax it to
N=49**: the P@Vᵀ GEMM uses K=N as the GEMM K axis, and N=49 violates both K%8 and the
16-byte cp.async alignment (row stride 49 halves = 98 B). The scalar `k_attn` fallback is
numerically correct for any N/kd/hd but currently exits on `Bn != 1`. Fix by batching the
grid (image stride inside one buffer = N·qs):

```cpp
// k_attn: add gridDim.z = image index
const __half* Q = QKV + (size_t)blockIdx.z * N * qs;   // and Yb = Y + blockIdx.z * N * ys
...
// dispatch (replace the Bn!=1 exit):
dim3 grid(N, op.heads, Bn);
k_attn<<<grid, 128, N * sizeof(float), st>>>(q.p, q.s, q.o, (__half*)y.p, y.s, y.o,
                                             N, op.heads, op.kd, op.hd);
```

Cost at N=49, heads=2: negligible. Detect (N=400) keeps its tensor-core path untouched.
`attnP`/`vt` scratch stays allocated-but-unused (maxN=49 → 19 KB, harmless).

### 5.6 forward(): task branch (replaces DECODE/NMS pipeline)

```cpp
static void forward(Net& net, cudaStream_t st, int Bn) {
  if (net.task == 0) CK(cudaMemsetAsync(net.detcnt, 0, Bn * sizeof(int), st));
  for (auto& op : net.ops) runOp(net, op, st, Bn);
  if (net.task == 0) {
    k_nms<<<Bn, 256, 0, st>>>(...);            // unchanged detect tail
    ... h_cnt/h_out copies ...
  } else {
    View p = view(net, net.probsTen);          // standalone contiguous buffer, B x nc fp16
    CK(cudaMemcpyAsync(net.h_probs, p.p, (size_t)Bn * net.nc * sizeof(__half),
                       cudaMemcpyDeviceToHost, st));
  }
}
```

All of this is CUDA-graph-capture-safe (plain launches + memcpyAsync into pinned memory,
same pattern as h_out). `fuseAdds` needs no change (the linear CONV is followed by
SOFTMAX, never ADD). Autotune loop needs no change (skips non-CONV; cls conv shapes fall
into existing branches).

### 5.7 CLI: printing class results (detect mode, task-dispatched)

Keep the mode name `detect` (it is the generic "run one inference" mode; model-dir decides
the task). After `forward` + sync, when `net.task == 1`:

```cpp
// load names once: build/<model>/names.txt, one per line (optional; print index if absent)
for (int b = 0; b < net.B; b++) {           // batch consistency like detect
  // partial top-5 over net.nc fp16 probs in net.h_probs + b*net.nc
}
printf("top5:\n");
for (k = 0..4) printf("%4d %-24s %.4f\n", cls, name, prob);
```

Expected output for the reference input: `654 minibus 0.5710 / 734 police_van 0.3375 /
874 trolleybus 0.0421 / 757 recreational_vehicle 0.0142 / 829 streetcar 0.0059`
(rank 5 may legitimately flip to 408 amphibian). With `--batch N`, print per-image top-1
consistency like the detect branch does with `h_cnt`. Do NOT call `fetchNMS` for cls.

### 5.8 CLI `--image` / `pipeline`: cls preprocess kernel

Detect's `k_preprocess` (640 letterbox, pad 114) is wrong for cls. Add
`k_preprocess_cls`, driven by `S = ib.H` (input buffer dims from the graph — do not
hardcode 224). Host-side geometry (matches torchvision exactly):

```cpp
int S = ib.H;
long rw, rh;                                      // resized dims: shortest edge -> S
if (sw <= sh) { rw = S; rh = (long)S * sh / sw; } // integer truncation == python int()
else          { rh = S; rw = (long)S * sw / sh; }
int dy = (int)rh - S, dx = (int)rw - S;
int top  = dy / 2 + ((dy & 1) && ((dy / 2) & 1));  // python round-half-to-even
int left = dx / 2 + ((dx & 1) && ((dx / 2) & 1));
```

Kernel: one thread per dst pixel (S×S); for dst (x,y), resized coords are
`rx = x + left, ry = y + top`; sample the source with a **PIL-style antialiased triangle
filter** per axis (this is what `T.Resize` does on PIL images — plain bilinear does NOT
match on downscale):

```
scale   = (float)src_dim / resized_dim;      // per axis
support = max(scale, 1.f);                   // bilinear filter support * filterscale
center  = (r + 0.5f) * scale;
i0 = max(0, (int)(center - support + 0.5f)); i1 = min(src_dim, (int)(center + support + 0.5f));
w(i) = max(0.f, 1.f - fabsf(i + 0.5f - center) / support);      // normalize by sum
out(x,y) = sum_ij wY(i) * wX(j) * src(i,j) / (sumY * sumX)      // 2D product == separable
```

Then BGR→RGB swap and /255 into the NHWC fp16 input slot (same tail as `k_preprocess`).
Taps ≈ (2·scale+1)² per pixel — fine for a one-shot CLI path. Bit-exactness with PIL is
impossible anyway (PIL rounds the horizontal pass to uint8 before the vertical pass),
expect ≤ ~2/255 per-channel differences; see §6 tolerances. `pipeline` mode for cls uses
this kernel in place of `k_preprocess`; embedding API (`yolo_preprocess`) stays
letterbox/detect-only for now — note in yolo11.h that it must not be used on cls models.

---

## 6. Verification

### Level-1 (exporter, automatic in `make export MODEL=yolo11n-cls`)

Hooks on layers 0..10 as in §4.5, fp32 weights: assert worst < 1e-4. Measured with the
prototype decomposition: 3.2e-5 (worst at layer 2). Layer-10 comparison target is the
probs tuple element `caps[10][0]`.

### Level-2 (per-op GPU dumps)

`make && make test MODEL=yolo11n-cls` → `dump` + `test/compare.py build/yolo11n-cls`
(54 ops incl. GAP/SOFTMAX/linear-CONV, all covered by the existing generic dump path) —
existing <3% max-rel gate. SOFTMAX probs: max value ~0.57, fp16 storage noise ~5e-4,
passes with wide margin. Then `./yolo11cuda detect build/yolo11n-cls` prints top-5.

### End-to-end vs ultralytics predict

```python
from ultralytics import YOLO
from ultralytics.utils import ASSETS
r = YOLO('yolo11n-cls.pt').predict(str(ASSETS / 'bus.jpg'), imgsz=224, device='cpu',
                                   verbose=False)[0]
print(r.probs.top5, r.probs.top5conf.tolist())    # [654,734,874,757,829], [0.571011, ...]
# full vector if needed: r.probs.data.numpy()  (shape [1000], fp32 softmax probs)
```

Compare against the engine's printed top-5 (which consumed the exporter's `input.f16`,
i.e. the bit-exact torchvision preprocessing):

- **top-1 and top-2 class ids: must match exactly** (0.57 vs 0.34, huge margin);
- top-5 probs: `|Δp| < 5e-3` each (engine fp16 + hybrid-mma vs torch fp32);
- ranks 4/5: allow id swaps among near-ties (757/829/408 are within 0.009/0.0001).

For `./yolo11cuda detect build/yolo11n-cls --image third_party/../bus.jpg` (engine-side
preprocess, PIL-antialias approximation): require top-1 match and `|Δp(top1)| < 0.03`;
do not gate tighter — resize filter differences legitimately move tail probs.

Batch: `./yolo11cuda detect build/yolo11n-cls --batch 4` → all images identical input ⇒
identical top-1/probs per slot (exercises the batched k_attn fallback and GAP/SOFTMAX Bn
paths).

### Regression (mandatory)

```bash
make export MODEL=yolo11n && make test MODEL=yolo11n      # detect graph byte-identical (v1 header)
./yolo11cuda bench build/yolo11n 300                      # ~0.90 ms ±3% (cuda graph line)
make export MODEL=yolo11s-cls (optional scale check: exports + level-1 passes)
```

---

## 7. Risks / gotchas

1. **ATTN N=49 alignment**: N%8 != 0 → tensor-core path (P@Vᵀ K axis = N, cp.async 16 B)
   is illegal. Must route to `k_attn` fallback; the fallback needs the batch extension
   (§5.5) or `--batch`/server use aborts. Do not "fix" by relaxing the N==400 guard.
2. **nc source**: `linear.out_features` (1000). `model.yaml['nc']` is 80 (stale detect
   default in the cls checkpoint) — a classic trap.
3. **Preprocessing ≠ letterbox**: shortest-edge resize (int-truncated long edge) +
   center crop (round-half-even offsets) + /255 RGB; mean/std are 0/1 (no ImageNet
   normalization). PIL resize is antialiased — cv2-style plain bilinear diverges on
   downscale; exporter must produce `input.f16` via torchvision so levels 1/2 are exact,
   and the engine's own `--image` path is only class-level accurate (§6 tolerances).
4. **M=1 GEMM (linear)**: safe in `k_conv_mma` (M-row guard `mok`, N tail guard in
   loadB/epilogue, Ktot=1280 divides BK) — but it's the first M<32 conv in the repo;
   verify op052 in the level-2 dump before trusting it. Autotuner correctly skips it.
5. **Channel alignment all OK, but minimal**: layer-2 bottleneck cv1 out = 8 channels —
   hits the mma path's `Cin >= 8` floor exactly. 1280 and 1000 are %8. Linear weight
   block is 1,280,000 halves, preserving 8-half weight offset alignment.
6. **Probs buffer contiguity**: SOFTMAX out (and GAP out) must be standalone buffers
   (coff=0, view C == buf C, H=W=1); the D2H copy and k_softmax_ch image stride (`xs`)
   rely on it. Assert in loadGraph.
7. **NMS/DECODE tail must be task-gated**: for cls there are no DECODE ops, `dets` is
   uninitialized garbage and `detcnt` un-memset — launching `k_nms` unconditionally would
   read junk. Gate memset + k_nms + h_out/h_cnt copies on task==detect (§5.6);
   `fetchNMS`/`yolo_get` are meaningless for cls.
8. **Batch-dim interactions**: new kernels index images at `img*HW*stride` inside each
   buffer (buffers are `net.B ×` sized) — GAP/SOFTMAX/k_attn-z all follow it; h_probs is
   `B × nc`. CUDA-graph capture per B works unchanged (all new work is capture-safe).
9. **Dropout**: p=0.0 inplace at eval — omit; do NOT emit any op.
10. **Eval output is a tuple** `(softmax, logits)`; ultralytics postprocess uses `[0]`.
    `ref/final.npy` must store the softmax probs, and the head hook comparison must
    unpack the tuple.
11. **Input size**: derive everything from the graph (`ib.H`), not a 224 literal — cls
    models can be trained at other imgsz. k_conv0 needs Cout%16==0: 16/32/64 for n/s/m ✓.
12. **Detect regression**: keep the detect exporter emitting `YOLO11GRAPH 1`; version-2
    header + TASK line only for cls. Engine parses both. profile-mode `names[]` must grow
    with the enum or it reads OOB.
13. **Server/proto impact** (notes only, do not implement): the `Detect` rpc returns
    boxes — wrong shape for cls. A future `Classify` rpc would need:
    `ClassifyRequest{bytes image}` →
    `ClassifyResponse{repeated ClassProb{int32 cls; float prob; string name;} top;
    float queue_ms; float infer_ms; int32 batch;}`, plus a task query in the embedding
    API (`int yolo_task(void*)`, `int yolo_cls_get(void*, int slot, int* cls, float* prob,
    int k)`) and a cls variant of `yolo_preprocess` (letterbox is wrong). serve.cpp is
    out of scope for this task; `yolo_create` on a cls dir works mechanically but its
    detect-oriented getters must not be used.
14. **Near-tie at rank 5**: 829 vs 408 differ by ~3e-5 — any end-to-end assert must not
    require rank-5 identity.
