# Spec: yolo11n-seg (instance segmentation) support

Target: `yolo11n-seg.pt`, 640x640 square letterbox input, ultralytics **8.4.92**
(everything below was read off the actual installed source + the actual loaded model — do not
re-derive from ultralytics; if the installed version changes, re-check `Segment`, `Proto`,
`ops.process_mask`, `ops.crop_mask`, `utils/nms.py`).

Plan in one paragraph: layers 0–22 are byte-identical to yolo11n detect (same channels, same
exporter code path). Only layer 23 changes: `Segment` = `Detect` + a `cv4` mask-coefficient
branch per level (plain convs — existing `CONV` covers them) + a `Proto` module on the P3
feature (three plain convs + one `ConvTranspose2d(k=2,s=2)`). We add **three new primitives**:
`PS2` (pixel-shuffle r=2, which together with an ordinary 1x1 `CONV` implements the
ConvTranspose exactly), `DECODESEG` (DFL decode that also copies the 32 mask coefficients into
the candidate record), and `MASKS` (a declarative graph line that names the proto tensor; the
engine assembles binary masks for NMS survivors only, on GPU, at proto resolution 160x160,
cropped to the box).

---

## 1. Model topology (n scale, exact; s/m scaling noted)

`YOLO('yolo11n-seg.pt')`, `model.fuse()` → `net = model.model`, 24 top-level layers.
Layers 0–22: identical module types AND channel counts to yolo11n detect
(P3 = layer 16 out, 80x80x64; P4 = layer 19 out, 40x40x128; P5 = layer 22 out, 20x20x256).
Layer 23 is `Segment` instead of `Detect`.

`Segment` attributes (n): `nc=80, nm=32, npr=64, nl=3, reg_max=16, no=144`,
`stride=[8,16,32]`, `end2end=False, legacy=False, xyxy=False`.

### 1.1 What Segment shares with Detect (already exported today)

- `cv2` (box branch, per level i): `Conv(ch[i],64,k3,SiLU) → Conv(64,64,k3,SiLU) → Conv2d(64,64,k1)`
  with `ch = (64,128,256)` for n. Final 1x1 has bias, **no activation**.
- `cv3` (cls branch, per level i):
  `[DWConv(ch[i],ch[i],k3,SiLU) → Conv(ch[i],80,k1,SiLU)] → [DWConv(80,80,k3,SiLU) → Conv(80,80,k1,SiLU)] → Conv2d(80,80,k1)`.
- `dfl`: `Conv2d(16,1,1,bias=False)` with weight = `arange(16)` (engine does this in `k_decode`).

The existing exporter head loop (`export/export_yolo11.py` `build()`, the `det.cv2/cv3` loop)
handles these unchanged.

### 1.2 Proto module (new) — exact structure, n scale

```
Proto(
  cv1:      Conv2d(64, 64, k=3, s=1, p=1, bias)  + SiLU        # in: P3 = layer16 out, 80x80x64
  upsample: ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=True)   # NO activation
  cv2:      Conv2d(64, 64, k=3, s=1, p=1, bias)  + SiLU        # 160x160
  cv3:      Conv2d(64, 32, k=1, s=1, p=0, bias)  + SiLU        # -> 160x160x32  (SiLU! unlike head finals)
)
forward(x): cv3(cv2(upsample(cv1(x))))
```

`upsample.weight` shape is `[Cin=64, Cout=64, 2, 2]` (torch ConvTranspose layout: Cin first),
`upsample.bias` shape `[64]`. Note `Proto.cv3` is an ultralytics `Conv` (has SiLU), not a bare
`nn.Conv2d` — the emitter's existing `isinstance(mod, uconv.Conv)` logic picks the right act
automatically.

### 1.3 cv4 mask-coefficient branch (new), per level i

```
cv4[i]: Conv(ch[i], 32, k3, SiLU) → Conv(32, 32, k3, SiLU) → Conv2d(32, 32, k1, bias, NO act)
```

n: `c4 = max(ch[0]//4, nm) = max(16,32) = 32`. Outputs: 80x80x32, 40x40x32, 20x20x32.

### 1.4 Scale table (verified by loading the actual s/m checkpoints)

| scale | ch (P3,P4,P5)   | npr (= Proto cv1/up/cv2 width) | nm | cv4 mid c4 |
|-------|-----------------|--------------------------------|----|------------|
| n     | 64, 128, 256    | 64                             | 32 | 32         |
| s     | 128, 256, 512   | 128                            | 32 | 32         |
| m     | 256, 512, 512   | 256                            | 32 | 64         |

`nm=32` and Proto output `=nm` at every scale; `npr = ch[0]`; `c4 = max(ch[0]//4, 32)`.
All channel counts are %8 at all scales. Per repo rules: derive all of these from the module
objects (`mod.cv1.conv.out_channels` etc.), never from this table.

---

## 2. Exact forward math (transcribed from installed source)

### 2.1 Head forward (eval, non-export, non-end2end)

```
feats = [P3 (1,64,80,80), P4 (1,128,40,40), P5 (1,256,20,20)]
boxes  = cat_i( cv2[i](feats[i]).view(1, 64, Hi*Wi) , dim=2)   # (1, 64, 8400)
scores = cat_i( cv3[i](feats[i]).view(1, 80, Hi*Wi) , dim=2)   # (1, 80, 8400)
coef   = cat_i( cv4[i](feats[i]).view(1, 32, Hi*Wi) , dim=2)   # (1, 32, 8400)  ["mask_coefficient"]
proto  = Proto(feats[0])                                        # (1, 32, 160, 160)
```

Anchor order along the 8400 axis: P3 row-major (6400) then P4 (1600) then P5 (400) — identical
for boxes/scores/coef. `make_anchors(..., offset=0.5)` → cell centers `(x+0.5, y+0.5)`; DFL
softmax-expectation over 16 bins per side; `dist2bbox(xywh=True)`; `dbox *= stride`. All of
this is exactly what `k_decode` already implements (xywh vs engine's xyxy differ only by the
NMS-side conversion; engine emits xyxy directly — equivalent).

```
y = cat( dbox(4, xywh, input-pixel units), scores.sigmoid()(80), coef(32, RAW — no activation) , dim=1)
  # (1, 116, 8400)
model(x)  ==  ((y, proto), preds_dict)      # preds_dict = {boxes, scores, feats, mask_coefficient, proto}
```

So for the exporter: `out = model(x); (y, proto) = out[0]`. The current exporter's
`y = y[0] if isinstance(y, (tuple,list))` yields the *tuple* `(y, proto)` — must unpack once more.

Coefficients are used raw (logit-space); the only nonlinearity in the whole mask path is the
final `> 0` threshold (≡ `sigmoid > 0.5`). No eps / normalization constants anywhere (BN is
already folded by `fuse()`; SiLU is exactly `x·σ(x)`).

### 2.2 Ultralytics postprocess (what predict() does)

`nms.non_max_suppression(y, conf=0.25(default), iou=0.7(default!), nc=80, multi_label=False, max_det=300)`:
candidate iff `max_c scores > conf` (best class only — same semantics as `k_decode`); xywh→xyxy;
per-class NMS via the +class·4096-offset trick; output rows `[x1,y1,x2,y2,score,cls, coef*32]`
in 640-letterbox pixel coords.

Then per image (`SegmentationPredictor.construct_result`, default `retina_masks=False`):

```
masks = process_mask(proto[0], pred[:,6:], pred[:,:4], shape=(640,640), upsample=True):
    logits = (coef[N,32] @ proto.view(32, 160*160)).view(N,160,160)         # fp32 matmul
    ratios = [160/640]*4 = 0.25
    logits = crop_mask(logits, boxes * 0.25)     # per pixel: keep iff  col >= x1 && col < x2
                                                 #                   && row >= y1 && row < y2
                                                 # (float compare, HALF-OPEN upper bound)
    logits = F.interpolate(logits[None], (640,640), mode="bilinear")[0]     # align_corners=False
    masks  = logits.gt_(0.0).byte()              # threshold AFTER upsample; sigmoid never applied
pred[:,:4] = scale_boxes(...)                    # boxes → original image coords; masks stay 640-space
keep = masks.amax((-2,-1)) > 0                   # drops detections whose mask is empty
```

**Ordering caveat**: ultralytics thresholds *after* the bilinear upsample to 640; we threshold
at proto resolution. Interior pixels agree exactly; only the boundary (sub-pixel) differs. For
numeric comparison use `upsample=False` (section 6), which produces *precisely* our semantics:
crop at proto res, threshold logit > 0, uint8 `{0,1}` masks of shape `[N,160,160]`.

**predict() rect gotcha**: `model.predict(...)` letterboxes to a *rectangular* stride-multiple
(bus.jpg → 640x480, proto 160x120!). Our engine is fixed 640x640. Always compare by feeding the
already-square-letterboxed image (or the raw tensor) — see section 6.

### 2.3 What OUR engine outputs (decision)

Per NMS survivor: the existing 6-float det `[x1,y1,x2,y2,score,cls]` (640-space, mapped to
original coords at fetch as today) **plus one binary u8 mask at proto resolution 160x160,
box-cropped, threshold logit > 0** (mask assembly for survivors only, on GPU, inside the CUDA
graph). Client maps mask pixel `(mx,my)` → letterbox pixels `[4mx, 4mx+4) x [4my, 4my+4)` →
original coords `((4mx - left)/scale, (4my - top)/scale)`. We do NOT replicate the
bilinear-upsample-then-threshold step; document this as the one intentional deviation
(boundary-only, ≤ ~2 px at 640 scale).

---

## 3. Decomposition into engine primitives

### 3.1 Covered by existing ops

- `cv2`, `cv3` per level: unchanged (already exported for detect).
- `cv4[i]`: three `CONV`s (3x3 SiLU, 3x3 SiLU, 1x1 no-act). Cin/Cout ∈ {64,128,256,32} → all %8,
  all take the mma path; autotuner handles shapes, no manual tuning.
- `Proto.cv1` (3x3 SiLU 64→64 @80x80), `Proto.cv2` (3x3 SiLU 64→64 @160x160),
  `Proto.cv3` (1x1 **SiLU** 64→32 @160x160): plain `CONV`s.
- `Proto.upsample`: 1x1 `CONV` (64→256, act=0) + new `PS2` op — see 3.2.

### 3.2 NEW primitive: `PS2` (pixel shuffle, r=2)

**Equivalence.** `ConvTranspose2d(C→C, k=2, s=2, p=0)` is non-overlapping (stride == kernel), so
each output pixel `(2i+di, 2j+dj)` depends on exactly one input pixel `(i,j)`:

```
y[o, 2i+di, 2j+dj] = Σ_c x[c,i,j] · Wt[c, o, di, dj] + b[o]       # Wt: [Cin, Cout, 2, 2]
```

That is a 1x1 conv producing `4C` channels followed by a spatial re-arrangement. We use the
**quadrant-block channel layout** (q-major, q = 2·di + dj), because each quadrant block starts
at channel offset `q·C` (%8 aligned → uint4 vector moves):

```
w1x1[q*C + o, c] = Wt[c, o, di, dj]         with q = 2*di + dj      # rows OHWI: [4C, 1, 1, C]
b1x1[q*C + o]    = b[o]
PS2: out(2i+di, 2j+dj, c) = in(i, j, (2*di+dj)*C + c)
```

Exporter re-layout (verified numerically vs `F.conv_transpose2d`, max abs diff 3.6e-7 fp32;
also equal to `F.pixel_shuffle` with the interleaved `o*4+q` layout — we deliberately use the
q-major layout instead):

```python
wt = mod.upsample.weight.detach()                       # [C, C, 2, 2]
C  = wt.shape[0]
w  = wt.permute(2, 3, 1, 0).reshape(4 * C, 1, 1, C).numpy()   # [(2*di+dj)*C + o][1][1][c], OHWI
b  = np.tile(mod.upsample.bias.detach().numpy(), 4)           # [4C], q-major tiling
woff, boff = g.add_weight(w, b)
g.ops.append(('CONV', t, up4, 1, 1, 0, 1, ACT_NONE, C, 4 * C, woff, boff))   # act = 0 !
g.ops.append(('PS2', up4, ps))
```

**Graph line** (consistent with UPSAMPLE2/COPYC style): `PS2 <in_ten> <out_ten>` where
`in = [H,W,4C]` view, `out = [2H,2W,C]` view. Exporter emitter + assert:

```python
def ps2(self, x, out):
    H, W, C4 = self.shape(x)
    assert self.shape(out) == (H * 2, W * 2, C4 // 4)
    self.ops.append(('PS2', x, out)); return out
```

**Torch reference** (`run_reference`, buffers are CHW):

```python
elif kind == 'PS2':
    _, x, out = op
    v = rd(x); C = v.shape[0] // 4; H, W = v.shape[1:]
    y = torch.zeros(C, 2 * H, 2 * W)
    for di in range(2):
        for dj in range(2):
            q = 2 * di + dj
            y[:, di::2, dj::2] = v[q * C:(q + 1) * C]
    wr(out, y)
```

**Kernel** — clone of `k_upsample2` (engine.cu:585) with a per-quadrant channel offset:

```cuda
// pixel shuffle r=2 from quadrant channel blocks: in [H,W,4C] -> out [2H,2W,C], C % 8 == 0.
// in(i, j, (2*di+dj)*C + c) -> out(2i+di, 2j+dj, c)
__global__ void k_ps2(const __half* X, int xs, int xo, __half* Y, int ys, int yo,
                      int H, int W, int C, int Bn) {           // C = OUTPUT channels
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Ho = H * 2, Wo = W * 2, cg = C >> 3;
  if (idx >= Bn * Ho * Wo * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int rem = pix % (Ho * Wo), brow = (pix / (Ho * Wo)) * H;
  int ox = rem % Wo, oy = rem / Wo;
  int q = ((oy & 1) << 1) | (ox & 1);
  *(uint4*)(Y + (size_t)pix * ys + yo + c) =
      *(const uint4*)(X + ((size_t)(brow + (oy >> 1)) * W + (ox >> 1)) * xs + xo + q * C + c);
}
```

**Dispatch** (`runOp`): grid over `Bn * 2H * 2W * (C/8)` threads, TB=256, exactly like the
UPSAMPLE2 case. Parse in `loadGraph`: same 2-int format as UPSAMPLE2. Add `PS2` to the
`OpKind` enum (append at the end, keep existing values), to the profile-mode `names[]` array,
and dump-mode covers it automatically (it has a real `op.out`).

### 3.3 NEW primitive: `DECODESEG`

**Graph line**: `DECODESEG <box_ten> <cls_ten> <coef_ten> <stride>` → parse into
`op.a = box, op.b = cls, op.out = coef (overloaded, per house style), op.stride`.
`nm = tens[op.out].C` (32). Exporter emitter:

```python
def decode_seg(self, box, cls, coef, stride):
    self.ops.append(('DECODESEG', box, cls, coef, stride))
```

Reference interpreter: same as DECODE (`dets.append((rd(box), rd(cls), rd(coef), st)); continue`)
— it produces no dumped tensor (dump loop must `continue` on it like DECODE).

**Semantics**: identical to `k_decode` (best-class score, sigmoid, conf 0.25, DFL, xyxy·stride),
plus: the candidate record grows from 6 to `6 + nm` floats, and the kernel appends the nm raw
fp16 coefficients (converted to fp32) read from the coef view at the same anchor index:

```
o[0..5] = as today;  o[6 + i] = (float)COEF[idx * ms + mo + i]   for i in 0..nm-1
```

**Kernel**: template the existing `k_decode` as `template <int NM>` with three extra params
`(const __half* MC, int ms, int mo)` and a `cstride = 6 + NM` record stride; the tail copy uses
uint4 loads (`NM % 8 == 0`). Instantiate `k_decode<0>` for DECODE (dead-strips to today's code —
detect codegen unchanged) and `k_decode<32>` for DECODESEG. If keeping the codegen provably
identical matters more than 40 lines, duplicate the kernel instead; either is acceptable, but
`make test MODEL=yolo11n && bench` is the gate.

**Dispatch**: like DECODE but with `View m = view(net, op.out)` and the seg candidate stride.

### 3.4 NEW primitive: `MASKS` (declarative) + `k_mask_assemble`

**Graph line**: `MASKS <proto_ten>` — emitted once, as the LAST op. Exporter:
`g.ops.append(('MASKS', t_proto))`. Reference interpreter: `continue` (no dump).
`loadGraph` records `net.protoTen = op.a; net.nm = tens[op.a].C;` `runOp` case is a no-op
(`break`) — assembly must run *after* NMS, so it is launched from `forward()`.

**Buffer/alloc changes in `loadGraph`** (all sized with `net.B ×`):

```
net.detStride = 6 + net.nm (6 when no MASKS op)          // candidate record stride, floats
net.dets:    B * MAXDET * detStride * 4 B                 (yolo11n-seg: 4096*38*4 ≈ 0.62 MB/img)
net.outidx:  B * MAX_OUT ints                             // NMS survivor -> candidate slot
net.masks:   B * MAX_OUT * Ph * Pw u8                     (300*160*160 = 7.68 MB/img, device only)
```

`Ph, Pw` come from the proto tensor's buffer (do NOT hardcode 160; it is netW/4).

**`k_nms` change** (shared with detect; NMS is a single block per image, not perf-critical):
add `int cstride` (read candidate rows at `cand + i * cstride`, only the first 6 floats are
used) and `int* outidx0`; in the tid==0 compaction loop also store `outidx[m] = o` (the
candidate slot index — `bx1[i]` etc. were loaded from slot `i`, and `order[]` permutes slot
indices, so `o` IS the slot). Detect passes `cstride=6`; `outidx` is always allocated.

**Mask math** (must equal `process_mask(..., upsample=False)` exactly):

```
for survivor d with candidate slot ci, det box (x1,y1,x2,y2) in 640-space:
  bx1 = x1 * (Pw/netW), bx2 = x2 * (Pw/netW), by1 = y1 * (Ph/netH), by2 = y2 * (Ph/netH)
  for each proto pixel (px, py):
    logit = Σ_{c=0..nm-1} coef[ci][c] * proto[py, px, c]          # fp32 accumulation
    inside = (px >= bx1 && px < bx2 && py >= by1 && py < by2)      # float compares, half-open
    mask[d, py, px] = (inside && logit > 0) ? 1 : 0
```

(no sigmoid: `logit > 0 ≡ sigmoid(logit) > 0.5`; if soft masks are ever wanted, emit
`σ(logit)` as fp16 instead of thresholding.)

**Kernel sketch** — one block per (survivor, image); fixed grid so it captures into the CUDA
graph; blocks past `outcnt` exit (same pattern as `k_nms`):

```cuda
// mask assembly for NMS survivors: logits = coef · proto, crop to box, threshold > 0.
__global__ void k_mask_assemble(const __half* P, int ps, int po, int Ph, int Pw, int nm,
                                const float* cand, int cstride, int maxdet,
                                const int* outidx, const int* outcnt, const float* outdets,
                                uint8_t* masks, float rx, float ry) {
  int img = blockIdx.y, d = blockIdx.x;
  if (d >= outcnt[img]) return;
  const float* det = outdets + ((size_t)img * MAX_OUT + d) * 6;
  int ci = outidx[img * MAX_OUT + d];
  const float* cf = cand + ((size_t)img * maxdet + ci) * cstride + 6;
  extern __shared__ float co[];                       // nm floats
  if ((int)threadIdx.x < nm) co[threadIdx.x] = cf[threadIdx.x];
  __syncthreads();
  float bx1 = det[0] * rx, by1 = det[1] * ry, bx2 = det[2] * rx, by2 = det[3] * ry;
  const __half* Pi = P + (size_t)img * Ph * Pw * ps;  // batch stride = rows * buffer stride
  uint8_t* out = masks + ((size_t)img * MAX_OUT + d) * Ph * Pw;
  for (int p = threadIdx.x; p < Ph * Pw; p += blockDim.x) {
    int x = p % Pw, y = p / Pw;
    const __half* pp = Pi + (size_t)p * ps + po;
    float acc = 0.f;
    for (int c = 0; c < nm; c += 8) {                 // nm % 8 == 0
      uint4 v = *(const uint4*)(pp + c);
      const __half* h = (const __half*)&v;
#pragma unroll
      for (int j = 0; j < 8; j++) acc += co[c + j] * __half2float(h[j]);
    }
    bool in = ((float)x >= bx1 && (float)x < bx2 && (float)y >= by1 && (float)y < by2);
    out[p] = (in && acc > 0.f) ? 1 : 0;
  }
}
```

Launch from `forward()` right after `k_nms`, before the D2H copies:

```c
if (net.protoTen >= 0) {
  View pr = view(net, net.protoTen);
  k_mask_assemble<<<dim3(MAX_OUT, Bn), 256, net.nm * sizeof(float), st>>>(
      pr.p, pr.s, pr.o, pr.H, pr.W, net.nm, net.dets, net.detStride, Net::MAXDET,
      net.outidx, net.outcnt, net.outdets, net.masks,
      (float)pr.W / netW, (float)pr.H / netH);          // netW/H from tens[0]'s buffer
}
```

Worst case cost: 300·25600·32 MACs ≈ 0.25 GFLOP ≈ 50–100 µs; typical (≤20 survivors) is
negligible. **Do NOT copy `net.masks` to host inside the graph** (7.7 MB/img ≈ 0.3 ms of D2H):
masks stay device-resident; fetch on demand after sync, copying only `h_cnt[img]` masks.

### 3.5 Exporter build() changes

```python
det = net[23]
if hasattr(det, 'proto'):                       # Segment
    # Proto on P3 (layer 16 output)
    t  = g.conv(outs[16], det.proto.cv1)                        # 3x3 SiLU, 80x80xC
    H, W, C = g.shape(t)
    up4 = g.newt(H, W, 4 * C)                                   # 80x80x4C
    <emit 1x1 CONV with re-laid-out ConvTranspose weights, act=0>   # section 3.2
    ps  = g.ps2(up4, g.newt(2 * H, 2 * W, C))                   # 160x160xC
    t   = g.conv(ps, det.proto.cv2)                             # 3x3 SiLU
    t_proto = g.conv(t, det.proto.cv3)                          # 1x1 SiLU -> 160x160x32
# per-level head loop (existing), extended:
for i, (f, st) in enumerate(zip(feats, strides)):
    b = <cv2 chain as today>; c = <cv3 chain as today>
    if seg:
        mc = f
        for m_ in det.cv4[i]: mc = g.conv(mc, m_)               # 3x3,3x3,1x1
        g.decode_seg(b, c, mc, st)
    else:
        g.decode(b, c, st)
if seg: g.ops.append(('MASKS', t_proto))
```

Also: `main()` must unpack `(y, proto) = model(x)[0]` and save `ref/final.npy` (shape
`[1,116,8400]`) **and** `ref/proto.npy` (`[1,32,160,160]`).

Level-1 hooks (add to the existing 0..22 loop):

```python
hp  = net[23].proto.register_forward_hook(...)        # compare vs t_proto view, < 1e-4
hcs = [net[23].cv4[i].register_forward_hook(...)]     # compare vs coef views, < 1e-4
```

(the fp32-weights `run_reference` pass provides the graph-side values, exactly as layers 0–22).

New activation memory: proto chain (80·80·64 + 80·80·256 + 2·160·160·64 + 160·160·32) +
coef maps (80·80·32 + 40·40·32 + 20·20·32) ≈ 6.7 M halves ≈ **13.4 MB fp16 per batch slot**.

---

## 4. Graph format / engine plumbing checklist

- `model.graph` header stays `YOLO11GRAPH 1` (the engine reads but ignores the version; new
  tags simply fail to parse on old binaries, which is the desired behavior). New tags: `PS2`
  (2 ints), `DECODESEG` (4 ints), `MASKS` (1 int).
- `OpKind` enum: append `PS2, DECODESEG, MASKS`.
- `fuseAdds`: safe — it only rewrites `CONV` followed by `ADD`; the new head has no residual
  adds, and DECODESEG's overloaded `out` field is never touched by the fusion scan.
- `autotune`: only iterates CONVs — the new 1x1 64→256 @80x80 (M=6400) lands in the (64,64)
  branch and gets timed automatically. Do not hand-tune.
- dump mode: `if (op.kind == DECODE || op.kind == DECODESEG || op.kind == MASKS) continue;`.
- profile mode: extend `names[]`; guard the `view(...)` call for MASKS (`op.out == -1` → use
  `op.a`).
- `Net` additions: `int detStride = 6, nm = 0, protoTen = -1; int* outidx; uint8_t* masks;`.
- `yolo11.h` / API: add
  `int yolo_get_mask(void* h, int slot, int det, unsigned char* out /* Ph*Pw bytes */);`
  and `void yolo_mask_dim(void* h, int* ph, int* pw);` — plain `cudaMemcpy` from
  `net.masks + (slot*MAX_OUT + det)*Ph*Pw` after `yolo_run`/`yolo_sync`. Coordinate mapping to
  the original image is the client's job (section 2.3); boxes keep today's mapping.
- detect CLI mode: when `net.nm > 0`, per detection also print mask pixel count
  (`sum(mask)`), and with `dump`-style flag write `build/<model>/gpu/masks.u8`
  (cnt × Ph × Pw bytes, in output order) + the dets — consumed by `test/compare_seg.py`.

### Batch-dim interactions

- Proto buffer is `B ×` like every buffer; `k_mask_assemble` indexes image `blockIdx.y` with
  batch stride `Ph*Pw*ps` (ps = buffer channel stride) — mirrors how other kernels fold B into
  the pixel index.
- `net.dets` rows are per-image (`img * MAXDET * detStride`), `outidx/outcnt/outdets/masks`
  per-image as shown. Per-B CUDA graphs share all of them (fixed `MAX_OUT` grid + early exit —
  same trick as `k_nms`, so graph capture is unchanged).
- `k_decode<32>`'s extra global-memory traffic exists only in seg graphs; detect graphs
  instantiate `k_decode<0>`.

---

## 5. Verification

### 5.1 Level 1 — exporter decomposition (in `export_yolo11.py`, runs in `make export`)

- layers 0..22 vs hooks: `< 1e-4` (existing assert, unchanged).
- `net[23].proto` hook vs `t_proto` view: `< 1e-4`.
- `net[23].cv4[i]` hooks vs the three coef views: `< 1e-4`.
- (The ConvTranspose→CONV+PS2 rewrite is exact linear algebra; measured 3.6e-7 in fp32.)

### 5.2 Level 2 — per-op kernel compare

`make export MODEL=yolo11n-seg && make && make test MODEL=yolo11n-seg` — the existing
dump/compare covers every new tensor-producing op (all new CONVs + PS2) at the 3% max-rel gate.

### 5.3 Level 3 — end-to-end masks vs ultralytics (new `test/compare_seg.py`)

Reference side (uses the same square 640 letterbox as the engine — never `predict()` on a raw
image, its rect letterbox gives 640x480/proto 160x120 for bus.jpg):

```python
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils import nms, ops
m = YOLO('yolo11n-seg.pt'); model = m.model.float().eval(); model.fuse()
x = torch.from_numpy(np.fromfile('build/yolo11n-seg/input.f16', np.float16)
        .astype(np.float32).reshape(640, 640, 3).transpose(2, 0, 1)[None].copy())
(y, proto), _ = model(x)
dets = nms.non_max_suppression(y, conf_thres=0.25, iou_thres=0.45, nc=80)[0]   # [N, 38]
masks = ops.process_mask(proto[0], dets[:, 6:], dets[:, :4], (640, 640),
                         upsample=False)                                        # [N,160,160] u8 {0,1}
```

`conf=0.25, iou=0.45` match the engine's `k_decode`/`k_nms` constants; this version's NMS uses
best-class only (`multi_label=False` default), same as the engine. Engine side: run
`./yolo11cuda detect build/yolo11n-seg` (with the mask-dump flag) and load
`gpu/masks.u8` + printed dets.

Matching & gates:

- Match dets greedily by (same class, box IoU > 0.8). Gate: every reference det with
  `score > 0.27` has a match; matched `|Δscore| ≤ 0.02`, box corners `≤ 2 px` (existing detect
  behavior; borderline dets near the 0.25 threshold may legitimately appear/disappear in fp16).
- Per matched pair: **binary mask IoU at 160x160**. Gate: `min IoU ≥ 0.95`, `median ≥ 0.98`.
  fp16 flips only pixels whose fp32 logit is near 0; if a stricter check is wanted, recompute
  fp32 logits and require exact agreement on pixels with `|logit| > 0.5`, ignore the rest.
- Sanity (non-gating): full `predict()` parity — feed the pre-letterboxed 640x640 BGR u8 array
  as source with `conf=0.25, iou=0.45`; upsample engine masks 4x nearest and expect IoU ≥ 0.9
  vs `res.masks.data` (lower due to the threshold-order deviation, section 2.2). bus.jpg
  reference (rect letterbox, indicative only): 6 dets — bus 0.899, persons 0.885/0.863/0.822/
  0.442, stop-sign 0.462.

### 5.4 Regression gates

```bash
make export MODEL=yolo11n && make test MODEL=yolo11n     # detect path byte-for-byte behavior
./yolo11cuda bench build/yolo11n 300                      # cuda-graph line ~0.90 ms ±3%
make export MODEL=yolo11s-seg                             # scale generality (level 1 only is fine)
```

---

## 6. Risks / gotchas

1. **predict() rect letterbox** — the single most likely way to "fail" a correct
   implementation. Compare only against square-640 inputs (5.3).
2. **Threshold-vs-upsample order**: our masks are thresholded at 160x160; ultralytics at 640
   after bilinear. Use `upsample=False` for numeric comparison; document the deviation for
   users.
3. **crop_mask is half-open** (`>= x1`, `< x2`) with *float* box coords scaled by exactly
   `Pw/netW = 0.25` — no rounding of the box. Copy the compare directions exactly.
4. **No sigmoid** in the binary path; coefficients and proto are raw logit-space (final Proto
   conv DOES have SiLU though — easy to miss that `Proto.cv3` is an ultralytics Conv, unlike
   the bare `nn.Conv2d` finals of cv2/cv3/cv4).
5. **Alignment**: all new channel counts (64, 256, 32; s:128; m:256/64) are %8 → weight offsets
   stay 8-half aligned, uint4 loads in `k_ps2`/`k_decode<32>`/`k_mask_assemble` are safe.
   `nm=32` constant across scales; assert `nm % 8 == 0` in the exporter anyway.
6. **ConvTranspose weight layout is [Cin, Cout, kh, kw]** — transposed vs Conv2d. The re-layout
   formula in 3.2 already accounts for it (`permute(2,3,1,0)`), verified to 3.6e-7.
7. **Candidate record stride** changes to `6+nm` in seg graphs: `k_nms` must take `cstride`;
   detect graphs pass 6. `Net::MAXDET`/`NMS_CAP`/`MAX_OUT` unchanged (NMS_CAP=1024 truncation
   can differ from ultralytics `max_nms=30000` on pathologically busy images — pre-existing
   detect behavior, not new).
8. **Buffer sizing**: `net.masks` = B·300·Ph·Pw u8 = 7.7 MB/img device-only. Do NOT put the
   mask D2H inside the CUDA graph (fixed-size 7.7 MB·B copy ≈ 0.3 ms·B — would wreck the
   latency line); fetch on demand post-sync, `h_cnt[img]` masks only.
9. **CUDA graph capture**: `k_mask_assemble` uses a fixed `(MAX_OUT, Bn)` grid + early exit on
   `outcnt` (data-dependent work in fixed topology, same as `k_nms`). Anything
   survivor-count-dependent (e.g. compacted copies) is forbidden inside the graph.
10. **Detect-path codegen**: instantiate `k_decode<0>` for DECODE so the detect kernel is
    unchanged; the bench gate (0.90 ms ±3%) is the arbiter. Seg adds ~5 convs + PS2 + mask
    kernel; expect roughly ~1.1–1.3 ms for yolo11n-seg (not gated, but report it).
11. **Input-size assumptions**: derive `Ph, Pw` and the crop ratio from the graph (proto
    tensor's buffer and `tens[0]`'s buffer), not the literal 160/640 — the engine hardcodes 640
    only in preprocess.
12. **`profile`/`dump` modes** touch `op.out` — guard MASKS (out = -1) and skip
    DECODESEG/MASKS in dump, or `compare.py` will report missing/misaligned dumps.
13. **Exporter output unpack**: `model(x)` now returns `((y, proto), preds)` — the current
    one-level `y[0]` unwrap silently yields a tuple and the `np.save` would fail late; unpack
    explicitly.
14. **Empty-mask filter**: ultralytics drops detections whose final mask is all-zero
    (`keep = masks.amax(...) > 0`). The engine does not; the comparison script should apply the
    same filter to the reference before matching (rare: tiny/degenerate boxes).
15. **s/m scales**: Proto width = ch[0] (128/256), cv4 mid = 64 for m — everything still %8 and
    derived from modules; `make export MODEL=yolo11s-seg` must pass level 1 with zero code
    changes.

---

## 7. Server / proto impact (future work, not in scope)

- `proto/yolo.proto` today returns only `Box`. Masks at proto res are 25,600 bytes raw u8 per
  detection — too fat for a per-box `bytes` field at scale. Plan: binary RLE over the 160x160
  bitmap (`repeated uint32 mask_rle` or varint-packed `bytes`, typically < 1 KB/mask) + response
  fields `int32 mask_h, mask_w` and per-box `bytes mask_rle`. Alternative cheap step:
  bit-packing (3,200 B/mask fixed).
- `server/serve.cpp` would call `yolo_get_mask` per kept det after `yolo_run`; RLE encode on
  CPU (µs at this size). No change to dynamic batching.
- Until then, seg models can be served box-only with the existing proto (the engine additions
  are backward compatible: `nm=0` graphs behave exactly as today).
