# Spec: yolo11n-obb (oriented bounding boxes, DOTA, 1024x1024)

Implementation-ready spec for adding the OBB task to the exporter + CUDA engine. All module
structures, math, and constants below were extracted from the *actual* `yolo11n-obb.pt`
checkpoint and ultralytics **8.4.92** source installed at
`/home/ubuntu/.local/lib/python3.12/site-packages/ultralytics` and verified numerically
(manual decode vs ultralytics forward: max abs diff 6.1e-5 on a random 1024x1024 input).
Do not re-derive from ultralytics source; everything needed is transcribed here.

---

## 1. Model facts (n scale)

- `YOLO('yolo11n-obb.pt')`: `m.task == 'obb'`, `m.model.args['imgsz'] == 1024`.
- 24 top-level layers. **Layers 0..22 are byte-identical in structure to yolo11n detect**
  (same widths: och(4)=128, och(6)=128, och(10)=256, och(13)=128, och(17)=64, och(20)=128;
  C2PSA attn: heads=2, key_dim kd=32, head_dim hd=64). The existing `build()` decomposition
  covers them unchanged — only spatial sizes differ (input 1024 instead of 640).
- Layer 23 is `ultralytics.nn.modules.head.OBB(Detect)` with:
  - `nc = 15` (DOTA classes), `ne = 1` (one angle per anchor), `reg_max = 16`,
    `stride = [8, 16, 32]`, `end2end = False`, non-legacy cv3 (DWConv variant).
  - DFL weight = `arange(16)` (same assert as detect holds).
- Head input channels `ch = (64, 128, 256)` = P3/P4/P5 (layers 16, 19, 22).
- Feature maps at 1024: P3 128x128 (stride 8), P4 64x64 (stride 16), P5 32x32 (stride 32).
  **Anchors: 16384 + 4096 + 1024 = 21504** (concat order P3,P4,P5; row-major y-outer).
- Final inference output: **[1, 20, 21504]** = [1, 4(xywh) + 15(cls) + 1(angle), anchors].

### Head module tree (n scale, after fuse(), exact)

Per level i in {0,1,2} with `ch = (64, 128, 256)`:

```
cv2[i] (box branch — identical structure to Detect):
  0: Conv(ch[i] -> 64, k3 s1 p1) + SiLU
  1: Conv(64    -> 64, k3 s1 p1) + SiLU
  2: Conv2d(64  -> 64, k1)            # 64 = 4*reg_max, no act
cv3[i] (cls branch — identical structure to Detect, but Cout = nc = 15):
  0: [ DWConv(ch[i] -> ch[i], k3 s1 p1, g=ch[i]) + SiLU ; Conv(ch[i] -> 64, k1) + SiLU ]
  1: [ DWConv(64    -> 64,    k3 s1 p1, g=64)    + SiLU ; Conv(64    -> 64, k1) + SiLU ]
  2: Conv2d(64 -> 15, k1)             # nc=15, no act    <-- NOT %8, see padding
cv4[i] (angle branch — NEW vs Detect):
  0: Conv(ch[i] -> 16, k3 s1 p1) + SiLU
  1: Conv(16    -> 16, k3 s1 p1) + SiLU
  2: Conv2d(16  -> 1,  k1)            # ne=1, no act     <-- NOT %8, see padding
```

### Channel formulas (how s/m scale) — derive from modules, never hardcode

From `Detect.__init__` / `OBB.__init__`:
- `c2 = max(16, ch[0]//4, 4*reg_max)` — cv2 hidden width: **64 for n, s and m**.
- `c3 = max(ch[0], min(nc,100)) = ch[0]` — cv3 hidden: n=64, s=128, m=256.
- `c4 = max(ch[0]//4, ne)` — cv4 hidden: n=16, s=32, m=64.
- ch: n=(64,128,256), s=(128,256,512), m=(256,512,512).
- All hidden widths are %8 at every scale. The only unaligned Couts are the two final
  1x1 convs (`nc=15`, `ne=1`) — the padding scheme below is needed at every scale.
- Exporter must read all of these from the module objects (`mod.conv.out_channels` etc.),
  exactly as the detect path does.

---

## 2. Ultralytics forward math, transcribed exactly (8.4.92)

`OBB.forward` -> `forward_head` -> `_inference`. With `x = [P3, P4, P5]`, bs=1:

**forward_head** (`head.py`):
```
boxes  = cat([cv2[i](x[i]).view(1, 64, -1) for i], dim=-1)      # [1, 64, 21504]
scores = cat([cv3[i](x[i]).view(1, 15, -1) for i], dim=-1)      # [1, 15, 21504]
angle  = cat([cv4[i](x[i]).view(1, 1,  -1) for i], dim=-1)      # [1, 1,  21504]
angle  = (angle.sigmoid() - 0.25) * math.pi                     # range [-pi/4, 3pi/4)
```
**The angle transform `(sigmoid(t) - 0.25) * pi` is applied to the raw cv4 logits AFTER the
cross-level concat but BEFORE decode** (per-anchor pointwise, so concat order is irrelevant
— apply it per anchor in the kernel). The stored `preds["angle"]` is already transformed;
it is used both inside `dist2rbox` and appended raw (transformed, NOT stride-scaled,
NOT sigmoid-again) as the last output channel.

**_inference** (OBB then Detect):
```
anchors: make_anchors(feats, stride, offset=0.5) ->
         per level, ax = x_grid + 0.5, ay = y_grid + 0.5 (grid units, row-major, y outer)
dist  = DFL(boxes)                        # [1, 4, 21504]
dbox  = dist2rbox(dist, angle, anchors.unsqueeze(0), dim=1) * strides   # [1, 4, 21504]
out   = cat([dbox, scores.sigmoid(), angle], dim=1)                     # [1, 20, 21504]
```
- `strides` is the per-anchor stride column (8/16/32); it scales **only** the 4 box
  channels, never the angle.
- scores get a plain sigmoid; the angle channel does **not** get another sigmoid.

**DFL** (`block.py`), input `[1, 64, a]`:
```
x.view(1, 4, 16, a).transpose(2,1).softmax(1) ...  # equivalent per (anchor, side):
  per side s in {0..3}: bins = channels [16*s .. 16*s+15]
  p = softmax(bins); d[s] = sum_i p_i * i           # expected value, in grid units
d = (l, t, r, b)  # side order: 0=left,1=top,2=right,3=bottom (same as detect k_decode)
```

**dist2rbox** (`tal.py`), per anchor (grid units), with theta = transformed angle:
```
lt, rb = (l,t), (r,b)
xf = (r - l) / 2                     # rotated-frame center offset x
yf = (b - t) / 2                     # rotated-frame center offset y
x  = xf*cos(theta) - yf*sin(theta) + ax
y  = xf*sin(theta) + yf*cos(theta) + ay
w  = l + r                           # width along the theta direction (NOT rotated)
h  = t + b
output per anchor: (x, y, w, h)      # then all four * stride
```

**Output tensor layout** `[1, 20, 21504]`, channel order:
`[cx, cy, w, h, cls_0..cls_14 (sigmoid), theta (radians)]`, pixel units of the
1024-letterboxed input, anchor order P3(16384) then P4(4096) then P5(1024), row-major.

This math was verified against the real checkpoint: max abs diff 6.1e-5 over all 20x21504
values (fp32).

---

## 3. Rotated NMS: exact ultralytics semantics (must be replicated on GPU)

`non_max_suppression(pred, conf, iou, nc=15, rotated=True)` (`utils/nms.py`):

1. Candidate mask: `pred[:, 4:19].amax(1) > conf_thres` (max over 15 class sigmoids,
   strictly greater). Default predict conf = 0.25.
2. Best class only (`multi_label=False`): `conf, j = cls.max(1)`, keep `conf > conf_thres`.
3. NO xywh->xyxy conversion for rotated. Box for NMS: `xywhr` where
   `boxes = cat((xy + c, wh, angle))` and `c = cls_index * 7680` is added to **x,y only**
   (class separation by coordinate offset, `agnostic=False` default). Distant offsets give
   probiou ~= 0, so a per-pair `cls_i == cls_j` check is exactly equivalent — use that.
4. Suppression = **`TorchNMS.fast_nms` (NOT greedy NMS)**:
   ```
   order = argsort(scores, descending)         # ties: unstable order
   ious  = batch_probiou(boxes[order], boxes[order]).triu_(diagonal=1)
   keep j  iff  no i earlier in order has ious[i, j] >= iou_threshold
   ```
   Crucially, a suppressed box i **still suppresses** later boxes (no greedy revival),
   and the threshold test is `>=` (not `>`). This is *simpler* than the greedy detect
   k_nms: dead flags have no sequential dependence.
5. `i = i[:max_det]` (300). Output rows: `[x, y, w, h, conf, cls, angle]` (7 floats).

**batch_probiou** (`utils/metrics.py`, eps = 1e-7), for boxes (x1,y1,w1,h1,r1) and
(x2,y2,w2,h2,r2) — Gaussian bounding-box covariance from `_get_covariance_matrix`
(floor = 0.0 in the NMS path):
```
a_i = w_i^2 / 12 ;  b_i = h_i^2 / 12                 # variances before rotation
A_i = a_i*cos^2(r_i) + b_i*sin^2(r_i)
B_i = a_i*sin^2(r_i) + b_i*cos^2(r_i)
C_i = (a_i - b_i)*cos(r_i)*sin(r_i)                  # cov matrix [[A,C],[C,B]]

sA = A1+A2 ; sB = B1+B2 ; sC = C1+C2
den = sA*sB - sC^2 + eps                             # eps INSIDE t1/t2 denominator
t1  = 0.25 * (sA*(y1-y2)^2 + sB*(x1-x2)^2) / den
t2  = 0.5  * (sC*(x2-x1)*(y1-y2)) / den              # note sign: (x2-x1)*(y1-y2)
det1 = max(A1*B1 - C1^2, 0) ; det2 = max(A2*B2 - C2^2, 0)
t3  = 0.5 * log( (sA*sB - sC^2) / (4*sqrt(det1*det2) + eps) + eps )
       # numerator here has NO eps; eps appears in the sqrt denominator AND added
       # to the log argument
bd  = clamp(t1 + t2 + t3, eps, 100.0)
hd  = sqrt(1.0 - exp(-bd) + eps)
iou = 1 - hd
```
Compute in fp32 (ultralytics runs this in fp32). Suppress when `iou >= thr`.

Predict defaults: conf=0.25, **iou=0.7** (NOT 0.45!), max_det=300, agnostic=False,
max_nms=30000 candidate cap. The engine's detect path uses iou=0.45; keep the engine at
0.45 for consistency and pass `iou=0.45` explicitly in every ultralytics comparison call.

Result mapping to original image (`OBBPredictor.construct_result` + `ops.scale_boxes`
with `xywh=True`): `x -= pad_x; y -= pad_y; x,y,w,h /= gain`; angle unchanged; **no
clipping** for obb (`clip_boxes` is skipped when xywh=True). `regularize_rboxes` is NOT
called in this version's predict path.

---

## 4. Decomposition into repo primitives

### 4.1 What existing ops already cover

- Layers 0..22: unchanged `build()` code (CONV/ADD/MAXPOOL5/UPSAMPLE2/COPYC/ATTN), only
  spatial constants parametrized by IMG (section 4.4).
- Head cv2 chain: 3 CONV ops per level (existing emission loop for `det.cv2[i]`).
- Head cv3 chain: 5 CONV ops per level (DW k3 + 1x1 + DW k3 + 1x1 + final 1x1) —
  existing `det.cv3[i][:2]` + `det.cv3[i][2]` loop works, **but the final conv must be
  channel-padded** (4.2).
- Head cv4 chain: 3 CONV ops per level — plain Conv+SiLU, Conv+SiLU, 1x1 no-act; the
  generic `g.conv()` emitter handles them, final conv channel-padded to 8.

### 4.2 REQUIRED: output-channel padding for the two unaligned head convs

The engine's invariants (`Cin % 8`, vec8 buffer loads, and the `k_conv_mma` epilogue which
stores `__half2` pairs and reads `Bias[n+1]`) make **odd/unaligned Cout illegal**: with
Cout=15 the epilogue at n=14 reads `Bias[15]` out of bounds and stores 2 halves into a
15-channel view; with Cout=1 it stores channels {0,1} into a 1-channel buffer. Padding is
mandatory, not an optimization.

Exporter: add a padded variant of the conv emitter:

```python
def conv_pad(g, x, conv, cpad, act=ACT_NONE):
    """plain nn.Conv2d, output channels zero-padded to cpad (pad rows: w=0, b=0)."""
    O, Ig, kh, kw = conv.weight.shape
    H, W, Cin = g.shape(x)
    out = g.newt(H, W, cpad)                      # buffer C = cpad (%8)
    w = conv.weight.detach().permute(0, 2, 3, 1).contiguous().numpy()
    w = np.concatenate([w, np.zeros((cpad - O, kh, kw, Ig), w.dtype)], 0)
    b = np.concatenate([conv.bias.detach().numpy(), np.zeros(cpad - O, np.float32)])
    woff, boff = g.add_weight(w, b)
    g.ops.append(('CONV', x, out, kh, conv.stride[0], conv.padding[0], 1, act,
                  Cin, cpad, woff, boff))
    return out
```
- cv3 final: `conv_pad(..., cpad=16)` -> cls view [H,W,16], channels 15 = zero+bias0.
  **Pad bias must be 0.0, not a large negative**: `test/compare.py` normalizes by
  `abs(ref).max()`, a -1e4 pad channel would mask real errors. The decode kernel guards
  `c0+i < nc` instead (4.5).
- cv4 final: `conv_pad(..., cpad=8)` -> angle view [H,W,8], only channel 0 meaningful.
- The torch reference (`run_reference`) needs no change: it convolves with the padded
  weights and reproduces the padded channels exactly, so level-2 per-op compare covers
  these ops as usual.

### 4.3 NEW primitive: `DECODEOBB` (graph op)

Replaces `DECODE` for the obb task; one per level, emitted after the three head chains.

- **Inputs**: `box` tensor view [H,W,64] (cv2 out), `cls` view [H,W,16] (padded cv3 out),
  `ang` view [H,W,8] (padded cv4 out), attribute `stride` (8/16/32).
- **Output**: appends candidates `[cx, cy, w, h, score, cls, theta]` (7 fp32) to the
  per-image candidate buffer via atomic counter; not a tensor — dump mode skips it,
  exactly like DECODE.
- **Graph-file syntax** (consistent with `DECODE a b stride`):
  ```
  DECODEOBB <box_tid> <cls_tid> <ang_tid> <stride>
  ```
  Exporter tuple: `('DECODEOBB', box, cls, ang, stride)` (the generic
  `' '.join(str(v) for v in op)` writer already handles it).
- **Op struct mapping** (`loadGraph`): `op.a = box`, `op.b = cls`, `op.out = ang`
  (field overloading as documented in the briefing), `op.stride = stride`. Parse:
  ```c
  } else if (!strcmp(tag, "DECODEOBB")) {
    op.kind = DECODEOBB;
    if (fscanf(f, "%d %d %d %d", &op.a, &op.b, &op.out, &op.stride) != 4) exit(1);
  ```
  Add `DECODEOBB` to `enum OpKind` (append at the end) and to the `profile` names array.
  `dump` mode: `if (op.kind == DECODE || op.kind == DECODEOBB) continue;`.
  `fuseAdds`/`autotune` are CONV-only and unaffected.

- **Exporter emission** (per level, after `b = cv2 chain`, `c = conv_pad(cv3 final)`,
  `a = conv_pad(cv4 final)`):
  ```python
  g.ops.append(('DECODEOBB', b, c, a, st))
  g.det_ref.append((b, c, a))
  ```

- **Torch reference** (`run_reference`), mirroring the DECODE branch:
  ```python
  elif kind == 'DECODEOBB':
      _, box, cls, ang, st = op
      dets.append((rd(box).clone(), rd(cls).clone(), rd(ang).clone(), st))
      continue
  ```
  and a final-assembly helper used by the level-1 check (validated math, see section 2):
  ```python
  def assemble_obb(dets, nc=15):
      outs = []
      for b, c, a, st in dets:                       # b [64,H,W], c [16,H,W], a [8,H,W]
          _, H, W = b.shape; n = H * W
          d = b.reshape(4, 16, n).softmax(1)
          d = (d * torch.arange(16.).view(1, 16, 1)).sum(1)          # [4,n] = l,t,r,bb
          th = (a[0].reshape(n).sigmoid() - 0.25) * math.pi
          gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
          ax, ay = gx.reshape(-1) + 0.5, gy.reshape(-1) + 0.5
          l, t, r, bb = d
          xf, yf = (r - l) / 2, (bb - t) / 2
          cx = (xf * th.cos() - yf * th.sin() + ax) * st
          cy = (xf * th.sin() + yf * th.cos() + ay) * st
          cls = c[:nc].reshape(nc, n).sigmoid()
          outs.append(torch.cat([torch.stack([cx, cy, (l + r) * st, (t + bb) * st])[None],
                                 cls[None], th.view(1, 1, n)], 1))
      return torch.cat(outs, 2)                       # [1, 4+nc+1, 21504]
  ```

- **Kernel `k_decode_obb`** — clone of `k_decode` with these deltas (thread per
  (img, y, x), same grid math, same `atomicAdd(cnt + img, 1)`):
  1. Class max: loop `for (c0 = 0; c0 < ncpad; c0 += 8)` with vec8 `uint4` loads
     (ncpad = cls-view C = 16, passed from dispatch as `view(net, op.b).C`), and the
     per-element guard `if (c0 + i < nc && f > best)`. `score = 1/(1+expf(-best))`,
     early-out `score < conf` — unchanged confidence semantics vs detect.
  2. DFL: identical 4x(16-bin softmax expected value) block, giving `d[0..3] = l,t,r,b`.
  3. Angle: `float t0 = __half2float(AP[(size_t)idx * as + ao]);` (channel 0 of the
     padded angle view), `float th = (1.f/(1.f+expf(-t0)) - 0.25f) * 3.14159265358979f;`
  4. Decode (fp32):
     ```c
     float xf = (d[2] - d[0]) * 0.5f, yf = (d[3] - d[1]) * 0.5f;
     float cs = cosf(th), sn = sinf(th);
     float* o = out + ((size_t)img * maxdet + slot) * 7;
     o[0] = (x + 0.5f + xf*cs - yf*sn) * stride;   // cx
     o[1] = (y + 0.5f + xf*sn + yf*cs) * stride;   // cy
     o[2] = (d[0] + d[2]) * stride;                // w
     o[3] = (d[1] + d[3]) * stride;                // h
     o[4] = score; o[5] = (float)bcls; o[6] = th;
     ```
  5. Candidate record is **7 floats** (not 6) — see 4.6 buffer sizing.
- **Dispatch** (`runOp`): new case mirroring DECODE:
  ```c
  case DECODEOBB: {
    View b = view(net, op.a), c = view(net, op.b), a = view(net, op.out);
    int total = Bn * b.H * b.W;
    k_decode_obb<<<(total + TB - 1) / TB, TB, 0, st>>>(
        b.p, b.s, b.o, c.p, c.s, c.o, a.p, a.s, a.o,
        b.H, b.W, op.stride, 0.25f, net.dets, net.detcnt, Net::MAXDET,
        net.nc, c.C, Bn);
    break; }
  ```
  While here, fix the detect DECODE dispatch (engine.cu:966) to pass `net.nc` instead of
  the literal `80` (net.nc defaults to 80 for v1 graphs, so detect is bit-identical).

### 4.4 NEW primitive: rotated NMS kernel `k_nms_obb`

Not a graph op — selected in `forward()` (engine.cu:975) by `net.task`:
```c
if (net.task == 1 /*obb*/)
  k_nms_obb<<<Bn, 256, 0, st>>>(net.dets, net.detcnt, net.outdets, net.outcnt,
                                net.nmsCov, 0.45f, Net::MAXDET);
else
  k_nms<<<Bn, 256, 0, st>>>(...unchanged...);
```

Semantics: ultralytics **fast-NMS** transcribed in section 3 (probiou, `>=` threshold,
suppressed boxes still suppress, same-class only, rank-ordered output capped at 300).

Constants: `#define OBB_NMS_CAP 2048` (detect keeps NMS_CAP 1024; **1024 is too small for
OBB** — the boats.jpg reference image produces 1346 candidates at conf 0.25).

Scratch: `float* nmsCov` device buffer, `Bn * OBB_NMS_CAP * 5` floats
(`x, y, A, B, C` per candidate), allocated in `loadGraph` when task==obb. Shared memory
stays small: `bsc[2048] (float) + order[2048] (short) + dead[2048] (uchar)` = 14 KB
(the detect kernel's 5 box arrays would not fit at CAP 2048; box/cov data lives in
global, which is L2-resident at 40 KB/image).

Algorithm sketch (one block per image, 256 threads):
```
n = min(cnt[img], OBB_NMS_CAP)
// pass 1: load scores, init order, precompute covariances into nmsCov
for i in tid..n step blockDim:
  c = cand + i*7
  w=c[2]; h=c[3]; r=c[6]
  a = w*w/12; b = h*h/12; cs=cosf(r); sn=sinf(r)
  cov[i] = { c[0], c[1], a*cs*cs + b*sn*sn, a*sn*sn + b*cs*cs, (a-b)*cs*sn }
  bsc[i] = c[4]; dead[i] = 0            // (pad slots: bsc=-1, dead=1)
  order[i] = i
__syncthreads()
// pass 2: bitonic sort of order[] by score desc — identical code to k_nms
// pass 3: fast-NMS, parallel over j (NO serial i-loop with syncthreads needed)
for jj in tid..n step blockDim:
  oj = order[jj]; cj = cand + oj*7; short cls_j = (short)cj[5]
  load (xj, yj, Aj, Bj, Cj) from cov[oj]
  for ii in 0..jj-1:
    oi = order[ii]
    if ((short)cand[oi*7+5] != cls_j) continue
    iou = probiou(cov[oi], cov[oj], centers)      // exact formulas from section 3,
                                                  // fp32, eps=1e-7, bd clamp [eps,100]
    if (iou >= thr) { dead[oj] = 1; break; }
__syncthreads()
// pass 4 (tid==0): walk order[], copy survivors' 7 floats to out, cap MAX_OUT=300,
//                  set *outcnt — same as k_nms
```
The O(n^2/2) pair loop at n=2048 is ~2M probiou evaluations; with cov precomputed there
is no trig in the inner loop, well under 1 ms and only paid on OBB models.

### 4.5 Task flagging in model.graph (header v2)

Exporter writes:
```
YOLO11GRAPH 2
TASK obb 15 1 1024
<nb> <nt> <no>
B ... / T ... / ops ...
```
`TASK <name> <nc> <ne> <imgsz>` (detect: `TASK detect 80 0 640`). Engine `loadGraph`:
```c
if (ver >= 2) {
  char task[16]; int ne, im;
  if (fscanf(f, "%31s %15s %d %d %d", tag, task, &net.nc, &ne, &im) != 5) exit(1);
  net.task = strcmp(task, "obb") ? 0 : 1;
} // ver 1: defaults net.task=0, net.nc=80 (old detect graphs keep loading)
```
Add to `Net`: `int task = 0; int nc = 80;` and derived `int detK() const { return task ? 7 : 6; }`.

### 4.6 Engine buffer / IO plumbing changes (exact lines)

- engine.cu:117-121: size candidate/output buffers with `detK` instead of 6:
  `net.dets: B*MAXDET*detK`, `net.outdets/h_out: B*300*detK`.
- engine.cu:975-977: NMS kernel switch (4.4) + `h_out` copy size `B*300*detK`.
- `fetchNMS` (engine.cu:1032-1041) and the `detect`/`pipeline` printers: for obb read
  7-float rows and print `cls score xywhr`. Keep detect output format untouched.
- ATTN fast path guard engine.cu:925: `if (op.kd == 32 && op.hd == 64 && N == 400)` —
  relax to `if (op.kd == 32 && op.hd == 64)`. The tensor-core path is generic in N
  (all launches derive grid/strides from N); at 1024 input the C2PSA runs at
  **N = 32*32 = 1024 tokens** and the fallback `k_attn` would be both slow and
  batch=1-only. `attnP` is already sized from the graph
  (2 heads * 1024 * 1024 * fp16 = 4 MB; `vt` 256 KB) — no alloc change needed.
- Preprocess 640 hardcodings — replace with `ib.H`/`ib.W` from the input buffer
  (tensor 0's buffer), which the graph already carries:
  - engine.cu:1116-1121 (`yolo_preprocess`): `640.f/sh`, `(640-nh)/2`, grid
    `(640*640+255)/256`, slot offset `640*640*3`, and the two `640, 640` k_preprocess args.
  - engine.cu:1222-1226 (`detect --image` path): same five spots.
  - engine.cu:1254-1263 (`pipeline` mode): `std::min(640.f/sh, ...)`, `(640-nh)/2`,
    grid and dims.
  - k_preprocess itself is already parametric (dh, dw args); only call sites change.
- engine.cu:966: literal `80` -> `net.nc` (see 4.3).

### 4.7 Exporter changes (exact lines in export/export_yolo11.py)

- Input size: `IMG = m.model.args.get('imgsz', 640)` (it is `1024` in this checkpoint;
  may be a list/tuple in some checkpoints — take `IMG[0]` if so; assert `IMG % 32 == 0`).
  Thread IMG through:
  - L218 `x = g.newt(640, 640, 3)` -> `g.newt(IMG, IMG, 3)`.
  - L229-232 concat buffers: `cat12/cat18 = g.buf(IMG//16, IMG//16, ...)`,
    `cat15 = g.buf(IMG//8, IMG//8, ...)`, `cat21 = g.buf(IMG//32, IMG//32, ...)`.
  - L340-344 `letterbox_input()`: all five 640s -> IMG.
  - Docstring L8/L10 and stale spatial comments (L234-257 "320x320" etc.) — update.
- L260 `det = net[23]` -> `det = net[len(net)-1]`; L371/L387 `range(23)` ->
  `range(len(net)-1)`.
- Head emission: branch on `isinstance(det, uhead.OBB)`; emit cv2/cv3 (with
  `conv_pad(..., 16)` for cv3 final), cv4 chain (`g.conv` x2 + `conv_pad(..., 8)`),
  then `DECODEOBB`. Keep the DFL-weight assert (holds for OBB).
- Graph header: version 2 + TASK line (4.5) — for detect exports too (engine keeps
  reading v1, but new exports are uniform v2).
- `run_reference`: DECODEOBB branch + `assemble_obb` (4.3).
- Level-1 check additions after the existing per-layer loop:
  ```python
  final = assemble_obb(dets)                       # from the fp32_weights run
  d = (final - torch.from_numpy(np.load(BUILD + '/ref/final.npy'))).abs().max()
  print('final decode diff', d.item()); assert d < 5e-4
  ```
  (Measured 6.1e-5 with fused fp32 weights; 5e-4 gives headroom.)
- `ref/final.npy` for obb is `[1, 20, 21504]` (the existing save line works as is —
  `model(x)` returns `(y, preds)` and the exporter already unwraps the tuple).

### 4.8 Server / proto / embedding API

`YoloDet` (engine/yolo11.h:7) is axis-aligned xyxy and `proto/yolo.proto` `Box` has no
angle. Minimal, ABI-stable plan:
- Add `int yolo_task(void* h);` (0=detect, 1=obb) and
  `typedef struct { float x, y, w, h, angle, score; int cls; } YoloObbDet;` with
  `int yolo_get_obb(void* h, int slot, YoloObbDet* out, int cap);` mapping like
  ultralytics `scale_boxes(xywh=True)`: `x=(x-left)/scale; y=(y-top)/scale;
  w/=scale; h/=scale;` angle unchanged, no clipping.
- `server/serve.cpp` (uses `YoloDet dets[300]` at serve.cpp:168): refuse obb model dirs
  at startup (`if (yolo_task(eng)) { fprintf(stderr, "server supports detect models only\n"); exit(1); }`).
  Extending the proto with a rotated box message is a separate follow-up.

---

## 5. Verification plan

1. **Level-1 (exporter, automatic in `make export MODEL=yolo11n-obb`)**
   - Existing hooks on layers 0..22 vs decomposed graph, `worst < 1e-4` (backbone/neck are
     unchanged code paths; this catches IMG-parametrization mistakes).
   - New: `assemble_obb(dets)` vs `ref/final.npy` (ultralytics `[1,20,21504]`),
     max abs `< 5e-4` (xywh in pixels up to 1024, scores in [0,1], theta in radians).
2. **Level-2 (per-op)**: `make && make test MODEL=yolo11n-obb` — `dump` + compare.py,
   <3% max-rel on every op including all 33 head convs (11 per level) and the padded
   final convs (pad channels are exact zeros+bias on both sides).
3. **End-to-end vs ultralytics.** Reference recipe (all verified working in this repo):
   ```python
   import numpy as np, torch, cv2
   from ultralytics import YOLO
   from ultralytics.utils import nms as unms
   m = YOLO('yolo11n-obb.pt')
   img = cv2.imread('boats.jpg')                     # https://ultralytics.com/images/boats.jpg
   h, w = img.shape[:2]; r = min(1024/h, 1024/w)
   nh, nw = round(h*r), round(w*r)
   sq = np.full((1024,1024,3), 114, np.uint8)
   top, left = (1024-nh)//2, (1024-nw)//2
   sq[top:top+nh, left:left+nw] = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
   x = torch.from_numpy(sq[:,:,::-1].astype(np.float32).transpose(2,0,1)[None].copy()/255.)
   model = m.model.cpu().float().eval(); model.fuse()
   y = model(x); y0 = y[0] if isinstance(y,(tuple,list)) else y
   kept = unms.non_max_suppression(y0, 0.25, 0.45, nc=15, rotated=True)[0]
   # rows: [cx, cy, w, h, conf, cls, theta] in 1024-letterbox coords
   ```
   Ground truth on boats.jpg: **1346 candidates > 0.25, 169 kept boxes** (top score 0.847,
   cls 1 'ship'). Equivalent high-level call (returns the same 169 rows, reordered
   columns `[x,y,w,h,theta,conf,cls]` in `res[0].obb.data`):
   `m.predict(sq, imgsz=1024, conf=0.25, iou=0.45, verbose=False)` — pass the *square*
   array so ultralytics' LetterBox is an identity and coords are directly comparable.
   Engine side: `./yolo11cuda detect build/yolo11n-obb --image <sq.png>` (square input =>
   scale 1, pad 0, printed coords are letterbox coords).
   **Matching criterion** (fp16 network, fp32-vs-fp16 flips near thresholds):
   greedy-match by class with center distance < 2 px; require |dw|,|dh| < 3 px,
   |dtheta| < 0.02 rad, |dscore| < 0.02; >= 95% of ultralytics detections matched and
   engine det count within +-5%. Exact equality is not achievable (per-op tolerance is
   already 3%); the tail of low-score boxes near conf 0.25 and probiou values near 0.45
   are the expected discrepancies.
4. **Regression**: `make export MODEL=yolo11n && make test MODEL=yolo11n` (v2 graph,
   detect path must be numerically unchanged) and
   `./yolo11cuda bench build/yolo11n 300` cuda-graph line at ~0.90 ms +-3%
   (none of the conv kernels change; DECODE only swaps a literal 80 for net.nc=80).
5. Batched self-check: `./yolo11cuda detect build/yolo11n-obb --batch 4` — the built-in
   per-image consistency print must show identical counts.

---

## 6. Risks / gotchas

- **Unaligned Cout is a hard crasher, not a tolerance issue**: cv3 final (15) and cv4
  final (1) hit `k_conv_mma`'s half2 epilogue (`Bias[n+1]` OOB read + 2-half store) —
  padding to 16/8 (section 4.2) is mandatory. Pad biases must be 0.0 (compare.py
  normalizes by ref max; a -1e4 sentinel would hide real errors) and the decode kernel
  must guard `c0+i < nc` when scanning the padded cls view.
- **NMS_CAP**: detect's 1024 cap silently drops candidates on OBB (boats.jpg: 1346 at
  conf 0.25; dense DOTA tiles go higher). Use OBB_NMS_CAP 2048 + global cov scratch
  (smem cannot hold 2048x5 box floats). Ultralytics caps at max_nms=30000 — on extremely
  dense scenes (>2048 candidates) counts may diverge; also `Net::MAXDET` (4096) bounds the
  decode stage. Acceptable for now; note in comparisons.
- **Fast-NMS, not greedy**: ultralytics rotated NMS suppresses with the full upper-tri
  matrix (a suppressed box still kills later boxes) and uses `>=`. Do NOT copy k_nms's
  greedy dead-check loop; replicate fast-NMS or box counts will differ on chains of
  overlapping boxes.
- **probiou eps placement**: eps=1e-7 appears in t1/t2 denominators, in the sqrt
  denominator of t3, and added to the log argument — but NOT in the t3 numerator. bd is
  clamped to [eps, 100]. Get this exactly right or near-threshold pairs flip.
- **Angle pipeline order**: `(sigmoid-0.25)*pi` on raw logits, before dist2rbox; theta is
  used for the center rotation only — w,h are plain `l+r`,`t+b`; stride scales xywh but
  never theta; output angle channel is the transformed theta (no extra sigmoid). Range
  [-pi/4, 3pi/4). No regularize_rboxes in this ultralytics version's predict path.
- **iou default mismatch**: ultralytics predict defaults to iou=0.7; the engine uses
  0.45. Always pass iou=0.45 explicitly when comparing (or you'll chase phantom extra
  boxes).
- **Class handling**: replicate best-class-only + per-class suppression. Ultralytics does
  class separation by adding `cls*7680` to centers only — a `cls_i == cls_j` test is
  equivalent (probiou of far-separated Gaussians hits the bd=100 clamp -> iou ~ 0).
- **ATTN at 1024 input**: token count N = 32x32 = 1024; the `N == 400` fast-path guard
  (engine.cu:925) must be relaxed or OBB falls to the slow batch=1-only fallback and
  `--batch N` exits. attnP self-sizes to 4 MB from the graph.
- **Ordering nondeterminism**: candidate order comes from atomicAdd; sort is by score
  only, ultralytics argsort is also unstable — equal-score ties may order differently
  and (rarely) change which of two mutually-overlapping boxes survives. Keep tolerance
  set-based, not index-based.
- **Batch dim**: DECODEOBB uses per-image counters and k_nms_obb is one block per image —
  same pattern as detect; the 7-float record stride must be applied consistently in
  dets/outdets/h_out/fetchNMS/yolo_get_obb (a missed 6->7 shows up as garbage cls/angle,
  not a crash).
- **Graph v2**: engine must keep parsing v1 (default task=detect, nc=80, imgsz from
  buffers) or existing build dirs break; `make test MODEL=yolo11n` re-exports to v2 —
  run the detect regression after the exporter change, not before.
- **Buffer sizes**: activations scale by (1024/640)^2 = 2.56x vs detect-n (input buffer
  alone is 6.3 MB/image); per-op ref/gpu dumps grow the same factor — fine on this box,
  but `--batch` memory scales linearly on top.
- **Server**: proto `Box` is axis-aligned and `YoloDet` has no angle — gate obb models
  out of yolo11serve (yolo_task accessor) rather than silently returning wrong geometry;
  proto extension is follow-up work.
- **s/m scales**: only ch/c3/c4 widths change (formulas in section 1) and all stay %8;
  the two padded convs are needed at every scale. Derive everything from modules —
  the briefing forbids per-scale hardcoding. Note `yolo11s-obb`/`yolo11m-obb` also train
  at imgsz=1024 (DOTA), so IMG must come from `model.args`, not a constant.
