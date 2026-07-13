#!/usr/bin/env python3
"""End-to-end segment check: engine dets + masks vs ultralytics on the SAME square-640
letterbox input (never predict() on a raw image — its rect letterbox gives a different
proto resolution). Consumes build/<model>/gpu/{dets.f32,masks.u8} written by
`./yolo11cuda detect build/<model> --save-masks`.

Reference masks use process_mask(upsample=False): crop at proto res, threshold logit > 0 —
precisely the engine's semantics (the engine intentionally skips ultralytics'
bilinear-upsample-then-threshold step; boundary-only deviation, see docs/specs/seg.md 2.2).

Gates (spec 5.3): every reference det with score > 0.27 has a (same-class, box IoU > 0.8)
match; matched |dscore| <= 0.02, box corners <= 2 px; binary mask IoU at proto res
min >= 0.95, median >= 0.98.
"""
import os, sys
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import nms, ops

torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
DIR = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/build/yolo11n-seg'
# build dir name is <model> or <model>-<S>; the .pt name is the part before a trailing -<S>
MODEL = os.path.basename(DIR.rstrip('/'))
parts = MODEL.rsplit('-', 1)
if len(parts) == 2 and parts[1].isdigit():
    MODEL = parts[0]

m = YOLO(f'{ROOT}/{MODEL}.pt')
model = m.model.float().eval()
model.fuse()

# same input tensor the engine consumed
x16 = np.fromfile(DIR + '/input.f16', np.float16)
S = int(round((x16.size / 3) ** 0.5))
x = torch.from_numpy(x16.astype(np.float32).reshape(S, S, 3).transpose(2, 0, 1)[None].copy())

(y, proto), _ = model(x)
# conf/iou match the engine's k_decode / k_nms constants; multi_label=False (best class
# only) is this version's default — same semantics as k_decode
ref = nms.non_max_suppression(y, conf_thres=0.25, iou_thres=0.45, nc=len(m.names))[0]  # [N, 38]
rmasks = ops.process_mask(proto[0], ref[:, 6:], ref[:, :4], (S, S), upsample=False)    # [N,Ph,Pw]
rmasks = rmasks.numpy().astype(np.uint8)
Ph, Pw = rmasks.shape[1:]
# ultralytics drops dets whose final mask is empty (construct_result keep filter)
nonempty = rmasks.reshape(len(ref), Ph * Pw).max(1) > 0   # explicit dim: len(ref) may be 0
ref, rmasks = ref[nonempty].numpy(), rmasks[nonempty]

# engine side
gd = np.fromfile(DIR + '/gpu/dets.f32', np.float32).reshape(-1, 6)      # 640-space letterbox
gm = np.fromfile(DIR + '/gpu/masks.u8', np.uint8).reshape(-1, Ph, Pw)
assert len(gd) == len(gm), (len(gd), len(gm))
print(f'ref dets={len(ref)} engine dets={len(gd)} proto={Ph}x{Pw}')
if len(ref) == 0 or len(gd) == 0:   # zero survivors on both sides is agreement; one-sided is not
    ok = len(ref) == len(gd)
    print('SEG E2E', 'PASS (no detections on either side)' if ok
          else 'FAIL (detections on one side only)')
    sys.exit(0 if ok else 1)

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)

used = set()
ious, fails = [], 0
for i in range(len(ref)):
    rb, rs, rc = ref[i, :4], ref[i, 4], int(ref[i, 5])
    best, bj = 0.0, -1
    for j in range(len(gd)):
        if j in used or int(gd[j, 5]) != rc:
            continue
        iou = box_iou(rb, gd[j, :4])
        if iou > best:
            best, bj = iou, j
    if bj < 0 or best <= 0.8:
        tag = 'FAIL' if rs > 0.27 else 'ok (borderline conf, unmatched)'
        fails += rs > 0.27
        print(f'ref {i}: cls={rc} score={rs:.4f} UNMATCHED  {tag}')
        continue
    used.add(bj)
    ds = abs(rs - gd[bj, 4])
    db = np.abs(rb - gd[bj, :4]).max()
    inter = (rmasks[i] & gm[bj]).sum()
    union = (rmasks[i] | gm[bj]).sum()
    miou = inter / max(union, 1)
    ious.append(miou)
    ok = ds <= 0.02 and db <= 2.0 and miou >= 0.95
    fails += not ok
    print(f'ref {i}: cls={rc} score={rs:.4f} -> eng {bj}: dscore={ds:.4f} dbox={db:.2f}px '
          f'mask IoU={miou:.4f} ({int(rmasks[i].sum())} vs {int(gm[bj].sum())} px) '
          f'{"ok" if ok else "FAIL"}')
ious = np.array(ious)
if len(ious):
    print(f'mask IoU: min={ious.min():.4f} median={np.median(ious):.4f}')
    fails += np.median(ious) < 0.98
ok = fails == 0 and len(ious) > 0
print('SEG E2E', 'PASS' if ok else f'FAIL ({fails} gate failures)')
sys.exit(0 if ok else 1)
