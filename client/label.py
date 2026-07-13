#!/usr/bin/env python3
"""Mass-labeling client for yolo11serve.

  python3 client/label.py imgs/*.jpg --out labels.jsonl        # label files -> JSONL
  python3 client/label.py bus.jpg --bench 2000 -c 64           # throughput/latency benchmark

Requires: pip install grpcio grpcio-tools
"""
import argparse, json, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

def make_stubs():
    """Generate proto stubs at runtime so the repo ships no generated code."""
    from grpc_tools import protoc
    root = os.path.dirname(os.path.abspath(__file__)) + '/..'
    out = tempfile.mkdtemp(prefix='yolopb_')
    protoc.main(['protoc', f'-I{root}/proto', f'--python_out={out}', f'--grpc_python_out={out}',
                 f'{root}/proto/yolo.proto'])
    sys.path.insert(0, out)
    import yolo_pb2, yolo_pb2_grpc
    return yolo_pb2, yolo_pb2_grpc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--server', default='localhost:50051')
    ap.add_argument('--model', default='', help='model name on multi-model servers (dir basename)')
    ap.add_argument('-c', '--concurrency', type=int, default=64,
                    help='in-flight requests (feed the batcher!)')
    ap.add_argument('--out', default=None, help='write results as JSONL')
    ap.add_argument('--coco', default=None,
                    help='write results as COCO annotations JSON (boxes; polygons for seg/obb)')
    ap.add_argument('--bench', type=int, default=0,
                    help='benchmark: send the first image N times')
    args = ap.parse_args()

    import grpc
    pb, pbg = make_stubs()
    chan = grpc.insecure_channel(args.server, [('grpc.max_receive_message_length', 64 << 20)])
    stub = pbg.YoloStub(chan)

    if args.bench:
        data = open(args.images[0], 'rb').read()
        lat, batches = [], []
        def one(_):
            t0 = time.perf_counter()
            r = stub.Detect(pb.DetectRequest(image=data, model=args.model))
            lat.append((time.perf_counter() - t0) * 1000)
            batches.append(r.batch)
        stub.Detect(pb.DetectRequest(image=data, model=args.model))  # warm
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as ex:
            list(ex.map(one, range(args.bench)))
        dt = time.perf_counter() - t0
        lat.sort()
        n = len(lat)
        print(f"{n} imgs in {dt:.2f}s = {n/dt:,.0f} img/s   "
              f"latency p50={lat[n//2]:.1f} p95={lat[int(n*.95)]:.1f} p99={lat[int(n*.99)]:.1f} ms   "
              f"avg batch={sum(batches)/n:.1f}")
        return

    results = {}
    def label(path):
        r = stub.Detect(pb.DetectRequest(image=open(path, 'rb').read(), model=args.model))
        out = {'task': r.task or 'detect'}
        if r.classes:
            out['classes'] = [dict(id=c.id, prob=c.prob) for c in r.classes]
        elif r.rboxes:
            out['rboxes'] = [dict(cx=b.cx, cy=b.cy, w=b.w, h=b.h, angle=b.angle,
                                  score=b.score, cls=b.cls) for b in r.rboxes]
        else:
            out['boxes'] = [dict(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, score=b.score, cls=b.cls)
                            for b in r.boxes]
            if r.masks:
                out['mask_h'], out['mask_w'] = r.mask_h, r.mask_w
                out['letterbox'] = dict(scale=r.lb_scale, top=r.lb_top, left=r.lb_left)
                out['masks'] = [list(m.rle) for m in r.masks]
        results[path] = out
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(label, args.images))
    dt = time.perf_counter() - t0
    print(f"labeled {len(args.images)} images in {dt:.2f}s ({len(args.images)/dt:,.0f} img/s)")
    if args.coco:
        write_coco(results, args.coco)
    if args.out:
        with open(args.out, 'w') as f:
            for p, res in results.items():
                f.write(json.dumps({'image': p, **res}) + '\n')
        print(f"wrote {args.out}")

# ---------------- COCO export ----------------

COCO80 = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
          "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
          "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
          "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup "
          "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut "
          "cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
          "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
          "hair_drier toothbrush").split()

DOTA15 = ("plane ship storage_tank baseball_diamond tennis_court basketball_court "
          "ground_track_field harbor bridge large_vehicle small_vehicle helicopter roundabout "
          "soccer_ball_field swimming_pool").split()

def rle_decode(rle, h, w):
    import numpy as np
    m = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for run in rle:
        if val: m[pos:pos + run] = 1
        pos += run
        val ^= 1
    return m.reshape(h, w)

def mask_polygons(mask, lb, eps=1.5):
    """binary mask at proto res -> polygons in original image coords (needs cv2)"""
    import cv2, numpy as np
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cont:
        c = cv2.approxPolyDP(c, eps / 4.0, True)   # eps in input px; contour is at proto res (/4)
        if len(c) < 3: continue
        # mask pixel (mx,my) covers letterbox [4mx,4mx+4): take pixel centers, map to original
        pts = (c.reshape(-1, 2).astype('float64') * 4 + 2)
        pts[:, 0] = (pts[:, 0] - lb['left']) / lb['scale']
        pts[:, 1] = (pts[:, 1] - lb['top']) / lb['scale']
        polys.append([round(v, 2) for v in pts.ravel().tolist()])
    return polys

def write_coco(results, path):
    import math
    images, anns = [], []
    aid = 1
    for iid, (p, res) in enumerate(sorted(results.items()), 1):
        images.append(dict(id=iid, file_name=os.path.basename(p)))
        if 'classes' in res:      # classification: no COCO detection annotations
            continue
        for j, b in enumerate(res.get('boxes', [])):
            a = dict(id=aid, image_id=iid, category_id=b['cls'], iscrowd=0,
                     score=round(b['score'], 4),
                     bbox=[round(b['x1'], 2), round(b['y1'], 2),
                           round(b['x2'] - b['x1'], 2), round(b['y2'] - b['y1'], 2)],
                     area=round((b['x2'] - b['x1']) * (b['y2'] - b['y1']), 2))
            if 'masks' in res:
                try:
                    m = rle_decode(res['masks'][j], res['mask_h'], res['mask_w'])
                    a['segmentation'] = mask_polygons(m, res['letterbox'])
                except ImportError:
                    a['segmentation'] = dict(counts=res['masks'][j],
                                             size=[res['mask_h'], res['mask_w']])  # raw RLE fallback
            anns.append(a); aid += 1
        for b in res.get('rboxes', []):   # obb: 4-corner polygon + its aabb
            ca, sa = math.cos(b['angle']), math.sin(b['angle'])
            hw, hh = b['w'] / 2, b['h'] / 2
            pts = [(b['cx'] + dx * ca - dy * sa, b['cy'] + dx * sa + dy * ca)
                   for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            anns.append(dict(id=aid, image_id=iid, category_id=b['cls'], iscrowd=0,
                             score=round(b['score'], 4),
                             segmentation=[[round(v, 2) for xy in pts for v in xy]],
                             bbox=[round(min(xs), 2), round(min(ys), 2),
                                   round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)],
                             area=round(b['w'] * b['h'], 2)))
            aid += 1
    names = DOTA15 if any(r.get('task') == 'obb' for r in results.values()) else COCO80
    cats = [dict(id=i, name=n) for i, n in enumerate(names)]
    with open(path, 'w') as f:
        json.dump(dict(images=images, annotations=anns, categories=cats), f)
    print(f"wrote {path} ({len(anns)} annotations)")


if __name__ == '__main__':
    main()
