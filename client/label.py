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
    ap.add_argument('-c', '--concurrency', type=int, default=64,
                    help='in-flight requests (feed the batcher!)')
    ap.add_argument('--out', default=None, help='write results as JSONL')
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
            r = stub.Detect(pb.DetectRequest(image=data))
            lat.append((time.perf_counter() - t0) * 1000)
            batches.append(r.batch)
        stub.Detect(pb.DetectRequest(image=data))  # warm
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
        r = stub.Detect(pb.DetectRequest(image=open(path, 'rb').read()))
        results[path] = [dict(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, score=b.score, cls=b.cls)
                         for b in r.boxes]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(label, args.images))
    dt = time.perf_counter() - t0
    print(f"labeled {len(args.images)} images in {dt:.2f}s ({len(args.images)/dt:,.0f} img/s)")
    if args.out:
        with open(args.out, 'w') as f:
            for p, boxes in results.items():
                f.write(json.dumps({'image': p, 'boxes': boxes}) + '\n')
        print(f"wrote {args.out}")

if __name__ == '__main__':
    main()
