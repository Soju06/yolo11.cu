#!/usr/bin/env python3
"""Compare CUDA per-op dumps (build/gpu/opNNN.bin fp16) vs torch refs (build/ref/opNNN.npy fp32)."""
import glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else ROOT + '/build/yolo11n'
refs = sorted(glob.glob(DIR + '/ref/op*.npy'))
bad = 0
worst = (0.0, None)
for r in refs:
    k = os.path.basename(r)[2:5]
    gp = DIR + f'/gpu/op{k}.bin'
    if not os.path.exists(gp):
        print(f'op{k}: MISSING gpu dump'); bad += 1; continue
    ref = np.load(r)                      # H,W,C fp32
    gpu = np.fromfile(gp, np.float16).astype(np.float32)
    if gpu.size != ref.size:
        print(f'op{k}: SIZE mismatch gpu={gpu.size} ref={ref.size}'); bad += 1; continue
    gpu = gpu.reshape(ref.shape)
    d = np.abs(gpu - ref)
    scale = np.abs(ref).max() + 1e-6
    rel = d.max() / scale
    if rel > worst[0]: worst = (rel, k)
    status = 'ok' if rel < 0.03 else 'FAIL'
    if status == 'FAIL' or '-v' in sys.argv:
        print(f'op{k}: max_abs={d.max():.4e} max={ref.max():.3e} rel={rel:.4e} {status}')
    if status == 'FAIL': bad += 1
print(f'--- {len(refs)} ops, {bad} failures, worst rel={worst[0]:.4e} at op{worst[1]}')
