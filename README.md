# yolo11-cuda

순수 수제 CUDA 커널로 만든 YOLO11 추론 엔진. **cuDNN / cuBLAS / TensorRT 불사용.**
YOLO11 n/s/m 스케일 지원 (익스포터가 모델 구조를 자동 유도), 640×640, COCO 80클래스.

기준 하드웨어: RTX 3060 Ti (Ampere SM 8.6, 8GB), CUDA 13.3.

## 결과

`bench` = net+decode+NMS (CUDA Graph 1회 런치), `pipeline` = 원본 이미지 H2D부터 NMS 결과까지.

| | yolo11n | yolo11m |
|---|---|---|
| ultralytics predict 전체 파이프라인 | 14.9 ms | 11.3 ms |
| PyTorch fp16 eager, net만 (cuDNN) | 6.26 ms | 9.57 ms |
| **이 엔진 bench** (하이브리드 누산 + 오토튠) | **0.89 ms** (1118 fps) | **3.88 ms** (258 fps) |
| 〃 `ACC32=1` (순수 fp32 누산) | 0.98 ms | 4.11 ms |
| **이 엔진 pipeline (엔드투엔드)** | **1.21 ms** (828 fps) | 4.07 ms (246 fps) |

정확도(PyTorch fp32 대비 op 단위 최대 상대오차): 기본 하이브리드 n **0.99%** / m **1.3%**,
`ACC32` n 0.45% / m 1.4%, `ACC16` n 1.9% / m 3.0%. 모든 모드에서 bus.jpg / zidane.jpg 검출이
ultralytics와 박스·클래스·점수 일치.

누산 모드: GA10x는 fp32-누산 HMMA가 fp16-누산의 절반 속도라(ncu: math_pipe_throttle 33%),
기본값은 **하이브리드** — fp16 mma로 K=64 윈도우(2 kstep)만 누산하고 fp32 레지스터로 플러시.
순수 fp16 속도의 대부분을 얻으면서 오차는 윈도우 크기로 바운드됩니다.

**시작 시 오토튠**: (64,64) 타일 conv마다 4-워프/8-워프 그리드를 실측해 op별 승자를 기록
(수십 ms, `NOTUNE=1`로 생략). 8-워프는 occupancy를 33→67%로 올려 barrier/wait 스톨을 은폐하지만
일부 형상에선 손해라 — 휴리스틱 대신 측정으로 선택합니다. NMS 결과 D2H도 그래프 안 pinned
복사라 프레임당 호스트 작업은 그래프 런치 1회 + 동기화 1회뿐입니다.

## 빠른 시작

```bash
pip install ultralytics          # 익스포트에만 필요 (런타임 의존성 아님)
make export MODEL=yolo11n        # 가중치 다운로드 → build/yolo11n/{model.graph, weights.f16, ...}
make                             # ./yolo11cuda 빌드
make test MODEL=yolo11n          # op 단위 수치 검증 + 검출 확인

./yolo11cuda detect build/yolo11n --image photo.jpg   # 임의 jpg/png 검출 (원본 좌표로 출력)
./yolo11cuda bench build/yolo11n 300                  # net-only 벤치
./yolo11cuda pipeline build/yolo11n 300               # 엔드투엔드 벤치
./yolo11cuda profile build/yolo11n                    # op별 시간
ACC16=1 ./yolo11cuda bench build/yolo11m              # fp16 누산 모드
```

다른 스케일: `make export MODEL=yolo11m && make test MODEL=yolo11m`.

## 설계

**그래프 익스포트** (`export/export_yolo11.py`)
- ultralytics 모델을 `fuse()`(BN 폴딩) 후 C3k2/C3k/SPPF/C2PSA/Detect를 7종 프리미티브로 분해:
  `CONV / ADD / MAXPOOL5 / UPSAMPLE2 / COPYC / ATTN / DECODE` (n: 112 ops, m: 144 ops)
- 채널 폭·헤드 수는 모듈에서 유도 — 스케일별 하드코딩 없음
- 2단계 검증: ① 분해 그래프를 torch fp32로 재실행 → ultralytics 후크와 대조(~1e-5)
  ② op별 레퍼런스 덤프 → CUDA 커널과 op 단위 자동 대조 (`make test`)

**메모리**: NHWC fp16 + 뷰 텐서(버퍼+채널 오프셋+스트라이드) → 모든 split/concat **제로카피**.

**커널** (`engine/engine.cu`, 단일 파일)
- `k_conv_mma` — implicit GEMM: `mma.sync.m16n8k16` + `cp.async` 멀티스테이지(싱글 sync),
  XOR 스위즐 smem + `ldmatrix`, **밀집-K 패킹**(im2col 좌표 증분 추적, 루프 내 나눗셈 0),
  에필로그 bias+SiLU+residual 퓨전. 타일 (64,64)/(64,32)/(32,32)/(64,128·256스레드)을
  M/Cout/블록수 휴리스틱으로 선택, fp16 누산은 템플릿 경로.
- **어텐션(C2PSA)도 같은 GEMM 커널 재사용**: QK^T는 K행렬을 weight-stride 트릭으로 제자리
  읽기, softmax 후 P×V는 V^T로. 나이브 대비 8배.
- `k_conv0`(Cin=3, 16채널 청크×blockIdx.y), `k_dwconv8`(depthwise, `[K²][C]` 리패킹+uint4),
  `k_maxpool5`(`__hmax2`), `k_decode`(DFL+sigmoid+conf, uint4), `k_preprocess`(BGR u8→letterbox
  bilinear→fp16, cv2 일치), `k_nms`(단일 블록 bitonic+greedy, ultralytics와 동일 결과)
- 전체 forward + decode + NMS를 **CUDA Graph** 1회 런치로. 호스트로는 최종 박스만 D2H.

## 엔지니어링 노트 (ncu 기반)

- ncu SpeedOfLight: 큰 conv는 **Compute 37% / Memory 78-83%** — L2 대역폭 바운드.
  원인은 im2col A-타일이 n-블록 수만큼 재읽히는 것 → 넓은 N 타일(64,128)로 완화.
- (64,128) 적용 후 재측정한 워프 스톨: math_pipe_throttle 33% / barrier 20% / wait 20% /
  long_scoreboard(글로벌 메모리) **1.7%** — 더 이상 메모리 바운드가 아님. 이 데이터로
  space-to-depth(s2 섹터 낭비 해소)는 무의미해져 착수 전에 기각.
- 시도 후 기각(측정 근거): ① 2-패스 explicit im2col — dense A(29MB)가 L2(4MB)를 넘어 DRAM
  왕복이 되며 역효과(4.08→4.31ms). ② (64,256)·STAGES=2 — 1×1 conv에서 얕은 파이프라인으로
  2.3배 손해. ③ `cp.async.ca`(L1 캐싱) — smem이 L1 파티션을 잠식해 무효과.
  ④ mbarrier 기반 `cuda::pipeline`(arrive/wait 분리로 barrier 스톨 제거 시도) — 스테이지당
  mbarrier 왕복 오버헤드가 절감보다 커서 역효과(n 0.99→1.11ms, m 4.10→4.75ms).
- 남은 여지: barrier+wait 스톨 40%는 커널 구조상 kstep 동기화 비용 — STAGES=5 페어링은
  smem 증가로 occupancy가 반토막이라 손해(분석상). Hopper+ 타깃이면 TMA + wgmma(async mma)로
  로더·동기화 재작성이 정답.

## 레이아웃

```
engine/engine.cu          엔진 전체 (그래프 실행기 + 커널 + CLI)
export/export_yolo11.py   그래프/가중치 익스포터 + 레퍼런스 생성
test/compare.py           op 단위 수치 비교
third_party/stb_image.h   jpg/png 로더 (vendored)
build/<model>/            익스포트 산출물 (git 미추적)
```
