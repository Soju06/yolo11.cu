// YOLO11n hand-written CUDA inference engine.
// Layout: NHWC fp16 with strided channel views (zero-copy split/concat).
// Build: make   |   Run: ./engine {dump|detect|bench N}
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "../third_party/stb_image.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// NMS candidate caps: detect fits 5 box arrays in smem at 1024; OBB scenes are denser
// (boats.jpg: 1346 candidates at conf 0.25) so the cap doubles and box/cov data lives in
// a global scratch buffer instead.
#define NMS_CAP 1024
#define OBB_NMS_CAP 2048
#define MAX_OUT 300

// ------------------------- graph structures -------------------------
struct Buf  { int H, W, C; __half* p = nullptr; };
struct Ten  { int buf, coff, C; };
enum OpKind { CONV, ADD, MAXPOOL5, UPSAMPLE2, COPYC, ATTN, DECODE, GAP, SOFTMAX, DECODEOBB,
              PS2, DECODESEG, MASKS };
struct Op {
  OpKind kind;
  int a = -1, b = -1, out = -1;        // tensor ids (CONV: a=in, out; ADD: a,b,out)
  int k = 0, s = 0, p = 0, g = 0, act = 0, Cin = 0, Cout = 0;
  long woff = 0, boff = 0;
  int heads = 0, kd = 0, hd = 0;       // ATTN
  int variant = 0;                      // autotuned warp grid (0=heuristic, 1=2x2, 2=2x4)
  int stride = 0;                       // DECODE
};

struct Net {
  int B = 1;                   // max batch size (buffers sized for this)
  int inH = 0, inW = 0;        // input size (graph tensor 0; v2 header cross-checks it)
  int nc = 80;                 // class count (v2 header; v1 default)
  char task[16] = "detect";    // v2 header; v1 default
  int cls = 0;                 // task == "classify"
  int obb = 0;                 // task == "obb": 7-float rotated dets, probiou NMS
  int seg = 0;                 // task == "segment": 6+nm-float candidates, post-NMS masks
  int nm = 0;                  // mask coefficient count (MASKS op; 0 = no masks)
  int protoTen = -1;           // segment: proto tensor named by the MASKS op
  int detStride = 6;           // floats per CANDIDATE record (obb 7, segment 6+nm)
  int detK() const { return obb ? 7 : 6; }   // floats per OUTPUT det record
  int probsTen = -1;           // classify: SOFTMAX output tensor (standalone [1,1,nc] buffer)
  __half* h_probs = nullptr;   // pinned host probs [B][nc] (copied inside the graph)
  std::vector<Buf> bufs;
  std::vector<Ten> tens;
  std::vector<Op> ops;
  __half* weights = nullptr;   // device
  float*  bias = nullptr;      // device
  float*  dets = nullptr;      // device candidate buffer [maxdet*6]
  int*    detcnt = nullptr;    // device counter
  float*  outdets = nullptr;   // device NMS output [MAX_OUT*6]
  int*    outcnt = nullptr;
  int*    outidx = nullptr;    // NMS survivor -> candidate slot [B][MAX_OUT]
  uint8_t* masks = nullptr;    // segment: survivor masks [B][MAX_OUT][Ph][Pw], device only
  float*  h_out = nullptr;     // pinned host mirrors (copied inside the graph)
  int*    h_cnt = nullptr;
  int*    h_rawcnt = nullptr;  // pre-NMS candidate count (NMS_CAP truncation tripwire)
  float*  nmsCov = nullptr;    // obb: covariance scratch [B][OBB_NMS_CAP][5] (x,y,A,B,C)
  __half* attnP = nullptr;     // attention probs [heads][N][N]
  __half* vt = nullptr;        // V transposed [heads][hd][N]
  float*  zerobias = nullptr;  // zero bias for GEMM-as-conv calls
  static const int MAXDET = 4096;
};

static std::vector<char> readFile(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> v(n); size_t rd = fread(v.data(), 1, n, f);
  (void)rd; fclose(f); return v;
}

static void loadGraph(Net& net, const std::string& dir) {
  FILE* f = fopen((dir + "/model.graph").c_str(), "r");
  if (!f) { fprintf(stderr, "no model.graph\n"); exit(1); }
  char magic[64]; int ver;
  char tag[32];
  if (fscanf(f, "%63s %d", magic, &ver) != 2) exit(1);
  if (ver >= 2) {  // v2: 'TASK <task> <nc> <imgsz>' line; v1 defaults detect/80/640
    int hsz;
    if (fscanf(f, "%31s %15s %d %d", tag, net.task, &net.nc, &hsz) != 4 || strcmp(tag, "TASK")) {
      fprintf(stderr, "bad TASK header\n"); exit(1);
    }
    net.inH = net.inW = hsz;
  }
  int nb, nt, no;
  if (fscanf(f, "%d %d %d", &nb, &nt, &no) != 3) exit(1);
  net.bufs.resize(nb); net.tens.resize(nt);
  for (int i = 0; i < nb; i++) { Buf& b = net.bufs[i]; if (fscanf(f, "%31s %d %d %d", tag, &b.H, &b.W, &b.C) != 4) exit(1); }
  for (int i = 0; i < nt; i++) { Ten& t = net.tens[i]; if (fscanf(f, "%31s %d %d %d", tag, &t.buf, &t.coff, &t.C) != 4) exit(1); }
  for (int i = 0; i < no; i++) {
    if (fscanf(f, "%31s", tag) != 1) exit(1);
    Op op;
    if (!strcmp(tag, "CONV")) {
      op.kind = CONV;
      if (fscanf(f, "%d %d %d %d %d %d %d %d %d %ld %ld", &op.a, &op.out, &op.k, &op.s, &op.p, &op.g, &op.act, &op.Cin, &op.Cout, &op.woff, &op.boff) != 11) exit(1);
    } else if (!strcmp(tag, "ADD")) {
      op.kind = ADD; if (fscanf(f, "%d %d %d", &op.a, &op.b, &op.out) != 3) exit(1);
    } else if (!strcmp(tag, "MAXPOOL5")) {
      op.kind = MAXPOOL5; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else if (!strcmp(tag, "UPSAMPLE2")) {
      op.kind = UPSAMPLE2; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else if (!strcmp(tag, "COPYC")) {
      op.kind = COPYC; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else if (!strcmp(tag, "ATTN")) {
      op.kind = ATTN; if (fscanf(f, "%d %d %d %d %d", &op.a, &op.out, &op.heads, &op.kd, &op.hd) != 5) exit(1);
    } else if (!strcmp(tag, "DECODE")) {
      op.kind = DECODE; if (fscanf(f, "%d %d %d", &op.a, &op.b, &op.stride) != 3) exit(1);
    } else if (!strcmp(tag, "DECODEOBB")) {
      op.kind = DECODEOBB;   // a = box view, b = padded cls view, out = padded angle view
      if (fscanf(f, "%d %d %d %d", &op.a, &op.b, &op.out, &op.stride) != 4) exit(1);
    } else if (!strcmp(tag, "PS2")) {
      op.kind = PS2; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else if (!strcmp(tag, "DECODESEG")) {
      op.kind = DECODESEG;   // a = box view, b = cls view, out = mask-coef view
      if (fscanf(f, "%d %d %d %d", &op.a, &op.b, &op.out, &op.stride) != 4) exit(1);
    } else if (!strcmp(tag, "MASKS")) {
      op.kind = MASKS;       // a = proto view; assembly runs post-NMS in forward()
      if (fscanf(f, "%d", &op.a) != 1) exit(1);
    } else if (!strcmp(tag, "GAP")) {
      op.kind = GAP; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else if (!strcmp(tag, "SOFTMAX")) {
      op.kind = SOFTMAX; if (fscanf(f, "%d %d", &op.a, &op.out) != 2) exit(1);
    } else { fprintf(stderr, "bad op %s\n", tag); exit(1); }
    net.ops.push_back(op);
  }
  fclose(f);

  // input dims: tensor 0's buffer is the source of truth; the header is a consistency check
  const Buf& ib0 = net.bufs[net.tens[0].buf];
  if (net.inH && (net.inH != ib0.H || net.inW != ib0.W)) { fprintf(stderr, "imgsz header mismatch\n"); exit(1); }
  net.inH = ib0.H; net.inW = ib0.W;

  // fail fast on the class-count invariants k_decode relies on (vec8 cls loads, cls view
  // width == nc); the obb/cls specs own lifting nc % 8
  for (const Op& op : net.ops)
    if (op.kind == DECODE && (net.nc % 8 || net.tens[op.b].C != net.nc)) {
      fprintf(stderr, "DECODE needs nc %% 8 == 0 and cls view C == nc (nc=%d, C=%d)\n",
              net.nc, net.tens[op.b].C); exit(1);
    }

  net.cls = !strcmp(net.task, "classify");
  net.obb = !strcmp(net.task, "obb");
  net.seg = !strcmp(net.task, "segment");
  // obb: cls/angle views are zero-padded to %8 (k_conv_mma epilogue); decode reads nc/1 real
  // channels out of them
  for (const Op& op : net.ops)
    if (op.kind == DECODEOBB &&
        (!net.obb || net.tens[op.b].C % 8 || net.tens[op.b].C < net.nc || net.tens[op.out].C % 8)) {
      fprintf(stderr, "DECODEOBB needs task=obb and %%8-padded cls/angle views covering nc=%d\n",
              net.nc); exit(1);
    }
  // segment: the MASKS op names the proto tensor and fixes nm = proto channel count
  for (const Op& op : net.ops)
    if (op.kind == MASKS) { net.protoTen = op.a; net.nm = net.tens[op.a].C; }
  for (const Op& op : net.ops)
    if (op.kind == DECODESEG &&
        (!net.seg || net.protoTen < 0 || net.nc % 8 || net.tens[op.b].C != net.nc ||
         net.tens[op.out].C != net.nm || net.nm != 32)) {   // k_decode<32> is the only NM inst.
      fprintf(stderr, "DECODESEG needs task=segment, a MASKS op, nc %% 8 == 0 and nm == 32 "
              "(nc=%d nm=%d coef C=%d)\n", net.nc, net.nm, net.tens[op.out].C); exit(1);
    }
  net.detStride = net.obb ? 7 : 6 + net.nm;
  if (net.cls) {
    for (const Op& op : net.ops)
      if (op.kind == SOFTMAX) net.probsTen = op.out;
    if (net.probsTen < 0) { fprintf(stderr, "classify graph without SOFTMAX\n"); exit(1); }
    // the D2H copy and k_softmax_ch index images at buffer stride: must be a standalone
    // full-width [1,1,nc] buffer (coff 0, view C == buffer C)
    const Ten& t = net.tens[net.probsTen]; const Buf& b = net.bufs[t.buf];
    if (t.coff || t.C != b.C || b.H != 1 || b.W != 1 || t.C != net.nc) {
      fprintf(stderr, "probs tensor must be a standalone [1,1,nc] buffer\n"); exit(1);
    }
    CK(cudaMallocHost(&net.h_probs, (size_t)net.B * net.nc * sizeof(__half)));
  }

  for (auto& b : net.bufs) CK(cudaMalloc(&b.p, (size_t)net.B * b.H * b.W * b.C * sizeof(__half)));
  for (auto& b : net.bufs) CK(cudaMemset(b.p, 0, (size_t)net.B * b.H * b.W * b.C * sizeof(__half)));

  auto wf = readFile(dir + "/weights.f16");
  auto bf = readFile(dir + "/bias.f32");
  // repack depthwise weights [C][K*K] -> [K*K][C] for vectorized channel loads
  {
    __half* w = (__half*)wf.data();
    for (const Op& op : net.ops) {
      if (op.kind != CONV || op.g <= 1) continue;
      int C = op.Cin, KK = op.k * op.k;
      std::vector<__half> t(C * KK);
      for (int c = 0; c < C; c++)
        for (int r = 0; r < KK; r++) t[r * C + c] = w[op.woff + c * KK + r];
      memcpy(w + op.woff, t.data(), t.size() * sizeof(__half));
    }
  }
  CK(cudaMalloc(&net.weights, wf.size())); CK(cudaMemcpy(net.weights, wf.data(), wf.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&net.bias, bf.size()));    CK(cudaMemcpy(net.bias, bf.data(), bf.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&net.dets, (size_t)net.B * Net::MAXDET * net.detStride * sizeof(float)));
  CK(cudaMalloc(&net.detcnt, net.B * sizeof(int)));
  CK(cudaMalloc(&net.outdets, (size_t)net.B * 300 * net.detK() * sizeof(float)));
  CK(cudaMalloc(&net.outcnt, net.B * sizeof(int)));
  CK(cudaMalloc(&net.outidx, (size_t)net.B * MAX_OUT * sizeof(int)));
  if (net.protoTen >= 0) {   // survivor masks at proto resolution, device-resident only
    const Buf& pb = net.bufs[net.tens[net.protoTen].buf];
    CK(cudaMalloc(&net.masks, (size_t)net.B * MAX_OUT * pb.H * pb.W));
  }
  CK(cudaMallocHost(&net.h_out, (size_t)net.B * 300 * net.detK() * sizeof(float)));
  CK(cudaMallocHost(&net.h_cnt, net.B * sizeof(int)));
  CK(cudaMallocHost(&net.h_rawcnt, net.B * sizeof(int)));
  memset(net.h_rawcnt, 0, net.B * sizeof(int));
  if (net.obb)   // k_nms_obb covariance scratch (smem cannot hold 2048x5 floats)
    CK(cudaMalloc(&net.nmsCov, (size_t)net.B * OBB_NMS_CAP * 5 * sizeof(float)));
  // attention scratch sized from the graph (heads varies by model scale)
  int maxheads = 0, maxN = 0, maxhd = 64;
  for (const Op& op : net.ops)
    if (op.kind == ATTN) {
      const Ten& t = net.tens[op.a]; const Buf& b = net.bufs[t.buf];
      maxheads = std::max(maxheads, op.heads);
      maxN = std::max(maxN, b.H * b.W);
      maxhd = std::max(maxhd, op.hd);
    }
  if (maxheads) {
    CK(cudaMalloc(&net.attnP, (size_t)maxheads * maxN * maxN * sizeof(__half)));
    CK(cudaMalloc(&net.vt, (size_t)maxheads * maxhd * maxN * sizeof(__half)));
  }
  // attention scores GEMM has Cout = N: the zero bias must cover N (+1 for the half2 epilogue)
  int zbN = std::max(1024, ((maxN + 7) & ~7) + 8);
  CK(cudaMalloc(&net.zerobias, zbN * sizeof(float)));
  CK(cudaMemset(net.zerobias, 0, zbN * sizeof(float)));
  size_t act = 0;
  for (const Buf& b : net.bufs) act += (size_t)net.B * b.H * b.W * b.C * sizeof(__half);
  printf("graph v%d: task=%s nc=%d input=%dx%d, activations %.1f MB (B=%d)\n",
         ver, net.task, net.nc, net.inH, net.inW, act / 1e6, net.B);
}

// ------------------------- kernels: phase B (tensor core implicit GEMM) -------------------------
// NHWC fp16, mma.sync.m16n8k16, cp.async 3-stage pipeline, xor-swizzled smem.
// GEMM view: C[M=Ho*Wo, N=Cout] = A[M, K=k*k*Cin] * B[N, K]^T ; weights are already [N, K] k-contiguous.
__device__ __forceinline__ float actf(float x, int act) {
  return act == 1 ? x / (1.f + expf(-x)) : x;
}

#define CP_ASYNC16(dst, src) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src))
#define CP_COMMIT asm volatile("cp.async.commit_group;\n")
#define CP_WAIT(n) asm volatile("cp.async.wait_group %0;\n" :: "n"(n))

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// swizzled physical half-offset inside a [rows][32] fp16 tile
__device__ __forceinline__ int swz(int row, int seg) {  // seg: which vec8 (0..3)
  return row * 32 + ((seg ^ (row & 3)) << 3);
}

// Dense-K implicit GEMM: the GEMM K axis is the flat (ky,kx,ci) index with NO per-tap
// padding — vec8 slots individually resolve (ky,kx,ci) so Cin=8/16 layers waste nothing.
// AMODE: 0 = fp32 mma accumulate, 1 = pure fp16 accumulate (fastest, ~2-3% err),
//         2 = hybrid: fp16 mma over a 4-kstep (K=128) window, flushed into fp32
template<int BM, int BN, int STAGES, bool KONE = false, int AMODE = 0,
         int WARPS_M = 2, int WARPS_N = 2>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2) k_conv_mma(
    const __half* __restrict__ X, int H, int W, int xs, int xo,
    __half* __restrict__ Y, int Ho, int Wo, int ys, int yo,
    const __half* __restrict__ Wt, long wstride, const float* __restrict__ Bias,
    const __half* __restrict__ Res, int rs, int ro,
    int Cin, int Cout, int K, int S, int P, int act, int Bn) {
  constexpr int BK = 32;
  constexpr int THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MI = WM / 16, NI = WN / 8;  // mma tiles per warp
  __shared__ __half As[STAGES][BM * BK];
  __shared__ __half Bs[STAGES][BN * BK];

  const int M = Bn * Ho * Wo;
  const int bm = blockIdx.x * BM, bn = blockIdx.y * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wm = (warp % WARPS_M) * WM, wn = (warp / WARPS_M) * WN;
  const int Ktot = K * K * Cin;
  const int ksteps = (Ktot + BK - 1) / BK;

  float acc[MI][NI][4];
  uint32_t hacc[MI][NI][2];   // fp16x2 accumulators (ACC16 path)
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NI; j++) {
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
      hacc[i][j][0] = hacc[i][j][1] = 0u;
    }

  // incremental im2col coordinates for this thread's A seg (advanced once per kstep):
  // per-slot pixel bases hoisted, (ky,kx,ci) tracked incrementally — no divs in the loop.
  const int asegbase = tid & 3;             // seg of slot tid (slots tid, tid+128 share seg)
  int a_ci = (asegbase * 8) % max(Cin, 1);
  int a_r = (asegbase * 8) / max(Cin, 1);
  int a_ky = a_r / K, a_kx = a_r % K;
  int a_kg = asegbase * 8;                  // this seg's flat k index
  constexpr int ASLOTS = (BM * 4 + THREADS - 1) / THREADS;
  int ybase[ASLOTS], xbase[ASLOTS], brow[ASLOTS];
  bool mok[ASLOTS];
#pragma unroll
  for (int i = 0; i < ASLOTS; i++) {
    int m = bm + ((tid + i * THREADS) >> 2);
    mok[i] = (m < M) && (tid + i * THREADS < BM * 4);
    int img = m / (Ho * Wo), rem = m % (Ho * Wo);
    int oy = rem / Wo, ox = rem % Wo;
    ybase[i] = oy * S - P; xbase[i] = ox * S - P;
    brow[i] = img * H;                   // batched row base of this slot's image
  }

  auto loadA = [&](int stage, int ks) {
#pragma unroll
    for (int i = 0; i < ASLOTS; i++) {
      if (tid + i * THREADS >= BM * 4) break;
      int row = (tid + i * THREADS) >> 2;
      uint32_t dst = smem_u32(&As[stage][swz(row, asegbase)]);
      if (KONE) {
        if (mok[i] && a_kg < Ktot) {
          CP_ASYNC16(dst, X + (size_t)(bm + row) * xs + xo + a_kg);
        } else {
          *(uint4*)&As[stage][swz(row, asegbase)] = make_uint4(0, 0, 0, 0);
        }
      } else {
        int iy = ybase[i] + a_ky, ix = xbase[i] + a_kx;
        bool ok = mok[i] && (iy >= 0) && (iy < H) && (ix >= 0) && (ix < W) && (a_kg < Ktot);
        if (ok) {
          CP_ASYNC16(dst, X + ((size_t)(brow[i] + iy) * W + ix) * xs + xo + a_ci);
        } else {
          *(uint4*)&As[stage][swz(row, asegbase)] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    a_kg += BK;
    if (!KONE) {
      a_ci += BK;
      while (a_ci >= Cin) {
        a_ci -= Cin;
        if (++a_kx == K) { a_kx = 0; a_ky++; }
      }
    }
  };
  auto loadB = [&](int stage, int ks) {
#pragma unroll
    for (int slot = tid; slot < BN * 4; slot += THREADS) {
      int row = slot >> 2, seg = slot & 3;
      int n = bn + row;
      int kg = ks * BK + seg * 8;
      uint32_t dst = smem_u32(&Bs[stage][swz(row, seg)]);
      bool ok = (n < Cout) && (kg < Ktot);
      if (ok) {
        CP_ASYNC16(dst, Wt + (size_t)n * wstride + kg);
      } else {
        *(uint4*)&Bs[stage][swz(row, seg)] = make_uint4(0, 0, 0, 0);
      }
    }
  };

  // prologue
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < ksteps) { loadA(s, s); loadB(s, s); }
    CP_COMMIT;
  }

  for (int ks = 0; ks < ksteps; ks++) {
    CP_WAIT(STAGES - 2);
    __syncthreads();
    // issue next-stage loads first (targets a different stage buffer), then compute
    int nk = ks + STAGES - 1;
    if (nk < ksteps) { loadA(nk % STAGES, nk); loadB(nk % STAGES, nk); }
    CP_COMMIT;
    int stage = ks % STAGES;
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {  // two k16 slices inside BK=32
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {
        int row = wm + mi * 16 + (lane & 15);
        int seg = kh * 2 + (lane >> 4);
        uint32_t addr = smem_u32(&As[stage][swz(row, seg)]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
      }
      uint32_t b[NI][2];
#pragma unroll
      for (int nj = 0; nj < NI / 2; nj++) {   // each ldmatrix.x4 covers two n8 tiles
        int j = lane >> 3, rowin = lane & 7;
        int n = wn + nj * 16 + (j >> 1) * 8 + rowin;
        int seg = kh * 2 + (j & 1);
        uint32_t addr = smem_u32(&Bs[stage][swz(n, seg)]);
        uint32_t r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
        b[nj * 2 + 0][0] = r0; b[nj * 2 + 0][1] = r1;
        b[nj * 2 + 1][0] = r2; b[nj * 2 + 1][1] = r3;
      }
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          if (AMODE >= 1)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                : "+r"(hacc[mi][ni][0]), "+r"(hacc[mi][ni][1])
                : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                  "r"(b[ni][0]), "r"(b[ni][1]));
          else
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                  "r"(b[ni][0]), "r"(b[ni][1]));
        }
    }
    if (AMODE == 2 && ((ks & 1) == 1 || ks == ksteps - 1)) {
      // flush the bounded fp16 window into fp32 accumulators
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          float2 lo = __half22float2(*(__half2*)&hacc[mi][ni][0]);
          float2 hi = __half22float2(*(__half2*)&hacc[mi][ni][1]);
          acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y;
          acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
          hacc[mi][ni][0] = hacc[mi][ni][1] = 0u;
        }
    }
  }
  if (AMODE == 1) {   // unpack fp16 accumulators into the float epilogue path
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        float2 lo = __half22float2(*(__half2*)&hacc[mi][ni][0]);
        float2 hi = __half22float2(*(__half2*)&hacc[mi][ni][1]);
        acc[mi][ni][0] = lo.x; acc[mi][ni][1] = lo.y;
        acc[mi][ni][2] = hi.x; acc[mi][ni][3] = hi.y;
      }
  }

  // ---- epilogue: acc -> act -> +res -> fp16 strided store (half2 along n)
#pragma unroll
  for (int mi = 0; mi < MI; mi++) {
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
#pragma unroll
      for (int half_ = 0; half_ < 2; half_++) {  // c pairs (rows +0 / +8)
        int m = bm + wm + mi * 16 + (lane >> 2) + half_ * 8;
        int n = bn + wn + ni * 8 + (lane & 3) * 2;
        if (m >= M || n >= Cout) continue;
        float v0 = acc[mi][ni][half_ * 2 + 0] + Bias[n];
        float v1 = acc[mi][ni][half_ * 2 + 1] + Bias[n + 1];
        if (act == 1) {
          v0 = v0 / (1.f + __expf(-v0));
          v1 = v1 / (1.f + __expf(-v1));
        }
        if (Res) {
          const __half2 rv = *(const __half2*)(Res + (size_t)m * rs + ro + n);
          v0 += __low2float(rv); v1 += __high2float(rv);
        }
        *(__half2*)(Y + (size_t)m * ys + yo + n) = __floats2half2_rn(v0, v1);
      }
    }
  }
}

// first conv: Cin=3, K=3, S=2. Thread computes a 16-channel chunk for one output pixel;
// blockIdx.y selects the chunk (keeps register pressure flat for wide first convs).
template<int CHUNK>
__global__ void k_conv0(const __half* __restrict__ X, int H, int W,
                        __half* __restrict__ Y, int Ho, int Wo, int Cout,
                        const __half* __restrict__ Wt, const float* __restrict__ B, int act, int Bn) {
  const int CIN = 3, K = 3, S = 2, P = 1;
  const int co0 = blockIdx.y * CHUNK;
  __shared__ __half ws[CHUNK * K * K * CIN];   // 432
  __shared__ float bs[CHUNK];
  for (int i = threadIdx.x; i < CHUNK * K * K * CIN; i += blockDim.x) ws[i] = Wt[co0 * K * K * CIN + i];
  for (int i = threadIdx.x; i < CHUNK; i += blockDim.x) bs[i] = B[co0 + i];
  __syncthreads();
  int pix = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= Bn * Ho * Wo) return;
  int rem = pix % (Ho * Wo);
  int ox = rem % Wo, oy = rem / Wo;
  const __half* Xb = X + (size_t)(pix / (Ho * Wo)) * H * W * CIN;
  float acc[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; c++) acc[c] = bs[c];
#pragma unroll
  for (int ky = 0; ky < K; ky++) {
    int iy = oy * S - P + ky;
    if (iy < 0 || iy >= H) continue;
#pragma unroll
    for (int kx = 0; kx < K; kx++) {
      int ix = ox * S - P + kx;
      if (ix < 0 || ix >= W) continue;
      float in[CIN];
      const __half* xp = Xb + ((size_t)iy * W + ix) * CIN;
#pragma unroll
      for (int i = 0; i < CIN; i++) in[i] = __half2float(xp[i]);
#pragma unroll
      for (int c = 0; c < CHUNK; c++) {
        const __half* wq = ws + (c * K * K + ky * K + kx) * CIN;
#pragma unroll
        for (int i = 0; i < CIN; i++) acc[c] += in[i] * __half2float(wq[i]);
      }
    }
  }
  __half out[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; c++) out[c] = __float2half(actf(acc[c], act));
  uint4* dst = (uint4*)(Y + (size_t)pix * Cout + co0);
#pragma unroll
  for (int i = 0; i < CHUNK / 8; i++) dst[i] = ((uint4*)out)[i];
}

// vectorized depthwise conv: thread per (pixel, 8-channel group); weights repacked to [K*K][C]
__global__ void k_dwconv8(const __half* __restrict__ X, int H, int W, int xs, int xo,
                          __half* __restrict__ Y, int Ho, int Wo, int ys, int yo,
                          const __half* __restrict__ Wt, const float* __restrict__ B,
                          const __half* __restrict__ Res, int rs, int ro,
                          int C, int K, int S, int P, int act, int Bn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cg = C >> 3;
  if (idx >= Bn * Ho * Wo * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int rem = pix % (Ho * Wo), brow = (pix / (Ho * Wo)) * H;
  int ox = rem % Wo, oy = rem / Wo;
  float acc[8];
  {
    const float4 b0 = *(const float4*)(B + c), b1 = *(const float4*)(B + c + 4);
    acc[0] = b0.x; acc[1] = b0.y; acc[2] = b0.z; acc[3] = b0.w;
    acc[4] = b1.x; acc[5] = b1.y; acc[6] = b1.z; acc[7] = b1.w;
  }
#pragma unroll 3
  for (int ky = 0; ky < 3; ky++) {
    int iy = oy * S - P + ky;
    if (iy < 0 || iy >= H) continue;
#pragma unroll 3
    for (int kx = 0; kx < 3; kx++) {
      int ix = ox * S - P + kx;
      if (ix < 0 || ix >= W) continue;
      const __half* xp = X + ((size_t)(brow + iy) * W + ix) * xs + xo + c;
      const __half* wq = Wt + (ky * 3 + kx) * C + c;
      uint4 xv = *(const uint4*)xp, wv = *(const uint4*)wq;
      const __half *xh = (const __half*)&xv, *wh = (const __half*)&wv;
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] += __half2float(xh[i]) * __half2float(wh[i]);
    }
  }
  __half out[8];
  if (Res) {
    uint4 rv = *(const uint4*)(Res + (size_t)pix * rs + ro + c);
    const __half* rh = (const __half*)&rv;
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = __float2half(actf(acc[i], act) + __half2float(rh[i]));
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = __float2half(actf(acc[i], act));
  }
  *(uint4*)(Y + (size_t)pix * ys + yo + c) = *(uint4*)out;
}

// row-wise softmax(x * scale) in place; one warp per row
__global__ void k_softmax_rows(__half* __restrict__ P, int rows, int N, float scale) {
  int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  __half* p = P + (size_t)row * N;
  float lmax = -1e30f;
  for (int j = lane; j < N; j += 32) lmax = fmaxf(lmax, __half2float(p[j]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(~0u, lmax, o));
  float lsum = 0;
  for (int j = lane; j < N; j += 32) lsum += __expf((__half2float(p[j]) - lmax) * scale);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(~0u, lsum, o);
  float inv = 1.f / lsum;
  for (int j = lane; j < N; j += 32)
    p[j] = __float2half(__expf((__half2float(p[j]) - lmax) * scale) * inv);
}

// gather V^T: out[h][c][tok] from qkv[tok][h*per + 2kd + c]
__global__ void k_vtrans(const __half* __restrict__ QKV, int qs, int qo,
                         __half* __restrict__ VT, int N, int kd, int hd, int heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * hd * N) return;
  int tok = idx % N, c = (idx / N) % hd, h = idx / (N * hd);
  VT[idx] = QKV[(size_t)tok * qs + qo + h * (2 * kd + hd) + 2 * kd + c];
}

// ------------------------- kernels: phase A (direct) -------------------------
// generic direct conv, one thread per (pixel, cout). NHWC strided.
__global__ void k_conv_direct(const __half* __restrict__ X, int H, int W, int xs, int xo,
                              __half* __restrict__ Y, int Ho, int Wo, int ys, int yo,
                              const __half* __restrict__ Wt, const float* __restrict__ B,
                              int Cin, int Cout, int K, int S, int P, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ho * Wo * Cout) return;
  int co = idx % Cout, pix = idx / Cout;
  int ox = pix % Wo, oy = pix / Wo;
  float acc = B[co];
  const __half* wp = Wt + (size_t)co * K * K * Cin;
  for (int ky = 0; ky < K; ky++) {
    int iy = oy * S - P + ky;
    if (iy < 0 || iy >= H) continue;
    for (int kx = 0; kx < K; kx++) {
      int ix = ox * S - P + kx;
      if (ix < 0 || ix >= W) continue;
      const __half* xp = X + ((size_t)iy * W + ix) * xs + xo;
      const __half* wq = wp + (ky * K + kx) * Cin;
      for (int ci = 0; ci < Cin; ci++)
        acc += __half2float(xp[ci]) * __half2float(wq[ci]);
    }
  }
  Y[((size_t)oy * Wo + ox) * ys + yo + co] = __float2half(actf(acc, act));
}

// depthwise conv (g == Cin == Cout), weight [C, K, K]
__global__ void k_dwconv(const __half* __restrict__ X, int H, int W, int xs, int xo,
                         __half* __restrict__ Y, int Ho, int Wo, int ys, int yo,
                         const __half* __restrict__ Wt, const float* __restrict__ B,
                         const __half* __restrict__ Res, int rs, int ro,
                         int C, int K, int S, int P, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ho * Wo * C) return;
  int c = idx % C, pix = idx / C;
  int ox = pix % Wo, oy = pix / Wo;
  float acc = B[c];
  const __half* wp = Wt + (size_t)c * K * K;
  for (int ky = 0; ky < K; ky++) {
    int iy = oy * S - P + ky;
    if (iy < 0 || iy >= H) continue;
    for (int kx = 0; kx < K; kx++) {
      int ix = ox * S - P + kx;
      if (ix < 0 || ix >= W) continue;
      acc += __half2float(X[((size_t)iy * W + ix) * xs + xo + c]) * __half2float(wp[ky * K + kx]);
    }
  }
  float v = actf(acc, act);
  if (Res) v += __half2float(Res[((size_t)oy * Wo + ox) * rs + ro + c]);
  Y[((size_t)oy * Wo + ox) * ys + yo + c] = __float2half(v);
}

__global__ void k_add(const __half* A, int as, int ao, const __half* Bp, int bs, int bo,
                      __half* Y, int ys, int yo, int HW, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= HW * C) return;
  int c = idx % C, pix = idx / C;
  float v = __half2float(A[(size_t)pix * as + ao + c]) + __half2float(Bp[(size_t)pix * bs + bo + c]);
  Y[(size_t)pix * ys + yo + c] = __float2half(v);
}

__global__ void k_maxpool5(const __half* X, int xs, int xo, __half* Y, int ys, int yo,
                           int H, int W, int C, int Bn) {  // C % 8 == 0, vec8 channels
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cg = C >> 3;
  if (idx >= Bn * H * W * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int rem = pix % (H * W), brow = (pix / (H * W)) * H;
  int x0 = rem % W, y0 = rem / W;
  __half2 m[4];
#pragma unroll
  for (int i = 0; i < 4; i++) m[i] = __floats2half2_rn(-6e4f, -6e4f);
  for (int dy = -2; dy <= 2; dy++) {
    int iy = y0 + dy; if (iy < 0 || iy >= H) continue;
    for (int dx = -2; dx <= 2; dx++) {
      int ix = x0 + dx; if (ix < 0 || ix >= W) continue;
      uint4 v = *(const uint4*)(X + ((size_t)(brow + iy) * W + ix) * xs + xo + c);
      const __half2* h2 = (const __half2*)&v;
#pragma unroll
      for (int i = 0; i < 4; i++) m[i] = __hmax2(m[i], h2[i]);
    }
  }
  *(uint4*)(Y + (size_t)pix * ys + yo + c) = *(uint4*)m;
}

__global__ void k_upsample2(const __half* X, int xs, int xo, __half* Y, int ys, int yo,
                            int H, int W, int C, int Bn) {  // out is 2H x 2W, C % 8 == 0
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Wo = W * 2, Ho = H * 2, cg = C >> 3;
  if (idx >= Bn * Ho * Wo * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int rem = pix % (Ho * Wo), brow = (pix / (Ho * Wo)) * H;
  int ox = rem % Wo, oy = rem / Wo;
  *(uint4*)(Y + (size_t)pix * ys + yo + c) =
      *(const uint4*)(X + ((size_t)(brow + (oy >> 1)) * W + (ox >> 1)) * xs + xo + c);
}

// pixel shuffle r=2 from quadrant channel blocks: in [H,W,4C] -> out [2H,2W,C], C % 8 == 0.
// in(i, j, (2*di+dj)*C + c) -> out(2i+di, 2j+dj, c); with a 1x1 CONV to 4C this implements
// ConvTranspose2d(k=2, s=2) exactly.
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

__global__ void k_copyc(const __half* X, int xs, int xo, __half* Y, int ys, int yo, int HW, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= HW * C) return;
  int c = idx % C, pix = idx / C;
  Y[(size_t)pix * ys + yo + c] = X[(size_t)pix * xs + xo + c];
}

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

// channel softmax on a [1,1,C] view; one block per image; fp32 math, fp16 out.
// not in-place: the dump compares both the logits CONV out and the probs.
__global__ void k_softmax_ch(const __half* __restrict__ X, int xs, int xo,
                             __half* __restrict__ Y, int ys, int yo, int C) {
  const __half* x = X + (size_t)blockIdx.x * xs + xo;   // H*W == 1 -> image stride == xs
  __half* y = Y + (size_t)blockIdx.x * ys + yo;
  __shared__ float red[32];
  const int tid = threadIdx.x;
  float lmax = -1e30f;
  for (int c = tid; c < C; c += blockDim.x) lmax = fmaxf(lmax, __half2float(x[c]));
  for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_down_sync(~0u, lmax, o));
  if ((tid & 31) == 0) red[tid >> 5] = lmax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x + 31) / 32 ? red[tid] : -1e30f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(~0u, v, o));
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  float gmax = red[0];
  __syncthreads();
  float lsum = 0.f;
  for (int c = tid; c < C; c += blockDim.x) lsum += expf(__half2float(x[c]) - gmax);
  for (int o = 16; o > 0; o >>= 1) lsum += __shfl_down_sync(~0u, lsum, o);
  if ((tid & 31) == 0) red[tid >> 5] = lsum;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x + 31) / 32 ? red[tid] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int c = tid; c < C; c += blockDim.x)
    y[c] = __float2half(expf(__half2float(x[c]) - gmax) * inv);
}

// attention: qkv buffer [N tokens, heads*(2kd+hd)] strided; out [N, heads*hd].
// one block per (token, head, image); scores in smem. Correct for any N (mma fallback).
__global__ void k_attn(const __half* __restrict__ QKV, int qs, int qo,
                       __half* __restrict__ Y, int ys, int yo,
                       int N, int heads, int kd, int hd) {
  int n = blockIdx.x, h = blockIdx.y;
  int per = 2 * kd + hd;
  QKV += (size_t)blockIdx.z * N * qs;   // image offset (tokens must not mix across images)
  Y += (size_t)blockIdx.z * N * ys;
  extern __shared__ float sm[];        // scores[N] + red[32]
  float* scores = sm;
  const __half* q = QKV + (size_t)n * qs + qo + h * per;
  float scale = rsqrtf((float)kd);
  // scores[j] = q . k_j
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const __half* k = QKV + (size_t)j * qs + qo + h * per + kd;
    float s = 0;
    for (int d = 0; d < kd; d++) s += __half2float(q[d]) * __half2float(k[d]);
    scores[j] = s * scale;
  }
  __syncthreads();
  // softmax (block reduction)
  __shared__ float red[32];
  float lmax = -1e30f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) lmax = fmaxf(lmax, scores[j]);
  for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_down_sync(~0u, lmax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lmax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : -1e30f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float gmax = red[0];
  __syncthreads();
  float lsum = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float e = expf(scores[j] - gmax); scores[j] = e; lsum += e;
  }
  for (int o = 16; o > 0; o >>= 1) lsum += __shfl_down_sync(~0u, lsum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lsum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  // out[n, h*hd + c] = sum_j p_j * v[j, c]
  for (int c = threadIdx.x; c < hd; c += blockDim.x) {
    float acc = 0;
    for (int j = 0; j < N; j++) {
      const __half* v = QKV + (size_t)j * qs + qo + h * per + 2 * kd;
      acc += scores[j] * __half2float(v[c]);
    }
    Y[(size_t)n * ys + yo + h * hd + c] = __float2half(acc * inv);
  }
}

// DFL decode + score threshold. box view [H,W,64], cls view [H,W,nc].
// NM > 0 (segment): candidate records are 6+NM floats and the NM raw fp16 mask coefficients
// at the same anchor are appended as fp32 (NM % 8 == 0, uint4 loads). NM == 0 dead-strips
// the tail copy and keeps the detect codegen unchanged (MC/ms/mo unused).
template <int NM>
__global__ void k_decode(const __half* __restrict__ BOX, int bs, int bo,
                         const __half* __restrict__ CLS, int cs, int co,
                         const __half* __restrict__ MC, int ms, int mo,
                         int H, int W, int stride, float conf,
                         float* __restrict__ out, int* __restrict__ cnt, int maxdet, int nc, int Bn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Bn * H * W) return;
  int img = idx / (H * W), rem = idx % (H * W);
  int x = rem % W, y = rem / W;
  const __half* cp = CLS + (size_t)idx * cs + co;
  float best = -1e30f; int bcls = 0;
  for (int c0 = 0; c0 < nc; c0 += 8) {         // nc % 8 == 0 (80)
    uint4 v = *(const uint4*)(cp + c0);
    const __half* h = (const __half*)&v;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float f = __half2float(h[i]);
      if (f > best) { best = f; bcls = c0 + i; }
    }
  }
  float score = 1.f / (1.f + expf(-best));
  if (score < conf) return;
  const __half* bp = BOX + (size_t)idx * bs + bo;
  float d[4];
#pragma unroll
  for (int side = 0; side < 4; side++) {
    uint4 v0 = *(const uint4*)(bp + side * 16), v1 = *(const uint4*)(bp + side * 16 + 8);
    float l[16];
    const __half* h0 = (const __half*)&v0; const __half* h1 = (const __half*)&v1;
#pragma unroll
    for (int i = 0; i < 8; i++) { l[i] = __half2float(h0[i]); l[i + 8] = __half2float(h1[i]); }
    float mx = -1e30f;
#pragma unroll
    for (int i = 0; i < 16; i++) mx = fmaxf(mx, l[i]);
    float sum = 0, e = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) { float p = __expf(l[i] - mx); sum += p; e += p * i; }
    d[side] = e / sum;
  }
  float cx = x + 0.5f, cy = y + 0.5f;
  int slot = atomicAdd(cnt + img, 1);
  if (slot >= maxdet) return;
  float* o = out + ((size_t)img * maxdet + slot) * (6 + NM);
  o[0] = (cx - d[0]) * stride; o[1] = (cy - d[1]) * stride;
  o[2] = (cx + d[2]) * stride; o[3] = (cy + d[3]) * stride;
  o[4] = score; o[5] = (float)bcls;
  if (NM > 0) {
    const __half* mp = MC + (size_t)idx * ms + mo;
#pragma unroll
    for (int i = 0; i < NM; i += 8) {
      uint4 v = *(const uint4*)(mp + i);
      const __half* h = (const __half*)&v;
#pragma unroll
      for (int j = 0; j < 8; j++) o[6 + i + j] = __half2float(h[j]);
    }
  }
}

// OBB decode: DFL + (sigmoid-0.25)*pi angle + dist2rbox. box view [H,W,64], cls view
// [H,W,ncpad] (zero-padded: pad logits are exactly 0 -> sigmoid 0.5, so the c0+i < nc
// guard is load-bearing, not cosmetic), angle view [H,W,8] (channel 0 real).
// Candidates are 7 floats: cx, cy, w, h, score, cls, theta (letterbox px / radians).
__global__ void k_decode_obb(const __half* __restrict__ BOX, int bs, int bo,
                             const __half* __restrict__ CLS, int cs, int co,
                             const __half* __restrict__ ANG, int as, int ao,
                             int H, int W, int stride, float conf,
                             float* __restrict__ out, int* __restrict__ cnt, int maxdet,
                             int nc, int ncpad, int Bn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Bn * H * W) return;
  int img = idx / (H * W), rem = idx % (H * W);
  int x = rem % W, y = rem / W;
  const __half* cp = CLS + (size_t)idx * cs + co;
  float best = -1e30f; int bcls = 0;
  for (int c0 = 0; c0 < ncpad; c0 += 8) {      // ncpad % 8 == 0 (vec8 loads)
    uint4 v = *(const uint4*)(cp + c0);
    const __half* h = (const __half*)&v;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float f = __half2float(h[i]);
      if (c0 + i < nc && f > best) { best = f; bcls = c0 + i; }
    }
  }
  float score = 1.f / (1.f + expf(-best));
  if (score < conf) return;
  const __half* bp = BOX + (size_t)idx * bs + bo;
  float d[4];
#pragma unroll
  for (int side = 0; side < 4; side++) {
    uint4 v0 = *(const uint4*)(bp + side * 16), v1 = *(const uint4*)(bp + side * 16 + 8);
    float l[16];
    const __half* h0 = (const __half*)&v0; const __half* h1 = (const __half*)&v1;
#pragma unroll
    for (int i = 0; i < 8; i++) { l[i] = __half2float(h0[i]); l[i + 8] = __half2float(h1[i]); }
    float mx = -1e30f;
#pragma unroll
    for (int i = 0; i < 16; i++) mx = fmaxf(mx, l[i]);
    float sum = 0, e = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) { float p = __expf(l[i] - mx); sum += p; e += p * i; }
    d[side] = e / sum;                          // l, t, r, b in grid units
  }
  float t0 = __half2float(ANG[(size_t)idx * as + ao]);
  float th = (1.f / (1.f + expf(-t0)) - 0.25f) * 3.14159265358979f;
  float xf = (d[2] - d[0]) * 0.5f, yf = (d[3] - d[1]) * 0.5f;
  float cs_ = cosf(th), sn = sinf(th);
  int slot = atomicAdd(cnt + img, 1);
  if (slot >= maxdet) return;
  float* o = out + ((size_t)img * maxdet + slot) * 7;
  o[0] = (x + 0.5f + xf * cs_ - yf * sn) * stride;   // cx
  o[1] = (y + 0.5f + xf * sn + yf * cs_) * stride;   // cy
  o[2] = (d[0] + d[2]) * stride;                     // w (along theta, not rotated)
  o[3] = (d[1] + d[3]) * stride;                     // h
  o[4] = score; o[5] = (float)bcls; o[6] = th;
}

// letterbox preprocess: BGR u8 HWC (sh x sw) -> RGB fp16 NHWC (dh x dw), /255, pad 114.
// bilinear resize matching cv2.INTER_LINEAR.
__global__ void k_preprocess(const uint8_t* __restrict__ src, int sh, int sw,
                             __half* __restrict__ dst, int dh, int dw,
                             float scale, int top, int left, int nh, int nw) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dh * dw) return;
  int x = idx % dw, y = idx / dw;
  float r, g, b;
  if (y < top || y >= top + nh || x < left || x >= left + nw) {
    r = g = b = 114.f;
  } else {
    // cv2 bilinear: src coord = (dst + 0.5) / scale - 0.5
    float fy = (y - top + 0.5f) / scale - 0.5f;
    float fx = (x - left + 0.5f) / scale - 0.5f;
    int y0 = (int)floorf(fy), x0 = (int)floorf(fx);
    float wy = fy - y0, wx = fx - x0;
    int y1 = min(y0 + 1, sh - 1), x1 = min(x0 + 1, sw - 1);
    y0 = max(y0, 0); x0 = max(x0, 0);
    const uint8_t* p00 = src + ((size_t)y0 * sw + x0) * 3;
    const uint8_t* p01 = src + ((size_t)y0 * sw + x1) * 3;
    const uint8_t* p10 = src + ((size_t)y1 * sw + x0) * 3;
    const uint8_t* p11 = src + ((size_t)y1 * sw + x1) * 3;
    float w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx, w10 = wy * (1 - wx), w11 = wy * wx;
    b = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
    g = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
    r = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;
  }
  __half* o = dst + (size_t)idx * 3;
  o[0] = __float2half(r / 255.f);   // RGB order
  o[1] = __float2half(g / 255.f);
  o[2] = __float2half(b / 255.f);
}

// classify preprocess: BGR u8 HWC (sh x sw) -> RGB fp16 NHWC (S x S), /255.
// torchvision path: resize shortest edge to S with PIL-style ANTIALIASED bilinear
// (triangle filter, support widens by the scale on downscale — plain bilinear diverges),
// then center-crop S x S. rh/rw = resized dims, top/left = crop offsets (host-computed).
__global__ void k_preprocess_cls(const uint8_t* __restrict__ src, int sh, int sw,
                                 __half* __restrict__ dst, int S,
                                 int rh, int rw, int top, int left) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S * S) return;
  int x = idx % S, y = idx / S;
  float scy = (float)sh / rh, scx = (float)sw / rw;   // per-axis src/resized scale
  float supy = fmaxf(scy, 1.f), supx = fmaxf(scx, 1.f);
  float cy = (y + top + 0.5f) * scy, cx = (x + left + 0.5f) * scx;
  int y0 = max(0, (int)(cy - supy + 0.5f)), y1 = min(sh, (int)(cy + supy + 0.5f));
  int x0 = max(0, (int)(cx - supx + 0.5f)), x1 = min(sw, (int)(cx + supx + 0.5f));
  float wysum = 0.f, wxsum = 0.f;
  for (int i = y0; i < y1; i++) wysum += fmaxf(0.f, 1.f - fabsf(i + 0.5f - cy) / supy);
  for (int j = x0; j < x1; j++) wxsum += fmaxf(0.f, 1.f - fabsf(j + 0.5f - cx) / supx);
  float b = 0.f, g = 0.f, r = 0.f;
  for (int i = y0; i < y1; i++) {
    float wy = fmaxf(0.f, 1.f - fabsf(i + 0.5f - cy) / supy);
    const uint8_t* row = src + (size_t)i * sw * 3;
    for (int j = x0; j < x1; j++) {
      float w = wy * fmaxf(0.f, 1.f - fabsf(j + 0.5f - cx) / supx);
      const uint8_t* p = row + (size_t)j * 3;
      b += w * p[0]; g += w * p[1]; r += w * p[2];
    }
  }
  float inv = 1.f / (wysum * wxsum * 255.f);
  __half* o = dst + (size_t)idx * 3;
  o[0] = __float2half(r * inv);   // RGB order
  o[1] = __float2half(g * inv);
  o[2] = __float2half(b * inv);
}

// host-side torchvision geometry: shortest edge -> S (int() truncation of the long edge),
// center crop with python round-half-to-even offsets
struct ClsGeom { int rh, rw, top, left; };
static ClsGeom clsGeom(int S, int sh, int sw) {
  long rh, rw;
  if (sw <= sh) { rw = S; rh = (long)S * sh / sw; }
  else          { rh = S; rw = (long)S * sw / sh; }
  int dy = (int)rh - S, dx = (int)rw - S;
  int top  = dy / 2 + ((dy & 1) && ((dy / 2) & 1));
  int left = dx / 2 + ((dx & 1) && ((dx / 2) & 1));
  return { (int)rh, (int)rw, top, left };
}

// single-block GPU NMS: bitonic sort by score, greedy suppression, compacted output.
// cstride = candidate record stride (6 detect, 6+nm segment; only the first 6 floats are
// read here); outidx0 records each survivor's candidate slot for post-NMS mask assembly.
__global__ void k_nms(const float* __restrict__ cand0, const int* __restrict__ cnt0,
                      float* __restrict__ out0, int* __restrict__ outcnt0,
                      int* __restrict__ outidx0, float thr, int maxdet, int cstride) {
  const float* cand = cand0 + (size_t)blockIdx.x * maxdet * cstride;
  float* out = out0 + (size_t)blockIdx.x * MAX_OUT * 6;
  int* outidx = outidx0 + (size_t)blockIdx.x * MAX_OUT;
  const int* cnt = cnt0 + blockIdx.x;
  int* outcnt = outcnt0 + blockIdx.x;
  __shared__ float bx1[NMS_CAP], by1[NMS_CAP], bx2[NMS_CAP], by2[NMS_CAP], bsc[NMS_CAP];
  __shared__ short bcl[NMS_CAP];
  __shared__ unsigned char dead[NMS_CAP];
  __shared__ short order[NMS_CAP];
  const int tid = threadIdx.x;
  int n = min(*cnt, NMS_CAP);
  for (int i = tid; i < NMS_CAP; i += blockDim.x) {
    if (i < n) {
      const float* c = cand + (size_t)i * cstride;
      bx1[i] = c[0]; by1[i] = c[1]; bx2[i] = c[2]; by2[i] = c[3];
      bsc[i] = c[4]; bcl[i] = (short)c[5]; dead[i] = 0;
    } else { bsc[i] = -1.f; dead[i] = 1; }
    order[i] = (short)i;
  }
  __syncthreads();
  // bitonic sort of order[] by score, descending (width = next pow2 of n)
  int pad = 32;
  while (pad < n) pad <<= 1;
  for (int k = 2; k <= pad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < pad; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool desc = ((i & k) == 0);
          float a = bsc[order[i]], b = bsc[order[ixj]];
          if ((a < b) == desc) { short t = order[i]; order[i] = order[ixj]; order[ixj] = t; }
        }
      }
      __syncthreads();
    }
  }
  // greedy suppression in score order
  for (int i = 0; i < n; i++) {
    int oi = order[i];
    if (!dead[oi]) {
      float ax1 = bx1[oi], ay1 = by1[oi], ax2 = bx2[oi], ay2 = by2[oi];
      float aarea = (ax2 - ax1) * (ay2 - ay1);
      short acl = bcl[oi];
      for (int j = i + 1 + tid; j < n; j += blockDim.x) {
        int oj = order[j];
        if (dead[oj] || bcl[oj] != acl) continue;
        float xx1 = fmaxf(ax1, bx1[oj]), yy1 = fmaxf(ay1, by1[oj]);
        float xx2 = fminf(ax2, bx2[oj]), yy2 = fminf(ay2, by2[oj]);
        float w = fmaxf(0.f, xx2 - xx1), h = fmaxf(0.f, yy2 - yy1);
        float inter = w * h;
        float ua = aarea + (bx2[oj] - bx1[oj]) * (by2[oj] - by1[oj]) - inter;
        if (ua > 0 && inter / ua > thr) dead[oj] = 1;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    int m = 0;
    for (int i = 0; i < n && m < MAX_OUT; i++) {
      int o = order[i];
      if (dead[o]) continue;
      float* d = out + m * 6;
      d[0] = bx1[o]; d[1] = by1[o]; d[2] = bx2[o]; d[3] = by2[o];
      d[4] = bsc[o]; d[5] = (float)bcl[o];
      outidx[m] = o;   // order[] permutes slot indices, so o IS the candidate slot
      m++;
    }
    *outcnt = m;
  }
}

// mask assembly for NMS survivors only: logits = coef . proto, crop to box (half-open float
// compares, exactly ultralytics crop_mask), threshold logit > 0 (== sigmoid > 0.5). One block
// per (survivor, image); fixed (MAX_OUT, Bn) grid + early exit captures into the CUDA graph.
// rx/ry = proto/input resolution ratio (Pw/netW, Ph/netH).
__global__ void k_mask_assemble(const __half* __restrict__ P, int ps, int po, int Ph, int Pw,
                                int nm, const float* __restrict__ cand, int cstride, int maxdet,
                                const int* __restrict__ outidx, const int* __restrict__ outcnt,
                                const float* __restrict__ outdets,
                                uint8_t* __restrict__ masks, float rx, float ry) {
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

// probiou of two Gaussian-bounded rotated boxes given centers + covariance (A,B,C).
// Exact ultralytics batch_probiou transcription (eps placement matters: t1/t2 denominator,
// t3 sqrt denominator AND log argument — NOT the t3 numerator). fp32.
__device__ __forceinline__ float probiou(const float* c1, const float* c2) {
  const float eps = 1e-7f;
  float dx = c1[0] - c2[0], dy = c1[1] - c2[1];
  float sA = c1[2] + c2[2], sB = c1[3] + c2[3], sC = c1[4] + c2[4];
  float disc = sA * sB - sC * sC;
  float den = disc + eps;
  float t1 = 0.25f * (sA * dy * dy + sB * dx * dx) / den;
  float t2 = 0.5f * (sC * (-dx) * dy) / den;                 // (x2-x1)*(y1-y2)
  float det1 = fmaxf(c1[2] * c1[3] - c1[4] * c1[4], 0.f);
  float det2 = fmaxf(c2[2] * c2[3] - c2[4] * c2[4], 0.f);
  float t3 = 0.5f * logf(disc / (4.f * sqrtf(det1 * det2) + eps) + eps);
  float bd = fminf(fmaxf(t1 + t2 + t3, eps), 100.f);
  return 1.f - sqrtf(1.f - expf(-bd) + eps);
}

// rotated NMS, ultralytics TorchNMS.fast_nms semantics (NOT greedy): a suppressed box
// still suppresses later boxes and the test is iou >= thr — so the j-loop parallelizes
// with no sequential dead-flag dependence. Same-class-only pairs (ultralytics separates
// classes by a +cls*7680 center offset; probiou of that hits the bd clamp -> iou ~ 0).
// One block per image; candidates are 7-float records.
__global__ void k_nms_obb(const float* __restrict__ cand0, const int* __restrict__ cnt0,
                          float* __restrict__ out0, int* __restrict__ outcnt0,
                          float* __restrict__ cov0, float thr, int maxdet) {
  const float* cand = cand0 + (size_t)blockIdx.x * maxdet * 7;
  float* out = out0 + (size_t)blockIdx.x * MAX_OUT * 7;
  float* cov = cov0 + (size_t)blockIdx.x * OBB_NMS_CAP * 5;
  const int* cnt = cnt0 + blockIdx.x;
  int* outcnt = outcnt0 + blockIdx.x;
  __shared__ float bsc[OBB_NMS_CAP];
  __shared__ short order[OBB_NMS_CAP];
  __shared__ unsigned char dead[OBB_NMS_CAP];
  const int tid = threadIdx.x;
  int n = min(*cnt, OBB_NMS_CAP);
  for (int i = tid; i < OBB_NMS_CAP; i += blockDim.x) {
    if (i < n) {
      const float* c = cand + i * 7;
      float a = c[2] * c[2] * (1.f / 12.f), b = c[3] * c[3] * (1.f / 12.f);
      float cs = cosf(c[6]), sn = sinf(c[6]);
      float* cv = cov + (size_t)i * 5;
      cv[0] = c[0]; cv[1] = c[1];
      cv[2] = a * cs * cs + b * sn * sn;
      cv[3] = a * sn * sn + b * cs * cs;
      cv[4] = (a - b) * cs * sn;
      bsc[i] = c[4]; dead[i] = 0;
    } else { bsc[i] = -1.f; dead[i] = 1; }
    order[i] = (short)i;
  }
  __syncthreads();
  // bitonic sort of order[] by score, descending (width = next pow2 of n)
  int pad = 32;
  while (pad < n) pad <<= 1;
  for (int k = 2; k <= pad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < pad; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool desc = ((i & k) == 0);
          float a = bsc[order[i]], b = bsc[order[ixj]];
          if ((a < b) == desc) { short t = order[i]; order[i] = order[ixj]; order[ixj] = t; }
        }
      }
      __syncthreads();
    }
  }
  // fast-NMS: j survives iff no earlier-ranked i (dead or not) has iou >= thr
  for (int jj = tid; jj < n; jj += blockDim.x) {
    int oj = order[jj];
    short clsj = (short)cand[oj * 7 + 5];
    const float* cj = cov + (size_t)oj * 5;
    for (int ii = 0; ii < jj; ii++) {
      int oi = order[ii];
      if ((short)cand[oi * 7 + 5] != clsj) continue;
      if (probiou(cov + (size_t)oi * 5, cj) >= thr) { dead[oj] = 1; break; }
    }
  }
  __syncthreads();
  if (tid == 0) {
    int m = 0;
    for (int i = 0; i < n && m < MAX_OUT; i++) {
      int o = order[i];
      if (dead[o]) continue;
      const float* c = cand + o * 7;
      float* d = out + m * 7;
#pragma unroll
      for (int k = 0; k < 7; k++) d[k] = c[k];
      m++;
    }
    *outcnt = m;
  }
}

// ------------------------- dispatch -------------------------
struct View { const __half* p; int s, o, H, W, C; };
static View view(const Net& n, int tid) {
  const Ten& t = n.tens[tid]; const Buf& b = n.bufs[t.buf];
  return { b.p, b.C, t.coff, b.H, b.W, t.C };
}

static void runOp(Net& net, const Op& op, cudaStream_t st, int Bn) {
  const int TB = 256;
  switch (op.kind) {
    case CONV: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = y.H * y.W * op.Cout;
      if (op.g == 1 && op.Cin == 3) {   // first conv
        if (op.Cout % 32 == 0) {
          dim3 g0((Bn * y.H * y.W + 255) / 256, op.Cout / 32);
          k_conv0<32><<<g0, 256, 0, st>>>(x.p, x.H, x.W, (__half*)y.p, y.H, y.W, op.Cout,
                                          net.weights + op.woff, net.bias + op.boff, op.act, Bn);
        } else {
          dim3 g0((Bn * y.H * y.W + 255) / 256, op.Cout / 16);
          k_conv0<16><<<g0, 256, 0, st>>>(x.p, x.H, x.W, (__half*)y.p, y.H, y.W, op.Cout,
                                          net.weights + op.woff, net.bias + op.boff, op.act, Bn);
        }
      } else if (op.g == 1 && op.Cin >= 8) {
        View r{};
        bool hasRes = op.b >= 0;
        if (hasRes) r = view(net, op.b);
        int M = Bn * y.H * y.W;
        int b64 = ((M + 63) / 64) * ((op.Cout + 63) / 64);
        bool kone = (op.k == 1 && op.s == 1 && op.p == 0);
        long Ktot = (long)op.k * op.k * op.Cin;
        auto launch = [&](auto kern, int bm, int bn, int threads) {
          dim3 grid((M + bm - 1) / bm, (op.Cout + bn - 1) / bn);
          kern<<<grid, threads, 0, st>>>(
              x.p, x.H, x.W, x.s, x.o, (__half*)y.p, y.H, y.W, y.s, y.o,
              net.weights + op.woff, Ktot, net.bias + op.boff,
              hasRes ? r.p : nullptr, hasRes ? r.s : 0, hasRes ? r.o : 0,
              op.Cin, op.Cout, op.k, op.s, op.p, op.act, Bn);
        };
        static const int amode = getenv("ACC16") ? 1 : (getenv("ACC32") ? 0 : 2);
#define LAUNCH_TILE(BM_, BN_, ST_, WM_, WN_)                                                     \
        do {                                                                                     \
          if (kone) {                                                                            \
            if (amode == 1)      launch(k_conv_mma<BM_, BN_, ST_, true, 1, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
            else if (amode == 0) launch(k_conv_mma<BM_, BN_, ST_, true, 0, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
            else                 launch(k_conv_mma<BM_, BN_, ST_, true, 2, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
          } else {                                                                               \
            if (amode == 1)      launch(k_conv_mma<BM_, BN_, ST_, false, 1, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
            else if (amode == 0) launch(k_conv_mma<BM_, BN_, ST_, false, 0, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
            else                 launch(k_conv_mma<BM_, BN_, ST_, false, 2, WM_, WN_>, BM_, BN_, WM_ * WN_ * 32); \
          }                                                                                      \
        } while (0)
        // wide-N tiles only where A (im2col) re-reads dominate: 3x3 with big M and wide Cout.
        // 1x1 convs have no im2col amplification — narrow tiles + deep pipeline win there.
        int bwide = ((M + 63) / 64) * ((op.Cout + 127) / 128);
        bool wide = (op.k == 3) && M >= 1600 && op.Cout >= 128 && bwide >= 38;
        if (op.variant == 3)                      LAUNCH_TILE(64, 128, 3, 2, 4);
        else if (op.variant == 0 && wide)         LAUNCH_TILE(64, 128, 3, 2, 4);
        else if (op.Cout <= 32 && M >= 6400)      LAUNCH_TILE(64, 32, 3, 2, 2);
        else if (b64 >= 24 || op.variant) {
          // tile + warp-grid variants; the startup autotuner picks per op (heuristic if 0)
          bool w8 = op.variant ? (op.variant == 2) : (kone || M > 400);
          if (w8) LAUNCH_TILE(64, 64, 3, 2, 4);
          else    LAUNCH_TILE(64, 64, 3, 2, 2);
        }
        else                                      LAUNCH_TILE(32, 32, 3, 2, 2);
#undef LAUNCH_TILE
      } else if (op.g == 1)
        k_conv_direct<<<(total + TB - 1) / TB, TB, 0, st>>>(
            x.p, x.H, x.W, x.s, x.o, (__half*)y.p, y.H, y.W, y.s, y.o,
            net.weights + op.woff, net.bias + op.boff, op.Cin, op.Cout, op.k, op.s, op.p, op.act);
      else {
        View r{};
        bool hasRes = op.b >= 0;
        if (hasRes) r = view(net, op.b);
        int nthr = Bn * y.H * y.W * (op.Cin / 8);
        k_dwconv8<<<(nthr + TB - 1) / TB, TB, 0, st>>>(
            x.p, x.H, x.W, x.s, x.o, (__half*)y.p, y.H, y.W, y.s, y.o,
            net.weights + op.woff, net.bias + op.boff,
            hasRes ? r.p : nullptr, hasRes ? r.s : 0, hasRes ? r.o : 0,
            op.Cin, op.k, op.s, op.p, op.act, Bn);
      }
      break;
    }
    case ADD: {
      View a = view(net, op.a), b = view(net, op.b), y = view(net, op.out);
      int total = Bn * a.H * a.W * a.C;
      k_add<<<(total + TB - 1) / TB, TB, 0, st>>>(a.p, a.s, a.o, b.p, b.s, b.o,
                                                 (__half*)y.p, y.s, y.o, Bn * a.H * a.W, a.C);
      break;
    }
    case MAXPOOL5: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = Bn * x.H * x.W * (x.C / 8);
      k_maxpool5<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.H, x.W, x.C, Bn);
      break;
    }
    case UPSAMPLE2: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = Bn * y.H * y.W * 4 * (x.C / 8);
      k_upsample2<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.H, x.W, x.C, Bn);
      break;
    }
    case COPYC: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = Bn * x.H * x.W * x.C;
      k_copyc<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, Bn * x.H * x.W, x.C);
      break;
    }
    case ATTN: {
      View q = view(net, op.a), y = view(net, op.out);
      int N = q.H * q.W;
      // mma path alignment: cp.async vec8 tiles need kd % 8 and N % 8 (P@V^T reads attnP
      // rows of length N), half2 stores need hd % 2. q.s % 8 holds by construction.
      bool mma_ok = (op.kd % 8 == 0) && (op.hd % 2 == 0) && (N % 8 == 0);
      if (mma_ok) {
        // tensor-core attention via the implicit-GEMM kernel:
        //   scores[h] = Q[Nxkd] @ K^T   (K rows read in place from the qkv buffer)
        //   P = softmax(scores * 1/sqrt(kd));  out[h] = P[NxN] @ V[Nxhd] via V^T
        const int per = 2 * op.kd + op.hd;
        for (int img = 0; img < Bn; img++) {   // per-image attention (tokens must not mix)
          const __half* qb = q.p + (size_t)img * N * q.s;
          __half* yb = (__half*)y.p + (size_t)img * N * y.s;
          for (int h = 0; h < op.heads; h++) {
            dim3 g1((N + 63) / 64, (N + 63) / 64);
            k_conv_mma<64, 64, 3, true><<<g1, 128, 0, st>>>(
                qb, N, 1, q.s, q.o + h * per,
                net.attnP + (size_t)h * N * N, N, 1, N, 0,
                qb + q.o + h * per + op.kd, q.s, net.zerobias,
                nullptr, 0, 0, op.kd, N, 1, 1, 0, 0, 1);
          }
          k_softmax_rows<<<(op.heads * N + 7) / 8, 256, 0, st>>>(
              net.attnP, op.heads * N, N, rsqrtf((float)op.kd));
          k_vtrans<<<(op.heads * op.hd * N + 255) / 256, 256, 0, st>>>(
              qb, q.s, q.o, net.vt, N, op.kd, op.hd, op.heads);
          for (int h = 0; h < op.heads; h++) {
            dim3 g2((N + 31) / 32, (op.hd + 31) / 32);
            k_conv_mma<32, 32, 3, true><<<g2, 128, 0, st>>>(
                net.attnP + (size_t)h * N * N, N, 1, N, 0,
                yb, N, 1, y.s, y.o + h * op.hd,
                net.vt + (size_t)h * op.hd * N, N, net.zerobias,
                nullptr, 0, 0, N, op.hd, 1, 1, 0, 0, 1);
          }
        }
      } else {
        // slow but shape-generic; gridDim.z carries the batch (one image per z-slice)
        dim3 grid(N, op.heads, Bn);
        size_t smem = N * sizeof(float);
        k_attn<<<grid, 128, smem, st>>>(q.p, q.s, q.o, (__half*)y.p, y.s, y.o, N, op.heads, op.kd, op.hd);
      }
      break;
    }
    case DECODE: {
      View b = view(net, op.a), c = view(net, op.b);
      int total = Bn * b.H * b.W;
      k_decode<0><<<(total + TB - 1) / TB, TB, 0, st>>>(b.p, b.s, b.o, c.p, c.s, c.o,
          nullptr, 0, 0,
          b.H, b.W, op.stride, 0.25f, net.dets, net.detcnt, Net::MAXDET, net.nc, Bn);
      break;
    }
    case DECODESEG: {
      View b = view(net, op.a), c = view(net, op.b), m = view(net, op.out);
      int total = Bn * b.H * b.W;
      k_decode<32><<<(total + TB - 1) / TB, TB, 0, st>>>(b.p, b.s, b.o, c.p, c.s, c.o,
          m.p, m.s, m.o,
          b.H, b.W, op.stride, 0.25f, net.dets, net.detcnt, Net::MAXDET, net.nc, Bn);
      break;
    }
    case PS2: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = Bn * y.H * y.W * (y.C / 8);
      k_ps2<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o,
                                                  x.H, x.W, y.C, Bn);
      break;
    }
    case MASKS:   // declarative (proto tensor name); assembly runs post-NMS in forward()
      break;
    case DECODEOBB: {
      View b = view(net, op.a), c = view(net, op.b), a = view(net, op.out);
      int total = Bn * b.H * b.W;
      k_decode_obb<<<(total + TB - 1) / TB, TB, 0, st>>>(
          b.p, b.s, b.o, c.p, c.s, c.o, a.p, a.s, a.o,
          b.H, b.W, op.stride, 0.25f, net.dets, net.detcnt, Net::MAXDET,
          net.nc, c.C, Bn);
      break;
    }
    case GAP: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = Bn * x.C;
      k_gap<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o,
                                                  x.H * x.W, x.C, Bn);
      break;
    }
    case SOFTMAX: {
      View x = view(net, op.a), y = view(net, op.out);
      k_softmax_ch<<<Bn, 256, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.C);
      break;
    }
  }
}

static void forward(Net& net, cudaStream_t st, int Bn) {
  if (!net.cls) CK(cudaMemsetAsync(net.detcnt, 0, Bn * sizeof(int), st));
  for (auto& op : net.ops) runOp(net, op, st, Bn);
  if (net.cls) {   // no decode/NMS: D2H the softmax probs (dets/detcnt hold garbage)
    View p = view(net, net.probsTen);
    CK(cudaMemcpyAsync(net.h_probs, p.p, (size_t)Bn * net.nc * sizeof(__half),
                       cudaMemcpyDeviceToHost, st));
    return;
  }
  if (net.obb)
    k_nms_obb<<<Bn, 256, 0, st>>>(net.dets, net.detcnt, net.outdets, net.outcnt,
                                  net.nmsCov, 0.45f, Net::MAXDET);
  else
    k_nms<<<Bn, 256, 0, st>>>(net.dets, net.detcnt, net.outdets, net.outcnt, net.outidx,
                              0.45f, Net::MAXDET, net.detStride);
  if (net.protoTen >= 0) {   // segment: assemble survivor masks (device-resident; the D2H
    // of B x 300 x Ph x Pw would cost ~0.3 ms/img — fetch on demand after sync instead)
    View pr = view(net, net.protoTen);
    k_mask_assemble<<<dim3(MAX_OUT, Bn), 256, net.nm * sizeof(float), st>>>(
        pr.p, pr.s, pr.o, pr.H, pr.W, net.nm, net.dets, net.detStride, Net::MAXDET,
        net.outidx, net.outcnt, net.outdets, net.masks,
        (float)pr.W / net.inW, (float)pr.H / net.inH);
  }
  CK(cudaMemcpyAsync(net.h_rawcnt, net.detcnt, Bn * sizeof(int), cudaMemcpyDeviceToHost, st));
  CK(cudaMemcpyAsync(net.h_cnt, net.outcnt, Bn * sizeof(int), cudaMemcpyDeviceToHost, st));
  CK(cudaMemcpyAsync(net.h_out, net.outdets, (size_t)Bn * 300 * net.detK() * sizeof(float), cudaMemcpyDeviceToHost, st));
}

// fuse CONV + following ADD (residual) into the conv epilogue
static void fuseAdds(Net& net) {
  std::vector<Op> out;
  int fused = 0;
  for (size_t i = 0; i < net.ops.size(); i++) {
    Op op = net.ops[i];
    if (op.kind == CONV && op.b < 0 && i + 1 < net.ops.size()) {
      const Op& nx = net.ops[i + 1];
      if (nx.kind == ADD && (nx.a == op.out || nx.b == op.out)) {
        op.b = (nx.a == op.out) ? nx.b : nx.a;
        op.out = nx.out;
        i++; fused++;
      }
    }
    out.push_back(op);
  }
  net.ops = out;
  printf("fused %d ADDs into conv epilogues (%zu ops remain)\n", fused, net.ops.size());
}

// startup autotune: time both (64,64) warp-grid variants per conv, keep the winner
static void autotune(Net& net, cudaStream_t st) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int tuned = 0, wide8 = 0;
  for (auto& op : net.ops) {
    if (op.kind != CONV || op.g != 1 || op.Cin < 8) continue;
    View y = view(net, op.out);
    int M = y.H * y.W;
    int b64 = ((M + 63) / 64) * ((op.Cout + 63) / 64);
    int bwide = ((M + 63) / 64) * ((op.Cout + 127) / 128);
    if ((op.Cout <= 32 && M >= 6400) || b64 < 24) continue;  // fixed-tile branches
    bool wideOk = op.Cout >= 128 && bwide >= 24;   // (64,128) grid still fills the SMs
    int nv = wideOk ? 3 : 2;
    float t[3] = {1e30f, 1e30f, 1e30f};
    for (int v = 1; v <= nv; v++) {
      op.variant = v;
      runOp(net, op, st, net.B);                // warm
      cudaEventRecord(e0, st);
      for (int r = 0; r < 8; r++) runOp(net, op, st, net.B);
      cudaEventRecord(e1, st);
      CK(cudaStreamSynchronize(st));
      cudaEventElapsedTime(&t[v - 1], e0, e1);
    }
    op.variant = 1;
    for (int v = 2; v <= nv; v++)
      if (t[v - 1] < t[op.variant - 1]) op.variant = v;
    tuned++; wide8 += op.variant >= 2;
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  printf("autotuned %d convs (%d chose 8-warp grid)\n", tuned, wide8);
}

// ------------------------- CPU NMS -------------------------
struct Det { float x1, y1, x2, y2, sc; int cls; };
// k_nms only sorts the first NMS_CAP candidates (smem-bound); make truncation observable
static void nmsTripwire(const Net& net, int img) {
  static bool warned = false;
  int cap = net.obb ? OBB_NMS_CAP : NMS_CAP;
  if (!warned && net.h_rawcnt[img] > cap) {
    fprintf(stderr, "warning: img %d had %d conf-passing candidates, NMS saw only %d\n",
            img, net.h_rawcnt[img], cap);
    warned = true;
  }
}
// NMS ran on the GPU inside the graph; just fetch the compacted results.
static std::vector<Det> fetchNMS(Net& net, int img = 0) {
  // results were copied into pinned host memory inside the graph; stream is already synced
  nmsTripwire(net, img);
  int cnt = net.h_cnt[img];
  std::vector<Det> v;
  for (int i = 0; i < cnt; i++) {
    const float* d = net.h_out + ((size_t)img * 300 + i) * 6;
    v.push_back({d[0], d[1], d[2], d[3], d[4], (int)d[5]});
  }
  return v;
}

struct ObbDet { float cx, cy, w, h, sc, ang; int cls; };
static std::vector<ObbDet> fetchNMSObb(Net& net, int img = 0) {
  nmsTripwire(net, img);
  int cnt = net.h_cnt[img];
  std::vector<ObbDet> v;
  for (int i = 0; i < cnt; i++) {
    const float* d = net.h_out + ((size_t)img * 300 + i) * 7;
    v.push_back({d[0], d[1], d[2], d[3], d[4], d[6], (int)d[5]});
  }
  return v;
}

// scale_boxes(xywh=True) equivalent: center un-padded + everything but the angle un-scaled,
// no clipping (ultralytics skips clip_boxes for obb)
static void printObb(const std::vector<ObbDet>& keep, float scale, int top, int left) {
  printf("kept=%zu\n", keep.size());
  for (const auto& d : keep)
    printf("cls=%2d score=%.4f rbox=(%.1f, %.1f, %.1f, %.1f, %.3f)\n", d.cls, d.sc,
           (d.cx - left) / scale, (d.cy - top) / scale, d.w / scale, d.h / scale, d.ang);
}

// ------------------------- classify CLI output -------------------------
static std::vector<std::string> loadNames(const std::string& dir) {
  std::vector<std::string> v;
  FILE* f = fopen((dir + "/names.txt").c_str(), "r");
  if (!f) return v;   // optional: print bare class ids when absent
  char line[256];
  while (fgets(line, sizeof line, f)) { line[strcspn(line, "\r\n")] = 0; v.push_back(line); }
  fclose(f);
  return v;
}

// partial top-k over the pinned fp16 probs of batch slot img (insertion into k slots)
static void topk(const Net& net, int img, int k, int* cls, float* prob) {
  const __half* p = net.h_probs + (size_t)img * net.nc;
  for (int i = 0; i < k; i++) { cls[i] = -1; prob[i] = -1.f; }
  for (int c = 0; c < net.nc; c++) {
    float v = __half2float(p[c]);
    for (int i = 0; i < k; i++)
      if (v > prob[i]) {
        for (int j = k - 1; j > i; j--) { prob[j] = prob[j - 1]; cls[j] = cls[j - 1]; }
        prob[i] = v; cls[i] = c;
        break;
      }
  }
}

static void printCls(const Net& net, const std::string& dir) {
  auto names = loadNames(dir);
  int cls[5]; float prob[5];
  if (net.B > 1) {   // batched self-consistency: every image got identical input
    printf("batch=%d per-image top1:", net.B);
    bool same = true;
    int c0 = -1;
    for (int b = 0; b < net.B; b++) {
      topk(net, b, 1, cls, prob);
      if (b == 0) c0 = cls[0];
      same &= cls[0] == c0;
      printf(" %d/%.4f", cls[0], prob[0]);
    }
    printf("  %s\n", same ? "(consistent)" : "(MISMATCH!)");
  }
  topk(net, 0, 5, cls, prob);
  printf("top5:\n");
  for (int i = 0; i < 5; i++)
    printf("%4d %-24s %.4f\n", cls[i],
           cls[i] >= 0 && cls[i] < (int)names.size() ? names[cls[i]].c_str() : "?", prob[i]);
}

// load any image via stb_image as BGR u8 HWC (the layout k_preprocess expects)
static uint8_t* loadImageBGR(const std::string& path, int& sh, int& sw) {
  int comp;
  unsigned char* img = stbi_load(path.c_str(), &sw, &sh, &comp, 3);  // RGB
  if (!img) { fprintf(stderr, "cannot load image %s\n", path.c_str()); exit(1); }
  for (size_t i = 0; i < (size_t)sh * sw; i++) std::swap(img[i * 3], img[i * 3 + 2]);
  return img;
}

static void usage() {
  printf("yolocuda — hand-written CUDA inference engine for YOLOv8 / YOLO11\n\n"
         "usage: yolocuda <mode> [model-dir] [iters] [--image path] [--batch N]\n\n"
         "modes:\n"
         "  detect    run one inference, print detections (--image for any jpg/png;\n"
         "            segment: --save-masks writes gpu/masks.u8 + gpu/dets.f32, image 0 only)\n"
         "  bench     net-only benchmark (stream vs CUDA graph)\n"
         "  pipeline  end-to-end benchmark: H2D + preprocess + net + decode + NMS\n"
         "  profile   per-op timing breakdown\n"
         "  dump      write per-op outputs for test/compare.py\n\n"
         "model-dir defaults to build/yolo11n (create with: make export MODEL=yolo11n)\n"
         "env: ACC16=1 -> pure fp16 mma accumulation (fastest, ~2-3%% err)\n"
         "     ACC32=1 -> pure fp32 accumulation (exact-est). default: hybrid K=64 window\n");
}

// ------------------------- embedding API (see engine/yolo11.h) -------------------------
#include "yolo.h"

struct YoloHandle {
  Net net;
  cudaStream_t st;
  std::vector<cudaGraphExec_t> graphs;   // graphs[B-1]
  std::vector<float> execMs;             // calibrated per-B latency
  struct SlotMap { float scale; int top, left; };
  std::vector<SlotMap> slots;
};

extern "C" void* yolo_create(const char* model_dir, int max_batch) {
  auto* h = new YoloHandle();
  h->net.B = max_batch;
  loadGraph(h->net, model_dir);
  fuseAdds(h->net);
  CK(cudaStreamCreate(&h->st));
  autotune(h->net, h->st);
  h->graphs.resize(max_batch);
  h->execMs.resize(max_batch);
  h->slots.resize(max_batch);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int B = 1; B <= max_batch; B++) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(h->st, cudaStreamCaptureModeGlobal));
    forward(h->net, h->st, B);
    CK(cudaStreamEndCapture(h->st, &g));
    CK(cudaGraphInstantiate(&h->graphs[B - 1], g, 0));
    CK(cudaGraphDestroy(g));
    for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(h->graphs[B - 1], h->st));
    CK(cudaStreamSynchronize(h->st));
    cudaEventRecord(e0, h->st);
    for (int i = 0; i < 10; i++) CK(cudaGraphLaunch(h->graphs[B - 1], h->st));
    cudaEventRecord(e1, h->st);
    CK(cudaStreamSynchronize(h->st));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    h->execMs[B - 1] = ms / 10.f;
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  printf("engine ready: B=1..%d, exec %.2f..%.2f ms\n", max_batch, h->execMs[0], h->execMs[max_batch - 1]);
  return h;
}

extern "C" int yolo_task(void* p) {
  const Net& n = ((YoloHandle*)p)->net;
  return n.seg ? 3 : (n.obb ? 1 : (n.cls ? 2 : 0));
}
extern "C" float yolo_exec_ms(void* p, int B) { return ((YoloHandle*)p)->execMs[B - 1]; }
extern "C" cudaStream_t yolo_stream(void* p) { return ((YoloHandle*)p)->st; }

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

extern "C" void yolo_run(void* p, int B) {
  auto* h = (YoloHandle*)p;
  CK(cudaGraphLaunch(h->graphs[B - 1], h->st));
  CK(cudaStreamSynchronize(h->st));
}

extern "C" void yolo_run_async(void* p, int B) {
  auto* h = (YoloHandle*)p;
  CK(cudaGraphLaunch(h->graphs[B - 1], h->st));
}

extern "C" void yolo_sync(void* p) { CK(cudaStreamSynchronize(((YoloHandle*)p)->st)); }

extern "C" int yolo_get(void* p, int slot, YoloDet* out, int cap) {
  auto* h = (YoloHandle*)p;
  const auto& sm = h->slots[slot];
  nmsTripwire(h->net, slot);
  int cnt = std::min(h->net.h_cnt[slot], cap);
  for (int i = 0; i < cnt; i++) {
    const float* d = h->net.h_out + ((size_t)slot * 300 + i) * 6;
    out[i] = {(d[0] - sm.left) / sm.scale, (d[1] - sm.top) / sm.scale,
              (d[2] - sm.left) / sm.scale, (d[3] - sm.top) / sm.scale, d[4], (int)d[5]};
  }
  return cnt;
}

extern "C" void yolo_mask_dim(void* p, int* ph, int* pw) {
  const Net& n = ((YoloHandle*)p)->net;
  if (n.protoTen < 0) { *ph = *pw = 0; return; }
  const Buf& b = n.bufs[n.tens[n.protoTen].buf];
  *ph = b.H; *pw = b.W;
}

extern "C" int yolo_get_mask(void* p, int slot, int det, unsigned char* out) {
  auto* h = (YoloHandle*)p;
  const Net& n = h->net;
  if (n.protoTen < 0 || det < 0 || det >= n.h_cnt[slot]) return -1;
  const Buf& b = n.bufs[n.tens[n.protoTen].buf];
  size_t sz = (size_t)b.H * b.W;
  CK(cudaMemcpy(out, n.masks + ((size_t)slot * MAX_OUT + det) * sz, sz, cudaMemcpyDeviceToHost));
  return 0;
}

// classify-model preprocessing (torchvision resize-shortest + center-crop geometry)
extern "C" void yolo_preprocess_cls(void* p, const unsigned char* dev_bgr, int sh, int sw, int slot) {
  auto* h = (YoloHandle*)p;
  Buf& ib = h->net.bufs[h->net.tens[0].buf];
  ClsGeom gm = clsGeom(ib.H, sh, sw);
  k_preprocess_cls<<<(ib.H * ib.W + 255) / 256, 256, 0, h->st>>>(
      dev_bgr, sh, sw, ib.p + (size_t)slot * ib.H * ib.W * ib.C, ib.H, gm.rh, gm.rw, gm.top, gm.left);
}

// top-k class ids + probs for a slot (classify models). Returns count written.
extern "C" int yolo_get_cls(void* p, int slot, int* ids, float* probs, int k) {
  const Net& n = ((YoloHandle*)p)->net;
  if (!n.cls) return 0;
  k = std::min(k, n.nc);
  topk(n, slot, k, ids, probs);
  return k;
}

// letterbox geometry of a slot (set by yolo_preprocess) — needed to map segment masks,
// which live in letterbox space at proto resolution, back to original image coords.
extern "C" void yolo_slot_geom(void* p, int slot, float* scale, int* top, int* left) {
  const auto& sm = ((YoloHandle*)p)->slots[slot];
  *scale = sm.scale; *top = sm.top; *left = sm.left;
}

extern "C" int yolo_get_obb(void* p, int slot, YoloObbDet* out, int cap) {
  auto* h = (YoloHandle*)p;
  const auto& sm = h->slots[slot];
  nmsTripwire(h->net, slot);
  int cnt = std::min(h->net.h_cnt[slot], cap);
  for (int i = 0; i < cnt; i++) {
    const float* d = h->net.h_out + ((size_t)slot * 300 + i) * 7;
    out[i] = {(d[0] - sm.left) / sm.scale, (d[1] - sm.top) / sm.scale,
              d[2] / sm.scale, d[3] / sm.scale, d[6], d[4], (int)d[5]};
  }
  return cnt;
}

#ifndef YOLO11_LIB
// ------------------------- main -------------------------
int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "detect";
  std::string dir = "build/yolo11n";
  int iterArg = -1;
  std::string image;
  int batch = 1;
  bool saveMasks = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--image" && i + 1 < argc) image = argv[++i];
    else if (a == "--batch" && i + 1 < argc) batch = atoi(argv[++i]);
    else if (a == "--save-masks") saveMasks = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (a[0] >= '0' && a[0] <= '9') iterArg = atoi(a.c_str());
    else dir = a;
  }
  if (mode != "detect" && mode != "bench" && mode != "pipeline" &&
      mode != "profile" && mode != "dump") {
    usage();
    return mode == "-h" || mode == "--help" || mode == "help" ? 0 : 1;
  }
  Net net;
  net.B = batch;
  loadGraph(net, dir);
  if (mode != "dump" && !getenv("NOFUSE")) fuseAdds(net);

  // default input: the exporter's pre-letterboxed reference image
  auto in = readFile(dir + "/input.f16");
  Buf& ib = net.bufs[net.tens[0].buf];
  for (int b = 0; b < net.B; b++)   // replicate reference input across the batch
    CK(cudaMemcpy(ib.p + (size_t)b * (in.size() / sizeof(__half)), in.data(), in.size(),
                  cudaMemcpyHostToDevice));

  cudaStream_t st; CK(cudaStreamCreate(&st));
  if (mode != "dump" && !getenv("NOTUNE")) autotune(net, st);

  auto makeGraph = [&](cudaGraphExec_t& gexec) {
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    forward(net, st, net.B);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&gexec, graph, 0));
    CK(cudaGraphDestroy(graph));
  };

  if (mode == "dump") {
    CK(cudaMemsetAsync(net.detcnt, 0, sizeof(int), st));
    system(("mkdir -p " + dir + "/gpu").c_str());
    for (size_t k = 0; k < net.ops.size(); k++) {
      runOp(net, net.ops[k], st, net.B);
      CK(cudaStreamSynchronize(st));
      CK(cudaGetLastError());
      const Op& op = net.ops[k];
      if (op.kind == DECODE || op.kind == DECODEOBB || op.kind == DECODESEG ||
          op.kind == MASKS) continue;
      View y = view(net, op.out);
      // copy strided view to contiguous host
      std::vector<__half> host((size_t)y.H * y.W * y.C);
      CK(cudaMemcpy2D(host.data(), y.C * sizeof(__half), y.p + y.o, y.s * sizeof(__half),
                      y.C * sizeof(__half), (size_t)y.H * y.W, cudaMemcpyDeviceToHost));
      char path[256]; snprintf(path, sizeof path, "%s/gpu/op%03zu.bin", dir.c_str(), k);
      FILE* f = fopen(path, "wb");
      fwrite(host.data(), sizeof(__half), host.size(), f);
      fclose(f);
      // resync: overwrite the output with the torch reference so every op is compared
      // against exact reference inputs — dump mode is an isolated KERNEL check.
      // Without it, fp16 activation-storage noise cascades and amplifies with network
      // depth and resolution (obb cv4 at 1024 reached 11% rel, yolo11l 5% at 640, from
      // <2% actual kernel error). Compounded-path fidelity is what the end-to-end
      // ultralytics comparison verifies.
      {
        snprintf(path, sizeof path, "%s/ref/op%03zu.npy", dir.c_str(), k);
        f = fopen(path, "rb");
        if (f) {
          size_t cnt = (size_t)y.H * y.W * y.C;
          fseek(f, 0, SEEK_END); long off = ftell(f) - (long)(cnt * sizeof(float));
          if (off >= 10 && off <= 256) {   // npy v1 header; fp32 C-order [H,W,C] payload
            fseek(f, off, SEEK_SET);
            std::vector<float> rf(cnt);
            if (fread(rf.data(), sizeof(float), cnt, f) == cnt) {
              for (size_t i = 0; i < cnt; i++) host[i] = __float2half(rf[i]);
              CK(cudaMemcpy2D((__half*)y.p + y.o, y.s * sizeof(__half), host.data(),
                              y.C * sizeof(__half), y.C * sizeof(__half), (size_t)y.H * y.W,
                              cudaMemcpyHostToDevice));
            }
          }
          fclose(f);
        }
      }
    }
    printf("dumped %zu ops\n", net.ops.size());
  } else if (mode == "detect") {
    float scale = 1.f; int top = 0, left = 0;
    if (!image.empty()) {   // arbitrary image via stb_image + GPU preprocess
      int sh, sw;
      uint8_t* img = loadImageBGR(image, sh, sw);
      uint8_t* dimg; CK(cudaMalloc(&dimg, (size_t)sh * sw * 3));
      CK(cudaMemcpy(dimg, img, (size_t)sh * sw * 3, cudaMemcpyHostToDevice));
      stbi_image_free(img);
      if (net.cls) {   // resize-shortest + center crop (all batch slots, keeps them consistent)
        ClsGeom gm = clsGeom(ib.H, sh, sw);
        for (int b = 0; b < net.B; b++)
          k_preprocess_cls<<<(ib.H * ib.W + 255) / 256, 256, 0, st>>>(
              dimg, sh, sw, ib.p + (size_t)b * ib.H * ib.W * ib.C, ib.H, gm.rh, gm.rw, gm.top, gm.left);
      } else {         // letterbox
        scale = std::min((float)ib.H / sh, (float)ib.W / sw);
        int nh = (int)roundf(sh * scale), nw = (int)roundf(sw * scale);
        top = (ib.H - nh) / 2; left = (ib.W - nw) / 2;
        k_preprocess<<<(ib.H * ib.W + 255) / 256, 256, 0, st>>>(
            dimg, sh, sw, ib.p, ib.H, ib.W, scale, top, left, nh, nw);
      }
    }
    forward(net, st, net.B);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    if (net.cls) { printCls(net, dir); return 0; }
    if (net.B > 1) {   // batched self-consistency: every image got identical input
      printf("batch=%d per-image kept:", net.B);
      bool same = true;
      for (int b = 0; b < net.B; b++) { printf(" %d", net.h_cnt[b]); same &= net.h_cnt[b] == net.h_cnt[0]; }
      printf("  %s\n", same ? "(consistent)" : "(MISMATCH!)");
      if (net.protoTen >= 0) {   // segment: total mask pixels must match across the batch too
        View pr = view(net, net.protoTen);
        size_t msz = (size_t)pr.H * pr.W;
        printf("batch=%d per-image mask px:", net.B);
        long a0 = 0; bool msame = true;
        for (int b = 0; b < net.B; b++) {
          std::vector<uint8_t> tmp((size_t)net.h_cnt[b] * msz);
          if (!tmp.empty())
            CK(cudaMemcpy(tmp.data(), net.masks + (size_t)b * MAX_OUT * msz, tmp.size(),
                          cudaMemcpyDeviceToHost));
          long a = 0; for (uint8_t v : tmp) a += v;
          if (b == 0) a0 = a;
          msame &= a == a0;
          printf(" %ld", a);
        }
        printf("  %s\n", msame ? "(consistent)" : "(MISMATCH!)");
      }
    }
    if (net.obb) {     // rotated boxes, mapped back to original image coordinates
      printObb(fetchNMSObb(net), scale, top, left);
    } else {
      auto keep = fetchNMS(net);
      printf("kept=%zu%s\n", keep.size(), image.empty() ? "" : "  (boxes in original image coords)");
      int Ph = 0, Pw = 0;
      std::vector<uint8_t> mh;   // segment: image-0 survivor masks (contiguous in output order)
      if (net.protoTen >= 0 && !keep.empty()) {
        View pr = view(net, net.protoTen);
        Ph = pr.H; Pw = pr.W;
        mh.resize(keep.size() * (size_t)Ph * Pw);
        CK(cudaMemcpy(mh.data(), net.masks, mh.size(), cudaMemcpyDeviceToHost));
      }
      for (size_t i = 0; i < keep.size(); i++) {   // map back to original coords when letterboxed
        const Det& d = keep[i];
        printf("cls=%2d score=%.4f box=(%.1f, %.1f, %.1f, %.1f)", d.cls, d.sc,
               (d.x1 - left) / scale, (d.y1 - top) / scale,
               (d.x2 - left) / scale, (d.y2 - top) / scale);
        if (!mh.empty()) {
          int area = 0;
          const uint8_t* mp = mh.data() + i * (size_t)Ph * Pw;
          for (int p = 0; p < Ph * Pw; p++) area += mp[p];
          printf("  mask=%d/%dpx", area, Ph * Pw);
        }
        printf("\n");
      }
      if (saveMasks && net.protoTen >= 0) {   // for test/compare_seg.py (letterbox 640-space dets;
                                              // image-0 masks/dets only when --batch > 1)
        if (system(("mkdir -p " + dir + "/gpu").c_str()) != 0)
          fprintf(stderr, "mkdir %s/gpu failed\n", dir.c_str());
        FILE* f = fopen((dir + "/gpu/masks.u8").c_str(), "wb");
        fwrite(mh.data(), 1, mh.size(), f); fclose(f);
        f = fopen((dir + "/gpu/dets.f32").c_str(), "wb");
        fwrite(net.h_out, sizeof(float), keep.size() * 6, f); fclose(f);
        printf("wrote %s/gpu/masks.u8 (%zu x %dx%d) + dets.f32\n", dir.c_str(), keep.size(), Ph, Pw);
      }
    }
  } else if (mode == "pipeline") {
    // full pipeline: pinned H2D (raw BGR u8) -> GPU letterbox/normalize -> net -> decode -> GPU NMS
    FILE* fd = fopen((dir + "/bus_raw.txt").c_str(), "r");
    int sh, sw;
    if (!fd || fscanf(fd, "%d %d", &sh, &sw) != 2) { fprintf(stderr, "no bus_raw\n"); return 1; }
    fclose(fd);
    auto raw = readFile(dir + "/bus_raw.u8");
    uint8_t *himg, *dimg;
    CK(cudaMallocHost(&himg, raw.size()));
    memcpy(himg, raw.data(), raw.size());
    CK(cudaMalloc(&dimg, raw.size()));
    Buf& inb = net.bufs[net.tens[0].buf];
    float scale = std::min((float)inb.H / sh, (float)inb.W / sw);
    int nh = (int)roundf(sh * scale), nw = (int)roundf(sw * scale);
    int top = (inb.H - nh) / 2, left = (inb.W - nw) / 2;

    ClsGeom gm = clsGeom(inb.H, sh, sw);   // classify: torchvision resize+crop geometry

    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    CK(cudaMemcpyAsync(dimg, himg, raw.size(), cudaMemcpyHostToDevice, st));
    if (net.cls)
      k_preprocess_cls<<<(inb.H * inb.W + 255) / 256, 256, 0, st>>>(
          dimg, sh, sw, inb.p, inb.H, gm.rh, gm.rw, gm.top, gm.left);
    else
      k_preprocess<<<(inb.H * inb.W + 255) / 256, 256, 0, st>>>(
          dimg, sh, sw, inb.p, inb.H, inb.W, scale, top, left, nh, nw);
    forward(net, st, net.B);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&gexec, graph, 0));
    CK(cudaGraphDestroy(graph));

    CK(cudaGraphLaunch(gexec, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    if (net.cls) {
      printCls(net, dir);
    } else if (net.obb) {
      printObb(fetchNMSObb(net), scale, top, left);
    } else {
      auto keep = fetchNMS(net);
      printf("kept=%zu\n", keep.size());
      for (auto& d : keep)
        printf("cls=%2d score=%.4f box=(%.1f, %.1f, %.1f, %.1f)\n", d.cls, d.sc, d.x1, d.y1, d.x2, d.y2);
    }

    int iters = iterArg > 0 ? iterArg : 300;
    std::chrono::steady_clock::time_point t0;
    for (int i = 0; i < 20 + iters; i++) {   // 20 warmup iters, then timed
      if (i == 20) t0 = std::chrono::steady_clock::now();
      CK(cudaGraphLaunch(gexec, st));
      CK(cudaStreamSynchronize(st));
      if (net.obb) fetchNMSObb(net);      // host consumption at the 7-float record stride
      else if (!net.cls) fetchNMS(net);   // classify results land in pinned h_probs inside the graph
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("end-to-end pipeline (H2D + preprocess + net + %s): %.3f ms/frame (%.1f fps)\n",
           net.cls ? "probs D2H" : "decode + NMS", ms, 1000.0 / ms);
  } else if (mode == "profile") {
    int iters = 50;
    std::vector<float> acc(net.ops.size(), 0.f);
    std::vector<cudaEvent_t> ev(net.ops.size() + 1);
    for (auto& e : ev) cudaEventCreate(&e);
    for (int i = 0; i < 10; i++) forward(net, st, net.B);
    CK(cudaStreamSynchronize(st));
    for (int it = 0; it < iters; it++) {
      CK(cudaMemsetAsync(net.detcnt, 0, sizeof(int), st));
      cudaEventRecord(ev[0], st);
      for (size_t k = 0; k < net.ops.size(); k++) {
        runOp(net, net.ops[k], st, net.B);
        cudaEventRecord(ev[k + 1], st);
      }
      CK(cudaStreamSynchronize(st));
      for (size_t k = 0; k < net.ops.size(); k++) {
        float ms; cudaEventElapsedTime(&ms, ev[k], ev[k + 1]);
        acc[k] += ms;
      }
    }
    const char* names[] = {"CONV", "ADD", "MAXPOOL5", "UPSAMPLE2", "COPYC", "ATTN", "DECODE",
                           "GAP", "SOFTMAX", "DECODEOBB", "PS2", "DECODESEG", "MASKS"};
    std::vector<int> order(net.ops.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return acc[a] > acc[b]; });
    float tot = 0; for (float a : acc) tot += a;
    printf("total %.3f ms/iter — top ops:\n", tot / iters);
    for (int i = 0; i < 25 && i < (int)order.size(); i++) {
      int k = order[i];
      const Op& op = net.ops[k];
      View y = view(net, op.kind == DECODE || op.kind == DECODEOBB || op.kind == DECODESEG ||
                         op.kind == MASKS ? op.a : op.out);   // MASKS has out == -1
      printf("op%03d %-9s %4dx%-4d Cin=%-3d Cout=%-3d k=%d s=%d g=%d res=%d  %7.1f us  (%4.1f%%)\n",
             k, names[op.kind], y.H, y.W, op.Cin, op.Cout, op.k, op.s, op.g, op.b >= 0 && op.kind == CONV,
             acc[k] / iters * 1000.f, 100.f * acc[k] / tot);
    }
  } else if (mode == "bench") {
    int iters = iterArg > 0 ? iterArg : 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 20; i++) forward(net, st, net.B);
    CK(cudaStreamSynchronize(st));
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; i++) forward(net, st, net.B);
    cudaEventRecord(e1, st);
    CK(cudaStreamSynchronize(st));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("stream launch: %.3f ms/iter (%.1f fps)\n", ms / iters, 1000.f * iters / ms);

    cudaGraphExec_t gexec; makeGraph(gexec);
    for (int i = 0; i < 20; i++) CK(cudaGraphLaunch(gexec, st));
    CK(cudaStreamSynchronize(st));
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; i++) CK(cudaGraphLaunch(gexec, st));
    cudaEventRecord(e1, st);
    CK(cudaStreamSynchronize(st));
    cudaEventElapsedTime(&ms, e0, e1);
    printf("cuda graph:    %.3f ms/iter (%.1f fps)\n", ms / iters, 1000.f * iters / ms);
  } else {
    usage();
    return 1;
  }
  return 0;
}
#endif  // YOLO11_LIB
