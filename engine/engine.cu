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

// ------------------------- graph structures -------------------------
struct Buf  { int H, W, C; __half* p = nullptr; };
struct Ten  { int buf, coff, C; };
enum OpKind { CONV, ADD, MAXPOOL5, UPSAMPLE2, COPYC, ATTN, DECODE };
struct Op {
  OpKind kind;
  int a = -1, b = -1, out = -1;        // tensor ids (CONV: a=in, out; ADD: a,b,out)
  int k = 0, s = 0, p = 0, g = 0, act = 0, Cin = 0, Cout = 0;
  long woff = 0, boff = 0;
  int heads = 0, kd = 0, hd = 0;       // ATTN
  int stride = 0;                       // DECODE
};

struct Net {
  std::vector<Buf> bufs;
  std::vector<Ten> tens;
  std::vector<Op> ops;
  __half* weights = nullptr;   // device
  float*  bias = nullptr;      // device
  float*  dets = nullptr;      // device candidate buffer [maxdet*6]
  int*    detcnt = nullptr;    // device counter
  float*  outdets = nullptr;   // device NMS output [MAX_OUT*6]
  int*    outcnt = nullptr;
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
  if (fscanf(f, "%63s %d", magic, &ver) != 2) exit(1);
  int nb, nt, no;
  if (fscanf(f, "%d %d %d", &nb, &nt, &no) != 3) exit(1);
  net.bufs.resize(nb); net.tens.resize(nt);
  char tag[32];
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
    } else { fprintf(stderr, "bad op %s\n", tag); exit(1); }
    net.ops.push_back(op);
  }
  fclose(f);

  for (auto& b : net.bufs) CK(cudaMalloc(&b.p, (size_t)b.H * b.W * b.C * sizeof(__half)));
  for (auto& b : net.bufs) CK(cudaMemset(b.p, 0, (size_t)b.H * b.W * b.C * sizeof(__half)));

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
  CK(cudaMalloc(&net.dets, Net::MAXDET * 6 * sizeof(float)));
  CK(cudaMalloc(&net.detcnt, sizeof(int)));
  CK(cudaMalloc(&net.outdets, 300 * 6 * sizeof(float)));
  CK(cudaMalloc(&net.outcnt, sizeof(int)));
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
  CK(cudaMalloc(&net.zerobias, 1024 * sizeof(float)));
  CK(cudaMemset(net.zerobias, 0, 1024 * sizeof(float)));
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
    int Cin, int Cout, int K, int S, int P, int act) {
  constexpr int BK = 32;
  constexpr int THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MI = WM / 16, NI = WN / 8;  // mma tiles per warp
  __shared__ __half As[STAGES][BM * BK];
  __shared__ __half Bs[STAGES][BN * BK];

  const int M = Ho * Wo;
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
  int ybase[ASLOTS], xbase[ASLOTS];
  bool mok[ASLOTS];
#pragma unroll
  for (int i = 0; i < ASLOTS; i++) {
    int m = bm + ((tid + i * THREADS) >> 2);
    mok[i] = (m < M) && (tid + i * THREADS < BM * 4);
    int oy = m / Wo, ox = m % Wo;
    ybase[i] = oy * S - P; xbase[i] = ox * S - P;
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
          CP_ASYNC16(dst, X + ((size_t)iy * W + ix) * xs + xo + a_ci);
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
__global__ void k_conv0(const __half* __restrict__ X, int H, int W,
                        __half* __restrict__ Y, int Ho, int Wo, int Cout,
                        const __half* __restrict__ Wt, const float* __restrict__ B, int act) {
  const int CHUNK = 16, CIN = 3, K = 3, S = 2, P = 1;
  const int co0 = blockIdx.y * CHUNK;
  __shared__ __half ws[CHUNK * K * K * CIN];   // 432
  __shared__ float bs[CHUNK];
  for (int i = threadIdx.x; i < CHUNK * K * K * CIN; i += blockDim.x) ws[i] = Wt[co0 * K * K * CIN + i];
  for (int i = threadIdx.x; i < CHUNK; i += blockDim.x) bs[i] = B[co0 + i];
  __syncthreads();
  int pix = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= Ho * Wo) return;
  int ox = pix % Wo, oy = pix / Wo;
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
      const __half* xp = X + ((size_t)iy * W + ix) * CIN;
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
  dst[0] = ((uint4*)out)[0]; dst[1] = ((uint4*)out)[1];
}

// vectorized depthwise conv: thread per (pixel, 8-channel group); weights repacked to [K*K][C]
__global__ void k_dwconv8(const __half* __restrict__ X, int H, int W, int xs, int xo,
                          __half* __restrict__ Y, int Ho, int Wo, int ys, int yo,
                          const __half* __restrict__ Wt, const float* __restrict__ B,
                          const __half* __restrict__ Res, int rs, int ro,
                          int C, int K, int S, int P, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cg = C >> 3;
  if (idx >= Ho * Wo * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int ox = pix % Wo, oy = pix / Wo;
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
      const __half* xp = X + ((size_t)iy * W + ix) * xs + xo + c;
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

// attention v3: two-kernel scheme.
// Kernel 1: P[h][q][j] = softmax_j(Q.K^T * scale) for a 64-query tile (K in smem, conflict-free pad).
// Kernel 2: out[q][h*hd+c] = sum_j P[h][q][j] * V[j][c] (V in smem).
#define AQT 32  // query tile
__global__ void k_attn_qk(const __half* __restrict__ QKV, int qs, int qo,
                          __half* __restrict__ P, int N, int kd, int hd) {
  extern __shared__ __half smq[];
  const int KP = 34;                    // padded row (conflict-free: 17 words, odd)
  __half* Ks = smq;                     // [N][KP]
  __half* Qs = smq + N * KP;            // [AQT][KP]
  __half* Ps = Qs + AQT * KP;           // [AQT][N] raw scores then probs
  const int h = blockIdx.y, per = 2 * kd + hd;
  const int q0 = blockIdx.x * AQT;
  const int nq = min(AQT, N - q0);
  for (int i = threadIdx.x; i < N * (kd / 2); i += blockDim.x) {
    int tok = i / (kd / 2), s = i % (kd / 2);
    *(__half2*)&Ks[tok * KP + s * 2] = *(const __half2*)(QKV + (size_t)tok * qs + qo + h * per + kd + s * 2);
  }
  for (int i = threadIdx.x; i < nq * (kd / 2); i += blockDim.x) {
    int q = i / (kd / 2), s = i % (kd / 2);
    *(__half2*)&Qs[q * KP + s * 2] = *(const __half2*)(QKV + (size_t)(q0 + q) * qs + qo + h * per + s * 2);
  }
  __syncthreads();
  const float scale = rsqrtf((float)kd);
  for (int e = threadIdx.x; e < nq * N; e += blockDim.x) {
    int q = e / N, j = e % N;
    const __half2* qp = (const __half2*)&Qs[q * KP];
    const __half2* kp = (const __half2*)&Ks[j * KP];
    float s = 0;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float2 a = __half22float2(qp[d]), b = __half22float2(kp[d]);
      s += a.x * b.x + a.y * b.y;
    }
    Ps[q * N + j] = __float2half(s * scale);
  }
  __syncthreads();
  // softmax per row: warp per row
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int q = warp; q < nq; q += blockDim.x >> 5) {
    float lmax = -1e30f;
    for (int j = lane; j < N; j += 32) lmax = fmaxf(lmax, __half2float(Ps[q * N + j]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(~0u, lmax, o));
    float lsum = 0;
    for (int j = lane; j < N; j += 32) lsum += __expf(__half2float(Ps[q * N + j]) - lmax);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(~0u, lsum, o);
    float inv = 1.f / lsum;
    for (int j = lane; j < N; j += 32) {
      float p = __expf(__half2float(Ps[q * N + j]) - lmax) * inv;
      P[((size_t)h * N + (q0 + q)) * N + j] = __float2half(p);
    }
  }
}

__global__ void k_attn_av(const __half* __restrict__ QKV, int qs, int qo,
                          const __half* __restrict__ P,
                          __half* __restrict__ Y, int ys, int yo,
                          int N, int kd, int hd) {
  extern __shared__ __half smv[];
  const int VP = 66;                    // padded V row
  __half* Vs = smv;                     // [N][VP]
  const int h = blockIdx.y, per = 2 * kd + hd;
  const int q0 = blockIdx.x * AQT;
  const int nq = min(AQT, N - q0);
  const int Npad = (N + 31) & ~31;
  for (int i = threadIdx.x; i < Npad * (hd / 2); i += blockDim.x) {
    int tok = i / (hd / 2), s = i % (hd / 2);
    *(__half2*)&Vs[tok * VP + s * 2] = tok < N
        ? *(const __half2*)(QKV + (size_t)tok * qs + qo + h * per + 2 * kd + s * 2)
        : __floats2half2_rn(0.f, 0.f);
  }
  __syncthreads();
  // warp per query; lanes span 64 channels; P row loaded coalesced + shfl-broadcast
  const int lane = threadIdx.x & 31, qslot = threadIdx.x >> 5;
  for (int q = qslot; q < nq; q += blockDim.x >> 5) {
    const __half* prow = P + ((size_t)h * N + (q0 + q)) * N;
    float a0 = 0, a1 = 0;
    for (int j0 = 0; j0 < N; j0 += 32) {
      float pl = (j0 + lane < N) ? __half2float(prow[j0 + lane]) : 0.f;
#pragma unroll
      for (int jj = 0; jj < 32; jj++) {
        float p = __shfl_sync(~0u, pl, jj);
        float2 v = __half22float2(*(const __half2*)&Vs[(j0 + jj) * VP + lane * 2]);
        a0 += p * v.x; a1 += p * v.y;
      }
    }
    *(__half2*)(Y + (size_t)(q0 + q) * ys + yo + h * hd + lane * 2) = __floats2half2_rn(a0, a1);
  }
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

// attention v2: K/V resident in shared memory, one block per (query-chunk, head).
// qkv layout per token: [h0: q(kd) k(kd) v(hd) | h1: ...], N tokens, stride qs.
#define ATTN_Q 50
__global__ void k_attn2(const __half* __restrict__ QKV, int qs, int qo,
                        __half* __restrict__ Y, int ys, int yo,
                        int N, int kd, int hd) {
  extern __shared__ __half smh[];
  __half* Ks = smh;                      // [N][kd]
  __half* Vs = smh + N * kd;             // [N][hd]
  float* probs = (float*)(Vs + N * hd);  // [8 warps][N]
  float* qsm = probs + 8 * N;            // [8 warps][kd]
  const int h = blockIdx.y, per = 2 * kd + hd;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float scale = rsqrtf((float)kd);
  for (int i = threadIdx.x; i < N * (kd + hd) / 8; i += blockDim.x) {
    // i indexes vec8 slots: first N*kd/8 are K, rest V
    int kslots = N * kd / 8;
    if (i < kslots) {
      int tok = i / (kd / 8), s = i % (kd / 8);
      *(uint4*)&Ks[tok * kd + s * 8] = *(const uint4*)(QKV + (size_t)tok * qs + qo + h * per + kd + s * 8);
    } else {
      int j = i - kslots;
      int tok = j / (hd / 8), s = j % (hd / 8);
      *(uint4*)&Vs[tok * hd + s * 8] = *(const uint4*)(QKV + (size_t)tok * qs + qo + h * per + 2 * kd + s * 8);
    }
  }
  __syncthreads();
  int q0 = blockIdx.x * ATTN_Q;
  for (int q = q0 + warp; q < min(q0 + ATTN_Q, N); q += 8) {
    // load q into per-warp smem
    for (int d = lane; d < kd; d += 32) qsm[warp * kd + d] = __half2float(QKV[(size_t)q * qs + qo + h * per + d]);
    __syncwarp();
    // scores: lane handles tokens lane, lane+32, ...
    float lmax = -1e30f;
    for (int j = lane; j < N; j += 32) {
      float s = 0;
      const __half* kp = Ks + (size_t)j * kd;
#pragma unroll
      for (int d = 0; d < 32; d++) s += qsm[warp * kd + d] * __half2float(kp[d]);
      s *= scale;
      probs[warp * N + j] = s;
      lmax = fmaxf(lmax, s);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(~0u, lmax, o));
    float lsum = 0;
    for (int j = lane; j < N; j += 32) {
      float e = __expf(probs[warp * N + j] - lmax);
      probs[warp * N + j] = e; lsum += e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(~0u, lsum, o);
    float inv = 1.f / lsum;
    __syncwarp();
    // output: lane covers channels lane, lane+32
    float a0 = 0, a1 = 0;
    for (int j = 0; j < N; j++) {
      float p = probs[warp * N + j];
      a0 += p * __half2float(Vs[(size_t)j * hd + lane]);
      a1 += p * __half2float(Vs[(size_t)j * hd + lane + 32]);
    }
    Y[(size_t)q * ys + yo + h * hd + lane] = __float2half(a0 * inv);
    Y[(size_t)q * ys + yo + h * hd + lane + 32] = __float2half(a1 * inv);
  }
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
                           int H, int W, int C) {  // C % 8 == 0, vec8 channels
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cg = C >> 3;
  if (idx >= H * W * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int x0 = pix % W, y0 = pix / W;
  __half2 m[4];
#pragma unroll
  for (int i = 0; i < 4; i++) m[i] = __floats2half2_rn(-6e4f, -6e4f);
  for (int dy = -2; dy <= 2; dy++) {
    int iy = y0 + dy; if (iy < 0 || iy >= H) continue;
    for (int dx = -2; dx <= 2; dx++) {
      int ix = x0 + dx; if (ix < 0 || ix >= W) continue;
      uint4 v = *(const uint4*)(X + ((size_t)iy * W + ix) * xs + xo + c);
      const __half2* h2 = (const __half2*)&v;
#pragma unroll
      for (int i = 0; i < 4; i++) m[i] = __hmax2(m[i], h2[i]);
    }
  }
  *(uint4*)(Y + ((size_t)y0 * W + x0) * ys + yo + c) = *(uint4*)m;
}

__global__ void k_upsample2(const __half* X, int xs, int xo, __half* Y, int ys, int yo,
                            int H, int W, int C) {  // out is 2H x 2W, C % 8 == 0
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Wo = W * 2, Ho = H * 2, cg = C >> 3;
  if (idx >= Ho * Wo * cg) return;
  int g = idx % cg, pix = idx / cg;
  int c = g << 3;
  int ox = pix % Wo, oy = pix / Wo;
  *(uint4*)(Y + ((size_t)oy * Wo + ox) * ys + yo + c) =
      *(const uint4*)(X + ((size_t)(oy >> 1) * W + (ox >> 1)) * xs + xo + c);
}

__global__ void k_copyc(const __half* X, int xs, int xo, __half* Y, int ys, int yo, int HW, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= HW * C) return;
  int c = idx % C, pix = idx / C;
  Y[(size_t)pix * ys + yo + c] = X[(size_t)pix * xs + xo + c];
}

// attention: qkv buffer [N tokens, heads*(2kd+hd)] strided; out [N, heads*hd].
// one block per (token, head); scores in smem.
__global__ void k_attn(const __half* __restrict__ QKV, int qs, int qo,
                       __half* __restrict__ Y, int ys, int yo,
                       int N, int heads, int kd, int hd) {
  int n = blockIdx.x, h = blockIdx.y;
  int per = 2 * kd + hd;
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

// DFL decode + score threshold. box view [H,W,64], cls view [H,W,80].
__global__ void k_decode(const __half* __restrict__ BOX, int bs, int bo,
                         const __half* __restrict__ CLS, int cs, int co,
                         int H, int W, int stride, float conf,
                         float* __restrict__ out, int* __restrict__ cnt, int maxdet, int nc) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * W) return;
  int x = idx % W, y = idx / W;
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
  int slot = atomicAdd(cnt, 1);
  if (slot >= maxdet) return;
  float* o = out + slot * 6;
  o[0] = (cx - d[0]) * stride; o[1] = (cy - d[1]) * stride;
  o[2] = (cx + d[2]) * stride; o[3] = (cy + d[3]) * stride;
  o[4] = score; o[5] = (float)bcls;
}

// letterbox preprocess: BGR u8 HWC (sh x sw) -> RGB fp16 NHWC 640x640, /255, pad 114.
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

// single-block GPU NMS: bitonic sort by score, greedy suppression, compacted output.
#define NMS_CAP 1024
#define MAX_OUT 300
__global__ void k_nms(const float* __restrict__ cand, const int* __restrict__ cnt,
                      float* __restrict__ out, int* __restrict__ outcnt, float thr) {
  __shared__ float bx1[NMS_CAP], by1[NMS_CAP], bx2[NMS_CAP], by2[NMS_CAP], bsc[NMS_CAP];
  __shared__ short bcl[NMS_CAP];
  __shared__ unsigned char dead[NMS_CAP];
  __shared__ short order[NMS_CAP];
  const int tid = threadIdx.x;
  int n = min(*cnt, NMS_CAP);
  for (int i = tid; i < NMS_CAP; i += blockDim.x) {
    if (i < n) {
      const float* c = cand + i * 6;
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

static void runOp(Net& net, const Op& op, cudaStream_t st) {
  const int TB = 256;
  switch (op.kind) {
    case CONV: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = y.H * y.W * op.Cout;
      if (op.g == 1 && op.Cin == 3) {   // first conv
        dim3 g0((y.H * y.W + 255) / 256, op.Cout / 16);
        k_conv0<<<g0, 256, 0, st>>>(x.p, x.H, x.W, (__half*)y.p, y.H, y.W, op.Cout,
                                    net.weights + op.woff, net.bias + op.boff, op.act);
      } else if (op.g == 1 && op.Cin >= 8) {
        View r{};
        bool hasRes = op.b >= 0;
        if (hasRes) r = view(net, op.b);
        int M = y.H * y.W;
        int b64 = ((M + 63) / 64) * ((op.Cout + 63) / 64);
        bool kone = (op.k == 1 && op.s == 1 && op.p == 0);
        long Ktot = (long)op.k * op.k * op.Cin;
        auto launch = [&](auto kern, int bm, int bn, int threads) {
          dim3 grid((M + bm - 1) / bm, (op.Cout + bn - 1) / bn);
          kern<<<grid, threads, 0, st>>>(
              x.p, x.H, x.W, x.s, x.o, (__half*)y.p, y.H, y.W, y.s, y.o,
              net.weights + op.woff, Ktot, net.bias + op.boff,
              hasRes ? r.p : nullptr, hasRes ? r.s : 0, hasRes ? r.o : 0,
              op.Cin, op.Cout, op.k, op.s, op.p, op.act);
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
        if (wide)                                 LAUNCH_TILE(64, 128, 3, 2, 4);
        else if (op.Cout <= 32 && M >= 6400)      LAUNCH_TILE(64, 32, 3, 2, 2);
        else if (b64 >= 24)                       LAUNCH_TILE(64, 64, 3, 2, 2);
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
        int nthr = y.H * y.W * (op.Cin / 8);
        k_dwconv8<<<(nthr + TB - 1) / TB, TB, 0, st>>>(
            x.p, x.H, x.W, x.s, x.o, (__half*)y.p, y.H, y.W, y.s, y.o,
            net.weights + op.woff, net.bias + op.boff,
            hasRes ? r.p : nullptr, hasRes ? r.s : 0, hasRes ? r.o : 0,
            op.Cin, op.k, op.s, op.p, op.act);
      }
      break;
    }
    case ADD: {
      View a = view(net, op.a), b = view(net, op.b), y = view(net, op.out);
      int total = a.H * a.W * a.C;
      k_add<<<(total + TB - 1) / TB, TB, 0, st>>>(a.p, a.s, a.o, b.p, b.s, b.o,
                                                 (__half*)y.p, y.s, y.o, a.H * a.W, a.C);
      break;
    }
    case MAXPOOL5: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = x.H * x.W * (x.C / 8);
      k_maxpool5<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.H, x.W, x.C);
      break;
    }
    case UPSAMPLE2: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = y.H * y.W * 4 * (x.C / 8);
      k_upsample2<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.H, x.W, x.C);
      break;
    }
    case COPYC: {
      View x = view(net, op.a), y = view(net, op.out);
      int total = x.H * x.W * x.C;
      k_copyc<<<(total + TB - 1) / TB, TB, 0, st>>>(x.p, x.s, x.o, (__half*)y.p, y.s, y.o, x.H * x.W, x.C);
      break;
    }
    case ATTN: {
      View q = view(net, op.a), y = view(net, op.out);
      int N = q.H * q.W;
      if (op.kd == 32 && op.hd == 64 && N == 400) {
        // tensor-core attention via the implicit-GEMM kernel:
        //   scores[h] = Q[400x32] @ K^T   (K rows read in place from the qkv buffer)
        //   P = softmax(scores * 1/sqrt(kd));  out[h] = P[400x400] @ V[400x64] via V^T
        const int per = 2 * op.kd + op.hd;
        for (int h = 0; h < op.heads; h++) {
          dim3 g1((N + 63) / 64, (N + 63) / 64);
          k_conv_mma<64, 64, 3, true><<<g1, 128, 0, st>>>(
              q.p, N, 1, q.s, q.o + h * per,
              net.attnP + (size_t)h * N * N, N, 1, N, 0,
              q.p + q.o + h * per + op.kd, q.s, net.zerobias,
              nullptr, 0, 0, op.kd, N, 1, 1, 0, 0);
        }
        k_softmax_rows<<<(op.heads * N + 7) / 8, 256, 0, st>>>(
            net.attnP, op.heads * N, N, rsqrtf((float)op.kd));
        k_vtrans<<<(op.heads * op.hd * N + 255) / 256, 256, 0, st>>>(
            q.p, q.s, q.o, net.vt, N, op.kd, op.hd, op.heads);
        for (int h = 0; h < op.heads; h++) {
          dim3 g2((N + 31) / 32, (op.hd + 31) / 32);
          k_conv_mma<32, 32, 3, true><<<g2, 128, 0, st>>>(
              net.attnP + (size_t)h * N * N, N, 1, N, 0,
              (__half*)y.p, N, 1, y.s, y.o + h * op.hd,
              net.vt + (size_t)h * op.hd * N, N, net.zerobias,
              nullptr, 0, 0, N, op.hd, 1, 1, 0, 0);
        }
      } else {
        dim3 grid(N, op.heads);
        size_t smem = N * sizeof(float);
        k_attn<<<grid, 128, smem, st>>>(q.p, q.s, q.o, (__half*)y.p, y.s, y.o, N, op.heads, op.kd, op.hd);
      }
      break;
    }
    case DECODE: {
      View b = view(net, op.a), c = view(net, op.b);
      int total = b.H * b.W;
      k_decode<<<(total + TB - 1) / TB, TB, 0, st>>>(b.p, b.s, b.o, c.p, c.s, c.o,
          b.H, b.W, op.stride, 0.25f, net.dets, net.detcnt, Net::MAXDET, 80);
      break;
    }
  }
}

static void forward(Net& net, cudaStream_t st) {
  CK(cudaMemsetAsync(net.detcnt, 0, sizeof(int), st));
  for (auto& op : net.ops) runOp(net, op, st);
  k_nms<<<1, 256, 0, st>>>(net.dets, net.detcnt, net.outdets, net.outcnt, 0.45f);
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

// ------------------------- CPU NMS -------------------------
struct Det { float x1, y1, x2, y2, sc; int cls; };
static float iou(const Det& a, const Det& b) {
  float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
  float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
  float inter = w * h;
  float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}
static std::vector<Det> nms(std::vector<Det> v, float thr) {
  std::sort(v.begin(), v.end(), [](const Det& a, const Det& b) { return a.sc > b.sc; });
  std::vector<Det> keep;
  std::vector<bool> dead(v.size(), false);
  for (size_t i = 0; i < v.size(); i++) {
    if (dead[i]) continue;
    keep.push_back(v[i]);
    for (size_t j = i + 1; j < v.size(); j++)
      if (!dead[j] && v[j].cls == v[i].cls && iou(v[i], v[j]) > thr) dead[j] = true;
  }
  return keep;
}

// NMS ran on the GPU inside the graph; just fetch the compacted results.
static std::vector<Det> fetchNMS(Net& net, cudaStream_t st) {
  int cnt; CK(cudaMemcpy(&cnt, net.outcnt, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> d(cnt * 6);
  CK(cudaMemcpy(d.data(), net.outdets, cnt * 6 * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<Det> v;
  for (int i = 0; i < cnt; i++)
    v.push_back({d[i*6], d[i*6+1], d[i*6+2], d[i*6+3], d[i*6+4], (int)d[i*6+5]});
  return v;
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
  printf("yolo11cuda — hand-written CUDA inference engine for YOLO11\n\n"
         "usage: yolo11cuda <mode> [model-dir] [iters] [--image path]\n\n"
         "modes:\n"
         "  detect    run one inference, print detections (--image for any jpg/png)\n"
         "  bench     net-only benchmark (stream vs CUDA graph)\n"
         "  pipeline  end-to-end benchmark: H2D + preprocess + net + decode + NMS\n"
         "  profile   per-op timing breakdown\n"
         "  dump      write per-op outputs for test/compare.py\n\n"
         "model-dir defaults to build/yolo11n (create with: make export MODEL=yolo11n)\n"
         "env: ACC16=1 -> pure fp16 mma accumulation (fastest, ~2-3%% err)\n"
         "     ACC32=1 -> pure fp32 accumulation (exact-est). default: hybrid K=64 window\n");
}

// ------------------------- main -------------------------
int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "detect";
  std::string dir = "build/yolo11n";
  int iterArg = -1;
  std::string image;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--image" && i + 1 < argc) image = argv[++i];
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
  loadGraph(net, dir);
  if (mode != "dump" && !getenv("NOFUSE")) fuseAdds(net);

  // default input: the exporter's pre-letterboxed reference image
  auto in = readFile(dir + "/input.f16");
  Buf& ib = net.bufs[net.tens[0].buf];
  CK(cudaMemcpy(ib.p, in.data(), in.size(), cudaMemcpyHostToDevice));

  cudaStream_t st; CK(cudaStreamCreate(&st));
  CK(cudaFuncSetAttribute(k_attn_qk, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
  CK(cudaFuncSetAttribute(k_attn_av, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));

  auto makeGraph = [&](cudaGraphExec_t& gexec) {
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    forward(net, st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&gexec, graph, 0));
    CK(cudaGraphDestroy(graph));
  };

  if (mode == "dump") {
    CK(cudaMemsetAsync(net.detcnt, 0, sizeof(int), st));
    system(("mkdir -p " + dir + "/gpu").c_str());
    for (size_t k = 0; k < net.ops.size(); k++) {
      runOp(net, net.ops[k], st);
      CK(cudaStreamSynchronize(st));
      CK(cudaGetLastError());
      const Op& op = net.ops[k];
      if (op.kind == DECODE) continue;
      View y = view(net, op.out);
      // copy strided view to contiguous host
      std::vector<__half> host((size_t)y.H * y.W * y.C);
      CK(cudaMemcpy2D(host.data(), y.C * sizeof(__half), y.p + y.o, y.s * sizeof(__half),
                      y.C * sizeof(__half), (size_t)y.H * y.W, cudaMemcpyDeviceToHost));
      char path[256]; snprintf(path, sizeof path, "%s/gpu/op%03zu.bin", dir.c_str(), k);
      FILE* f = fopen(path, "wb");
      fwrite(host.data(), sizeof(__half), host.size(), f);
      fclose(f);
    }
    printf("dumped %zu ops\n", net.ops.size());
  } else if (mode == "detect") {
    float scale = 1.f; int top = 0, left = 0;
    if (!image.empty()) {   // arbitrary image via stb_image + GPU letterbox
      int sh, sw;
      uint8_t* img = loadImageBGR(image, sh, sw);
      uint8_t* dimg; CK(cudaMalloc(&dimg, (size_t)sh * sw * 3));
      CK(cudaMemcpy(dimg, img, (size_t)sh * sw * 3, cudaMemcpyHostToDevice));
      stbi_image_free(img);
      scale = std::min(640.f / sh, 640.f / sw);
      int nh = (int)roundf(sh * scale), nw = (int)roundf(sw * scale);
      top = (640 - nh) / 2; left = (640 - nw) / 2;
      k_preprocess<<<(640 * 640 + 255) / 256, 256, 0, st>>>(
          dimg, sh, sw, ib.p, 640, 640, scale, top, left, nh, nw);
    }
    forward(net, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    auto keep = fetchNMS(net, st);
    printf("kept=%zu%s\n", keep.size(), image.empty() ? "" : "  (boxes in original image coords)");
    for (auto& d : keep)   // map back to original image coordinates when letterboxed
      printf("cls=%2d score=%.4f box=(%.1f, %.1f, %.1f, %.1f)\n", d.cls, d.sc,
             (d.x1 - left) / scale, (d.y1 - top) / scale,
             (d.x2 - left) / scale, (d.y2 - top) / scale);
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
    float scale = std::min(640.f / sh, 640.f / sw);
    int nh = (int)roundf(sh * scale), nw = (int)roundf(sw * scale);
    int top = (640 - nh) / 2, left = (640 - nw) / 2;
    Buf& inb = net.bufs[net.tens[0].buf];

    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    CK(cudaMemcpyAsync(dimg, himg, raw.size(), cudaMemcpyHostToDevice, st));
    k_preprocess<<<(640 * 640 + 255) / 256, 256, 0, st>>>(
        dimg, sh, sw, inb.p, 640, 640, scale, top, left, nh, nw);
    forward(net, st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&gexec, graph, 0));
    CK(cudaGraphDestroy(graph));

    CK(cudaGraphLaunch(gexec, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    auto keep = fetchNMS(net, st);
    printf("kept=%zu\n", keep.size());
    for (auto& d : keep)
      printf("cls=%2d score=%.4f box=(%.1f, %.1f, %.1f, %.1f)\n", d.cls, d.sc, d.x1, d.y1, d.x2, d.y2);

    int iters = iterArg > 0 ? iterArg : 300;
    for (int i = 0; i < 20; i++) { CK(cudaGraphLaunch(gexec, st)); CK(cudaStreamSynchronize(st)); fetchNMS(net, st); }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      CK(cudaGraphLaunch(gexec, st));
      CK(cudaStreamSynchronize(st));
      fetchNMS(net, st);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("end-to-end pipeline (H2D + preprocess + net + decode + NMS): %.3f ms/frame (%.1f fps)\n",
           ms, 1000.0 / ms);
  } else if (mode == "profile") {
    int iters = 50;
    std::vector<float> acc(net.ops.size(), 0.f);
    std::vector<cudaEvent_t> ev(net.ops.size() + 1);
    for (auto& e : ev) cudaEventCreate(&e);
    for (int i = 0; i < 10; i++) forward(net, st);
    CK(cudaStreamSynchronize(st));
    for (int it = 0; it < iters; it++) {
      CK(cudaMemsetAsync(net.detcnt, 0, sizeof(int), st));
      cudaEventRecord(ev[0], st);
      for (size_t k = 0; k < net.ops.size(); k++) {
        runOp(net, net.ops[k], st);
        cudaEventRecord(ev[k + 1], st);
      }
      CK(cudaStreamSynchronize(st));
      for (size_t k = 0; k < net.ops.size(); k++) {
        float ms; cudaEventElapsedTime(&ms, ev[k], ev[k + 1]);
        acc[k] += ms;
      }
    }
    const char* names[] = {"CONV", "ADD", "MAXPOOL5", "UPSAMPLE2", "COPYC", "ATTN", "DECODE"};
    std::vector<int> order(net.ops.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return acc[a] > acc[b]; });
    float tot = 0; for (float a : acc) tot += a;
    printf("total %.3f ms/iter — top ops:\n", tot / iters);
    for (int i = 0; i < 25 && i < (int)order.size(); i++) {
      int k = order[i];
      const Op& op = net.ops[k];
      View y = view(net, op.kind == DECODE ? op.a : op.out);
      printf("op%03d %-9s %4dx%-4d Cin=%-3d Cout=%-3d k=%d s=%d g=%d res=%d  %7.1f us  (%4.1f%%)\n",
             k, names[op.kind], y.H, y.W, op.Cin, op.Cout, op.k, op.s, op.g, op.b >= 0 && op.kind == CONV,
             acc[k] / iters * 1000.f, 100.f * acc[k] / tot);
    }
  } else if (mode == "bench") {
    int iters = iterArg > 0 ? iterArg : 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 20; i++) forward(net, st);
    CK(cudaStreamSynchronize(st));
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; i++) forward(net, st);
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
