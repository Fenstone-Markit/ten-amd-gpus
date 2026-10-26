// fp8-rdna3-lib-01: FP8 (e4m3fn) MoE kernels for RDNA3 (gfx1100) as a plain shared library with extern "C" launchers, for
// vLLM through ctypes and a torch custom op. Weights in vLLM's layout: w13 [E][2I][H] (gate rows, then up rows), w2 [E][H][I],
// e4m3 bytes REPACKED once per 4 bytes as [b0, b2, b1, b3]; scales are vLLM's weight_scale_inv as loaded, fp32, 128 x 128 blocks:
// w13 [E][2I/128][H/128], w2 [E][H/128][I/128]. Activations and outputs fp16 or bf16.
// Decode (few tokens): fp8-moe-02's two kernels (gate-up with SiLU x up; down with routing weights, summed over top-k, no atomics),
// now taking the scales as loaded and applying the 2^8 (fp16) or 2^120 (bf16) of the bit-trick conversion inside.
// Prefill: fp8-gemm-02's WMMA GEMM in vLLM's sorted-token format (moe_align_block_size with block 64): each 64-row block belongs to
// one expert (expert_ids), its rows are pair ids t*topk+k from sorted_token_ids, padding ids (>= T*topk) are skipped. Mode 1 gathers
// x[id / topk] and writes C1[id][2I]; mode 2 reads h[id] and writes C2[id][H] times topk_weights[id]. SiLU x up and the sum over
// top-k stay with vLLM's own ops. Exact: e4m3 to 16-bit through fp32, block scales applied in fp32 per 128-wide K block.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef __bf16 b2 __attribute__((ext_vector_type(2)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

template <bool BF16> __device__ __forceinline__ float dot2(uint32_t a, uint32_t w, float c) {
  if constexpr (BF16) return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(b2, a), __builtin_bit_cast(b2, w), c, false);
  else return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, w), c, false);
}
template <bool BF16> __device__ __forceinline__ uint32_t cvt(uint32_t t) {
  return BF16 ? ((t & 0x80008000u) | ((t & 0x7F007F00u) >> 4)) : ((t & 0x80008000u) | ((t & 0x7F007F00u) >> 1));
}
template <bool BF16> __device__ __forceinline__ uint16_t out_cvt(float v) {
  if constexpr (BF16) { uint32_t u = __float_as_uint(v); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
  else return __builtin_bit_cast(uint16_t, (_Float16)v);
}
template <bool BF16> __device__ __forceinline__ float fold() { return BF16 ? 0x1p120f : 256.f; }
__device__ __forceinline__ float wred(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}
template <bool BF16> __device__ __forceinline__ float dot16_fp8(uint4 w4, uint4 a, uint4 b, float c) {
  const uint32_t ww[4] = {w4.x, w4.y, w4.z, w4.w}, xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int g = 0; g < 4; g++) { c = dot2<BF16>(xs[2 * g], cvt<BF16>((ww[g] << 8) & 0xFF00FF00u), c); c = dot2<BF16>(xs[2 * g + 1], cvt<BF16>(ww[g] & 0xFF00FF00u), c); }
  return c;
}

// ---------------- decode ----------------
template <bool BF16, int U>
__global__ __launch_bounds__(256) void moe_gateup(const uint8_t* __restrict__ W1, const uint16_t* __restrict__ X, const float* __restrict__ S1,
                                                  const int* __restrict__ ids, uint16_t* __restrict__ Hout, int TOPK, int I, int H) {
  const int lane = threadIdx.x & 31, j = blockIdx.x * 8 + (threadIdx.x >> 5), p = blockIdx.y;
  if (j >= I) return;
  const int t = p / TOPK, e = ids[p], HB = H >> 7, IB2 = (2 * I) >> 7;
  const uint8_t* wg = W1 + ((size_t)e * 2 * I + j) * H; const uint8_t* wu = W1 + ((size_t)e * 2 * I + I + j) * H;
  const uint16_t* x = X + (size_t)t * H;
  const float* sg = S1 + ((size_t)e * IB2 + (j >> 7)) * HB; const float* su = S1 + ((size_t)e * IB2 + ((I + j) >> 7)) * HB;
  float ag = 0.f, au = 0.f;
  for (int k0 = lane * 16; k0 < H; k0 += 512 * U) {
    uint4 g4[U], u4[U];
#pragma unroll
    for (int u = 0; u < U; u++) if (k0 + 512 * u < H) { g4[u] = *(const uint4*)(wg + k0 + 512 * u); u4[u] = *(const uint4*)(wu + k0 + 512 * u); }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int k = k0 + 512 * u; if (k >= H) break;
      const uint4 a = *(const uint4*)(x + k), b = *(const uint4*)(x + k + 8);
      ag += dot16_fp8<BF16>(g4[u], a, b, 0.f) * sg[k >> 7];
      au += dot16_fp8<BF16>(u4[u], a, b, 0.f) * su[k >> 7];
    }
  }
  ag = wred(ag) * fold<BF16>(); au = wred(au) * fold<BF16>();
  if (lane == 0) Hout[(size_t)p * I + j] = out_cvt<BF16>(ag / (1.f + __expf(-ag)) * au);
}
template <bool BF16, int U>
__global__ __launch_bounds__(256) void moe_down(const uint8_t* __restrict__ W2, const uint16_t* __restrict__ Hin, const float* __restrict__ S2,
                                                const int* __restrict__ ids, const float* __restrict__ tw, uint16_t* __restrict__ Y, int TOPK, int I, int H) {
  const int lane = threadIdx.x & 31, n = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y;
  if (n >= H) return;
  const int IB = I >> 7, HB = H >> 7;
  float acc = 0.f;
  for (int kk = 0; kk < TOPK; kk++) {
    const int p = t * TOPK + kk, e = ids[p]; const float rw = tw[p];
    const uint8_t* w = W2 + ((size_t)e * H + n) * I; const uint16_t* h = Hin + (size_t)p * I;
    const float* s = S2 + ((size_t)e * HB + (n >> 7)) * IB;
    for (int k0 = lane * 16; k0 < I; k0 += 512 * U) {
      uint4 w4[U];
#pragma unroll
      for (int u = 0; u < U; u++) if (k0 + 512 * u < I) w4[u] = *(const uint4*)(w + k0 + 512 * u);
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int k = k0 + 512 * u; if (k >= I) break;
        acc += dot16_fp8<BF16>(w4[u], *(const uint4*)(h + k), *(const uint4*)(h + k + 8), 0.f) * (rw * s[k >> 7]);
      }
    }
  }
  acc = wred(acc) * fold<BF16>();
  if (lane == 0) Y[(size_t)t * H + n] = out_cvt<BF16>(acc);
}

// ---------------- prefill (WMMA, sorted-token format) ----------------
constexpr int BM = 64, BN = 64, BK = 64, LDP = BK + 8;
template <bool BF16> __device__ __forceinline__ v8f mma(uint4 a0, uint4 a1, uint4 b0, uint4 b1, v8f c) {
  const v8u av = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  if constexpr (BF16) return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(v16s, av), __builtin_bit_cast(v16s, bv), c);
  else return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(v16h, av), __builtin_bit_cast(v16h, bv), c);
}
template <bool BF16> __device__ __forceinline__ uint32_t fp8x2_to16(uint32_t b0, uint32_t b1) {
  const float f0 = __uint_as_float(((b0 & 0x80u) << 24) | ((b0 & 0x7Fu) << 20)) * 0x1p120f;
  const float f1 = __uint_as_float(((b1 & 0x80u) << 24) | ((b1 & 0x7Fu) << 20)) * 0x1p120f;
  if constexpr (BF16) return (__float_as_uint(f0) >> 16) | (__float_as_uint(f1) & 0xFFFF0000u);
  else { const _Float16 g0 = (_Float16)f0, g1 = (_Float16)f1; return (uint32_t)__builtin_bit_cast(uint16_t, g0) | ((uint32_t)__builtin_bit_cast(uint16_t, g1) << 16); }
}
// MODE 1: A = x [T][K], row = id / topk; MODE 2: A = h [T*topk][K], row = id, output times tw[id]. C [T*topk][N].
template <bool BF16, int MODE>
__global__ __launch_bounds__(128) void moe_gemm(const uint16_t* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
                                                const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids, const int* __restrict__ npost,
                                                const float* __restrict__ tw, uint16_t* __restrict__ C, int Mtot, int N, int K, int topk) {
  __shared__ __attribute__((aligned(16))) uint16_t As[BM * LDP];
  __shared__ __attribute__((aligned(16))) uint16_t Bs[BN * LDP];
  __shared__ int rid[BM];
  const int mb = blockIdx.y;
  if (mb * BM >= *npost) return;
  const int e = expert_ids[mb];
  if (e < 0) return;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave >> 1, wn = wave & 1, n0 = blockIdx.x * BN;
  if (tid < BM) rid[tid] = sorted_ids[mb * BM + tid];
  __syncthreads();
  W += (size_t)e * N * K; S += (size_t)e * (N >> 7) * (K >> 7);
  const int KB = K >> 7;
  v8f acc[2][2], part[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 2; j++) { acc[i][j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0}; part[i][j] = acc[i][j]; }
  uint4 ra[4], rw[2];
  auto gload = [&](int k0) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int idx = tid + r * 128, row = idx >> 3, c8 = (idx & 7) * 8, id = rid[row];
      ra[r] = id < Mtot ? *(const uint4*)(A + (size_t)(MODE == 1 ? id / topk : id) * K + k0 + c8) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int r = 0; r < 2; r++) { const int idx = tid + r * 128, row = idx >> 2, c16 = (idx & 3) * 16; rw[r] = *(const uint4*)(W + (size_t)(n0 + row) * K + k0 + c16); }
  };
  gload(0);
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int r = 0; r < 4; r++) { const int idx = tid + r * 128, row = idx >> 3, c8 = (idx & 7) * 8; *(uint4*)(As + row * LDP + c8) = ra[r]; }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int idx = tid + r * 128, row = idx >> 2, c16 = (idx & 3) * 16;
      // repacked bytes per word: [b0, b2, b1, b3]; k order pairs are (b0, b1) and (b2, b3)
      const uint32_t qq[4] = {rw[r].x, rw[r].y, rw[r].z, rw[r].w}; uint32_t o[8];
#pragma unroll
      for (int g = 0; g < 4; g++) {
        o[2 * g] = fp8x2_to16<BF16>(qq[g] & 0xFF, (qq[g] >> 16) & 0xFF);
        o[2 * g + 1] = fp8x2_to16<BF16>((qq[g] >> 8) & 0xFF, qq[g] >> 24);
      }
      *(uint4*)(Bs + row * LDP + c16) = make_uint4(o[0], o[1], o[2], o[3]);
      *(uint4*)(Bs + row * LDP + c16 + 8) = make_uint4(o[4], o[5], o[6], o[7]);
    }
    __syncthreads();
    if (k0 + BK < K) gload(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint4 af[2][2], bf[2][2];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const uint16_t* p = As + (wm * 32 + i * 16 + (lane & 15)) * LDP + kk;
        af[i][0] = *(const uint4*)p; af[i][1] = *(const uint4*)(p + 8);
        const uint16_t* q = Bs + (wn * 32 + i * 16 + (lane & 15)) * LDP + kk;
        bf[i][0] = *(const uint4*)q; bf[i][1] = *(const uint4*)(q + 8);
      }
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) part[i][j] = mma<BF16>(af[i][0], af[i][1], bf[j][0], bf[j][1], part[i][j]);
    }
    if ((k0 & 127) == 64) {
      const float s = S[(size_t)(n0 >> 7) * KB + (k0 >> 7)];
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) { acc[i][j] += part[i][j] * s; part[i][j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0}; }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int n = n0 + wn * 32 + j * 16 + (lane & 15);
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int id = rid[wm * 32 + i * 16 + 2 * q + (lane >> 4)];
        if (id < Mtot) C[(size_t)id * N + n] = out_cvt<BF16>(acc[i][j][q] * (MODE == 2 ? tw[id] : 1.f));
      }
    }
}

// ---------------- launchers ----------------
// All pointers are device pointers; stream is the caller's (torch's current) stream. Return value: hipGetLastError().
extern "C" int fp8r3_moe_decode(void* stream, int bf16, int U, const void* W1, const float* S1, const void* W2, const float* S2,
                                const void* X, const int* ids, const float* tw, void* Hbuf, void* Y, int T, int TOPK, int I, int H) {
  hipStream_t s = (hipStream_t)stream;
  dim3 g1((I + 7) / 8, T * TOPK), g2((H + 7) / 8, T);
#define DEC(BF, UU) { moe_gateup<BF, UU><<<g1, 256, 0, s>>>((const uint8_t*)W1, (const uint16_t*)X, S1, ids, (uint16_t*)Hbuf, TOPK, I, H); \
                     moe_down<BF, UU><<<g2, 256, 0, s>>>((const uint8_t*)W2, (const uint16_t*)Hbuf, S2, ids, tw, (uint16_t*)Y, TOPK, I, H); }
  if (bf16) { if (U == 4) DEC(true, 4) else if (U == 1) DEC(true, 1) else DEC(true, 2) }
  else { if (U == 4) DEC(false, 4) else if (U == 1) DEC(false, 1) else DEC(false, 2) }
#undef DEC
  return (int)hipGetLastError();
}
extern "C" int fp8r3_moe_gemm(void* stream, int bf16, int mode, const void* A, const void* W, const float* S, const int* sorted_ids,
                              const int* expert_ids, const int* npost, const float* tw, void* C, int Mtot, int N, int K, int topk, int nblocks) {
  hipStream_t s = (hipStream_t)stream;
  dim3 g(N / BN, nblocks);
#define GEMM(BF, MD) moe_gemm<BF, MD><<<g, 128, 0, s>>>((const uint16_t*)A, (const uint8_t*)W, S, sorted_ids, expert_ids, npost, tw, (uint16_t*)C, Mtot, N, K, topk)
  if (bf16) { if (mode == 2) GEMM(true, 2); else GEMM(true, 1); }
  else { if (mode == 2) GEMM(false, 2); else GEMM(false, 1); }
#undef GEMM
  return (int)hipGetLastError();
}
// repack e4m3 bytes in place, per 4 bytes [b0, b1, b2, b3] -> [b0, b2, b1, b3] (n multiple of 4)
__global__ void repack_k(uint32_t* w, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t v = w[i]; w[i] = (v & 0xFF0000FFu) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8);
  }
}
extern "C" int fp8r3_repack(void* stream, void* w, size_t nbytes) {
  repack_k<<<1024, 256, 0, (hipStream_t)stream>>>((uint32_t*)w, nbytes / 4);
  return (int)hipGetLastError();
}
