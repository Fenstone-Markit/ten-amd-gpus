/* fp8-convert-check-01: check the bit trick that turns e4m3fn bytes into fp16 (value x 2^-8) and bf16 (value x 2^-120)
   against an exact decode, for all 256 byte values, using the same 32-bit expressions the kernel uses on two bytes at once.
   Repack per 4 bytes is [b0, b2, b1, b3] so that lo = (w << 8) & 0xFF00FF00 holds b0, b1 and hi = w & 0xFF00FF00 holds b2, b3. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static double e4m3_exact(uint8_t x) {          /* e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa; 0x7F/0xFF are NaN; no infinities */
  int s = x >> 7, e = (x >> 3) & 15, m = x & 7; double v;
  if (e == 15 && m == 7) return NAN;
  v = e ? (1.0 + m / 8.0) * ldexp(1.0, e - 7) : (m / 8.0) * ldexp(1.0, -6);
  return s ? -v : v;
}
static double f16_bits(uint16_t h) { int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; double v;
  if (e == 31) return m ? NAN : INFINITY; v = e ? (1.0 + m / 1024.0) * ldexp(1.0, e - 15) : (m / 1024.0) * ldexp(1.0, -14); return s ? -v : v; }
static double bf16_bits(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t cvt_f16(uint32_t t)  { return (t & 0x80008000u) | ((t & 0x7F007F00u) >> 1); }   /* same as the kernel */
static uint32_t cvt_bf16(uint32_t t) { return (t & 0x80008000u) | ((t & 0x7F007F00u) >> 4); }
int main(void) {
  int bad16 = 0, badbf = 0, nan = 0;
  for (int a = 0; a < 256; a++) {
    int b = (a * 37 + 11) & 255, c = (a * 101 + 3) & 255, d = (a * 59 + 200) & 255;      /* exercise every byte in every slot */
    uint8_t v[4] = {(uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d};                    /* logical order b0..b3 */
    uint32_t w = v[0] | (v[2] << 8) | (v[1] << 16) | ((uint32_t)v[3] << 24);           /* repacked [b0, b2, b1, b3] */
    uint32_t lo = (w << 8) & 0xFF00FF00u, hi = w & 0xFF00FF00u;
    uint32_t L16 = cvt_f16(lo), H16 = cvt_f16(hi), Lbf = cvt_bf16(lo), Hbf = cvt_bf16(hi);
    uint16_t o16[4] = {L16 & 0xFFFF, L16 >> 16, H16 & 0xFFFF, H16 >> 16}, obf[4] = {Lbf & 0xFFFF, Lbf >> 16, Hbf & 0xFFFF, Hbf >> 16};
    for (int i = 0; i < 4; i++) {
      double want = e4m3_exact(v[i]);
      if (isnan(want)) { if (i == 0) nan++; continue; }
      double g16 = f16_bits(o16[i]) * 256.0, gbf = bf16_bits(obf[i]) * ldexp(1.0, 120);
      if (g16 != want) { if (bad16++ < 5) printf("fp16 MISMATCH byte 0x%02x slot %d: want %g got %g\n", v[i], i, want, g16); }
      if (gbf != want) { if (badbf++ < 5) printf("bf16 MISMATCH byte 0x%02x slot %d: want %g got %g\n", v[i], i, want, gbf); }
    }
  }
  printf("checked all 256 bytes in all 4 slots (NaN codes 0x7f/0xff skipped: %d): fp16 mismatches %d, bf16 mismatches %d; max finite %g\n",
         nan, bad16, badbf, e4m3_exact(0x7E));
  return bad16 || badbf;
}
