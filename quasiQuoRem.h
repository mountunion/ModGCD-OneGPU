/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = qf * yf + remf.
    
    K. Weber

*/

#include <cstdint>

constexpr int RCP_THRESHOLD_EXPT = 22;
constexpr int RCP_THRESHOLD_CLZ  = 32 - RCP_THRESHOLD_EXPT;
constexpr uint32_t RCP_THRESHOLD = 1 << RCP_THRESHOLD_EXPT;

__device__
inline
float
fastReciprocal(float yf)
{
  float rf;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rf) : "f"(yf));
  return rf;
}

//  quasiQuoRem computes a quotient qf such that xf - qf * yf < 2 * yf.
//  Precondition: xf and yf are truncated integers and 
//  if CHECK_RCP == true,
//  then 3*2^22 > xf >= 1.0 && 2^22 > yf >= 1.0
//  else 2^22 > xf, yf >= 1.0.
//  Note that __fdividef(x, y) is accurate to 2 ulp:
//  when yf >= 4.0, 0 <= xf/yf < 3*2^20 < 2^22 means 2 ulp <= 0.5,
//  so truncf(__fdividef(xf, yf)) should give either the true quotient or one less.
//  When yf == 2.0, 0 <= xf/2.0 < 3*2^20, so 2 ulp <= 0.5.
//  When yf == 3.0, 0 <= xf/3.0 < 2^22, so 2 ulp <= 0.5.
//  When yf == 1.0, the quotient should always be exact and equal to xf, 
//  since __fdividef(xf, yf) is based on multiplication by the reciprocal.
//  Also note that, when yf < xf < 2 * yf, that 1.0 + 1/yf <= qf <= 2.0 - 1/yf, 
//  and 1
//  with 2 ulp <= 2 * 2^
//  so trunc(qf) == 1, which
//  is the exact value of the true quotient.
template <bool CHECK_RCP>
__device__
inline
uint32_t
quasiQuoRem(float& xf, float yf)
{
  float qf = truncf(__fmul_rz(xf, fastReciprocal(yf)));
  xf = __fmaf_rz(qf, -yf, xf); 
  if (CHECK_RCP && xf < 0.0f)
    xf += yf, qf -= 1.0f;
  return __float2uint_rz(qf);
}
  
//  Computes an approximation for x / y, when x, y >= 2^21.
//  Approximation could be too small by 1 or 2.
//  The estimate of q from multiplying by the reciprocal here could be too high or too low by 1;
//  make it too low by 1 or 2, by subtracting 1.0 BEFORE truncating toward zero.
__device__
inline
uint32_t
quasiQuo2(uint32_t x, uint32_t y)
{ 
  return __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_rz(y)), -1.0f));
}
  
//  For the case 2^32 > x >= 2^22 > y > 0.
//  Using floating point division here is slightly faster than integer quotient 
//  and remainder for many architectures, but not all.
template <bool CHECK_RCP, bool QUASI>
__device__
inline
uint32_t
quoRem(float& __restrict__ xf, float& __restrict__ yf, uint32_t x, uint32_t y)
{
  uint32_t q;
  if (QUASI)
    {
      int i = __clz(y) - RCP_THRESHOLD_CLZ;
      q = quasiQuo2(x, y << i) << i;
    }
  else
    q = x / y;
  xf = __uint2float_rz(x - q * y);
  yf = __uint2float_rz(y);
  if (QUASI)
    q += quasiQuoRem<CHECK_RCP>(xf, yf);
  return q;
}

//  Faster divide possible when x and y are close in size.
//  Precondition: 2^32 > x, y >= RCP_THRESHOLD, so 1 <= x / y < 2^RCP_THRESHOLD_CLZ.
//  Could produce a quotient that's too small by 1--but modInv can tolerate that.
//  ***********THIS STILL NEEDS TO BE CHECKED MATHEMATICALLY***********
__device__
inline
uint32_t
quasiQuoRem(uint32_t& x, uint32_t y)
{ 
//  Computes an approximation q for x / y, when x, y >= RCP_THRESHOLD.
//  q could be too small by 1 or 2.
//  The estimate of q from multiplying by the reciprocal here could be too high or too low by 1;
//  make it too low by 1 or 2, by subtracting 1.0 BEFORE truncating toward zero.
  uint32_t q = __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_rz(y)), -1.0f));
  x -= q * y; 
  if (x >= y)  //  Now x < 3 * y.
    x -= y, q += 1;
  return q;               //  Now x < 2 * y, but unlikely that x >= y.
}
