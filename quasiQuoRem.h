/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = qf * yf + remf.
    
    K. Weber

*/

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
