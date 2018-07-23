/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = quotient * yf + remainder.
    
    K. Weber

*/

// Defines range of valid input for quasiQuoRem.
static constexpr int RCP_THRESHOLD_EXPT = 22;

__device__
static
inline
float
fastReciprocal(float y)
{
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y));
  return r;
}

//  quasiQuoRem computes a quasi-quotient q and quasi-remainder r = x - q * y
//  such that 0 <= r < 2 * y.
//  Preconditions: 
//    x and y are integers
//    0 < x < RCP_THRESHOLD * 2
//    0 < y < RCP_THRESHOLD
//    if x > 1, then x != y.
template <bool CHECK_RCP>
__device__
static
inline
uint32_t
quasiQuoRem(float& r, float x, float y)
{
  float q = truncf(__fmul_rz(x, fastReciprocal(y)));
  r = __fmaf_rz(q, -y, x); 
  if (CHECK_RCP && r < 0.0f)
    r += y, q -= 1.0f;
  return __float2uint_rz(q);
}
