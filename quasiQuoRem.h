/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = quotient * yf + remainder.
    
    K. Weber

*/

typedef enum {QUASI, EXACT} QuoRemType;

// Defines range of valid input for quasiQuoRem.
static constexpr int      FLOAT_THRESHOLD_EXPT = 22;
static constexpr uint32_t FLOAT_THRESHOLD      = 1 << FLOAT_THRESHOLD_EXPT;

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
template <QuoRemType quoRemType>
__device__
static
inline
uint32_t
quoRem(float& r, float x, float y)
{
  constexpr float ERR = (quoRemType == QUASI) ? 0.0f : 0.25f ;
  float q = truncf(__fmaf_rz(x, fastReciprocal(y), -ERR));
  r = __fmaf_rz(q, -y, x); 
  if (quoRemType == EXACT && r >= y)
    r -= y, q += 1.0f;
  return __float2uint_rz(q);
}
