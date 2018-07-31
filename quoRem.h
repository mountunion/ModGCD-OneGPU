/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = quotient * yf + remainder.
    
    K. Weber

*/

typedef enum {QUASI, EXACT} QuoRemType;

// Defines range of valid input for quasiQuoRem.
static constexpr int      FLOAT_THRESHOLD_EXPT = 22;
static constexpr uint32_t FLOAT_THRESHOLD      = 1 << FLOAT_THRESHOLD_EXPT;
static constexpr int      FLOAT_THRESHOLD_NORM_CLZ  = 32 - FLOAT_THRESHOLD_EXPT;  //  # leading zeros in a normalized denominator.
static constexpr float    QUASI_QUO_ERR             = 0x1p11f/FLOAT_THRESHOLD;    // == 2^(11 - FLOAT_THRESHOLD_EXPT).

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
//  Computes a "quasi" quotient for x / y, when x, y >= FLOAT_THRESHOLD/2.
//  Approximation could be too small by 1.
//  The floating point calculation of x/y from multiplying 
//  by the reciprocal here could be too high by as much as QUASI_QUO_ERR
//  and too low by as much as 2^(-1) + 2^(-13) + QUASI_QUO_ERR (slight overestimate);
//  make it always too low, by subtracting QUASI_QUO_ERR.  
//  Then obtain  quasi-quotient by truncating toward zero.
//  The quasi-quotient could either be OK or too low by 1.
__device__
static
inline
uint32_t
quasiQuo(uint32_t x, uint32_t y)
{ 
  return __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_ru(y)), -QUASI_QUO_ERR));
}

//  Assumes x >= FLOAT_THRESHOLD > y. (Recall that FLOAT_THRESHOLD == 2^FLOAT_THRESHOLD_EXPT.)
//  First computes i such that 2^FLOAT_THRESHOLD_EXPT > y * 2^i >= 2^(FLOAT_THRESHOLD_EXPT - 1),
//  then returns q = 2^i * q' such that x - q' * y * 2^i < 2 * y * 2^i,
//  i.e., x - q * y < 2 * FLOAT_THRESHOLD.
__device__
static
inline
uint32_t
quasiQuoNorm(uint32_t x, uint32_t y)
{
  int i = __clz(y) - FLOAT_THRESHOLD_NORM_CLZ;
  return quasiQuo(x, y << i) << i;
}
 
//  Precondition: 2^32 > x, y >= 2^FLOAT_THRESHOLD_EXPT, so 0 <= x / y < 2^FLOAT_THRESHOLD_NORM_CLZ.
//  Computes quotient q and remainder r for x / y, when x, y >= FLOAT_THRESHOLD.
__device__
static
inline
uint32_t
quoRem(uint32_t& r, uint32_t x, uint32_t y)
{ 
  uint32_t q = quasiQuo(x, y);
  r = x - q * y; 
  if (r >= y)  //  q is too low by 1; correct.
    r -= y, q += 1;
  return q; 
}

template <QuoRemType quoRemType>
__device__
static
inline
uint32_t
quoRem(float& r, uint32_t x, uint32_t y)
{ 
//  Make the cuda architecture number available as a constexpr for all compilation phases.
  constexpr int CUDA_ARCH =
#ifdef __CUDA_ARCH__
  __CUDA_ARCH__
#else
  -1
#endif
  ;
  constexpr bool USE_QUASI_TRANSITION = (CUDA_ARCH < 700);
  
  uint32_t q = (USE_QUASI_TRANSITION) ? quasiQuoNorm(x, y) : x / y;
  r = __uint2float_rz(x - q * y);
  if (USE_QUASI_TRANSITION)
    q += quoRem<quoRemType>(r, r, __uint2float_rz(y));
  return q;  
}
