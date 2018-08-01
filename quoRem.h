/*  quasiQuoRem.h.

    Provides inline functions that compute a "quasi" quotient and remainder for 
    the long division xf = quotient * yf + remainder.
    
    K. Weber
    
    1-Aug-2018.

*/

typedef enum {QUASI, FAST_EXACT, SAFE_EXACT} QuoRemType;

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

//  quasiQuoRem computes a quasi-quotient q and quasi-remainder r.
//  Preconditions: 
//    x and y are integers
//    0 < x < RCP_THRESHOLD * 2
//    0 < y < RCP_THRESHOLD
//    y != x when x > 1.
//  Postconditions:
//    0.0f <= r < y     when 0     <  x < y * 2
//    0.0f <= r < y * 2 when y * 2 <= x < FLOAT_THRESHOLD * 2
//    x == q * y + r
//  NOTE: error analysis depends on fastReciprocal(1.0f) == 1.0f and fastReciprocal(2.0f) == 0.5f exactly.
template <QuoRemType QRTYPE>
__device__
static
inline
uint32_t
quoRem(float& r, float x, float y)
{
  constexpr float ERR = (QRTYPE == FAST_EXACT) ? -(FLOAT_THRESHOLD/0x1p24f) : 0.0f;
  float q = truncf(__fmaf_rz(x, fastReciprocal(y), ERR));
  r = __fmaf_rz(q, -y, x); 
  if (QRTYPE == SAFE_EXACT && r < 0.0f)
    r += y, q -= 1.0f;
  if (QRTYPE != QUASI && r >= y)
    r -= y, q += 1.0f;
  return __float2uint_rz(q);
}

//  Computes a "quasi" quotient for x / y, 
//  when x > FLOAT_THRESHOLD && y >= FLOAT_THRESHOLD/2.
//  Approximation could be too small by 1.
//  The floating point calculation of x/y from multiplying 
//  by the reciprocal here could be too high by as much as 
//  ERR == 2^(11 - FLOAT_THRESHOLD_EXPT),
//  and too low by as much as 2^(-1) + 2^(-13) + ERR (slight overestimate);
//  make it always too low, by subtracting ERR.  
//  Then obtain  quasi-quotient by truncating toward zero.
//  The quasi-quotient could either be OK or too low by 1.
__device__
static
inline
uint32_t
quasiQuo(uint32_t x, uint32_t y)
{ 
  constexpr float ERR = -(0x1p11f/FLOAT_THRESHOLD);
  return __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_ru(y)), ERR));
}
 
//  Computes quotient q and remainder r for x / y, 
//  when x > FLOAT_THRESHOLD && y >= FLOAT_THRESHOLD/2.
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

template <QuoRemType QRTYPE>
__device__
static
inline
uint32_t
quoRem(float& r, uint32_t x, uint32_t y)
{ 
  uint32_t q;
#ifdef __CUDA_ARCH__
  if (__CUDA_ARCH__ == 700) // int division faster.
    {
      q = x / y;
      r = __uint2float_rz(x - q * y);
    }
  else                      // float reciprocal faster.
#endif
    {
      int i = __clz(y) - (32 - FLOAT_THRESHOLD_EXPT);
      q = quasiQuo(x, y << i) << i;
      r = __uint2float_rz(x - q * y);
      q += quoRem<QRTYPE>(r, r, __uint2float_rz(y));
    }
  return q;
}
