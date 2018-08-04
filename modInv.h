/*  modInv.h

    Provide the modInv inline function, together with ancillary functions,
    in a separate header file so that quoRem<QUASI> can be certified 
    for use on specific devices by a standalone program and modInv can be 
    easily incorporated into other source code files.
  
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
//  when x >= FLOAT_THRESHOLD && y >= FLOAT_THRESHOLD/2.
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

//  Version of quoRem to use at modInv transition, i.e., when
//  x >= FLOAT_THRESHOLD && 1 < y < FLOAT_THRESHOLD.
//  Although algorithm can tolerate a quasi-quotient here (i.e., possibly one less than
//  the true quotient), the true quotient is about as fast as the quasi-quotient,
//  so we decide which version to use when the compiler compiles to a specific architecture.
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

template
<typename T>
__device__
static
inline
void
swap(T& __restrict__ x, T& __restrict__ y)
{
  T tmp = x;
  x = y;
  y = tmp;
}

//  Return 1/v (mod m), assuming gcd(m,v) == 1 && m > v > 0.
//  Based on the extended Euclidean algorithm:
//  see Knuth, The Art of Computer Programming, vol. 2, 3/e,
//  Algorithm X on pp342-3.
template <QuoRemType QRTYPE>
__device__
static
uint32_t
modInv(uint32_t v, uint32_t m)
{
  uint32_t u2 = 0, u3 = m;
  uint32_t v2 = 1, v3 = v;
  
  //  When u3 and v3 are both large enough, divide with floating point hardware.
  while  (v3 >= FLOAT_THRESHOLD)
    {
      u2 += v2 * quoRem(u3, u3, v3);
      if (u3 <  FLOAT_THRESHOLD)
        break;
      v2 += u2 * quoRem(v3, v3, u3);
    }
    
  bool swapUV = (v3 > u3);
  if  (swapUV)
    {
      swap(u2, v2);
      swap(u3, v3);
    }

  //  u3 >= FLOAT_THRESHOLD > v3.
  //  Transition to both u3 and v3 small, so v3 is cast and u3 is reduced into float variables.
  float u3f, v3f = __uint2float_rz(v3);
  u2 += v2 * quoRem<QRTYPE>(u3f, u3, v3);
   
  //  When u3 and v3 are both small enough, divide with floating point hardware.   
  //  At this point v3f > u3f.
  //  The loop will stop when u3f <= 1.0.
  //  If u3f == 1.0, |result| is in u2.
  //  If u3f == 0.0, then v3f == 1.0 and |result| is in v2.
  while (u3f > 1.0f)
    {
      v2 += u2 * quoRem<QRTYPE>(v3f, v3f, u3f);
      u2 += v2 * quoRem<QRTYPE>(u3f, u3f, v3f);
    }
      
  bool resultInU = (v3f != 1.0f); 
  if  (resultInU)             
    v2 = u2;
  if  (resultInU ^ swapUV)
    v2 = m - v2;
  return v2;
}

