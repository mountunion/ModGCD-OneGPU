/*  GmpCudaDevice-getGcdKernel.cu -- provides GmpCudaDevice::getGcdKernel method
                                     (includes the gcd kernel code).

  Implementation of the modular integer gcd algorithm using L <= 32 bit moduli.
  
  Reference: Weber, Trevisan, Martins 2005. A Modular Integer GCD algorithm
             Journal of Algorithms 54, 2 (February, 2005) 152-167.

             Note that there is an error in Fig. 2, which shows that the
             final result can be recovered as the mixed radix representation
             is calculated.  In actuality, all the mixed radix digits and moduli
             must be computed before the actual GCD can be recovered.
  
  Based on initial work by
  Authors:  Justin Brew, Anthony Rizzo, Kenneth Weber
            Mount Union College
            June 25, 2009

  Further revisions by 
  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  See GmpCudaDevice.cu for revision history.
*/

//  Enforce use of CUDA 9 or higher at compile time.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#else
#error Requires CUDA 9 or more recent
#endif

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include "GmpCuda.h"
//  Include the devicesRcpNoCheck definition, which is generated
//  by a configuration script.
#include "GmpCudaDevice-gcdDevicesRcpNoCheck.h"

//  Include the fastReciprocal and quasiQuoRem inline functions,
//  which are in a separate header file so that quasiQuoRem<false> can be certified
//  for use on specific devices.
#include "quasiQuoRem.h"

using namespace GmpCuda;

static constexpr int WARPS_PER_BLOCK   = GmpCudaDevice::GCD_BLOCK_SZ / WARP_SZ;  
static constexpr unsigned FULL_MASK    = 0xFFFFFFFF;           //  Used in sync functions.
static constexpr uint64_t MODULUS_MASK = uint64_t{0xFFFFFFFF}; //  Mask for modulus portion of pair.
static constexpr int32_t  MOD_INFINITY = INT32_MIN;            //  Larger than any modulur value

static constexpr int RCP_THRESHOLD_NORM_CLZ  = 32 - RCP_THRESHOLD_EXPT;  //  # leading zeros in a normalized denominator.
static constexpr uint32_t RCP_THRESHOLD = 1 << RCP_THRESHOLD_EXPT;

//  Make the cuda architecture number available as a constexpr for all compilation phases.
static constexpr int CUDA_ARCH =
#ifdef __CUDA_ARCH__
  __CUDA_ARCH__
#else
  -1
#endif
  ;
  
typedef GmpCudaDevice::pair_t pair_t;  //  Used to pass back result.

//  This type is used to conveniently manipulate the modulus and its inverse.
typedef struct {uint32_t modulus; uint64_t inverse;} modulus_t;


//  Which thread in the warp satisfying the predicate has a nonzero value?
//  Uses ballot so that every multiprocessor (deterministically) chooses the same pair.
//  In case there is no winner, use the 0 from warpLane 0.
__device__
static
inline
int
findAnyNonZero(pair_t pair, bool predicate = true)
{
  return max(0, __ffs(__ballot_sync(FULL_MASK, predicate && pair.value != 0)) - 1);
}

//  Posts to the barrier one of the pair parameters whose value is not 0.
//  If no such value is found, a pair with a 0 value is posted.
//  Preconditions:  all threads in block participate.
__device__
static
void
postAnyPairPriorityNonzero(pair_t pair, GmpCudaBarrier &bar)
{
   __shared__ pair_t sharedPair[WARP_SZ];
   
  __syncthreads();  // protect shared memory against last call to this function.

  if (findAnyNonZero(pair) == threadIdx.x % WARP_SZ)
    sharedPair[threadIdx.x / WARP_SZ] = pair;

  __syncthreads();
  
  pair = sharedPair[findAnyNonZero(sharedPair[threadIdx.x], threadIdx.x < WARPS_PER_BLOCK)];
  
  bar.post(*reinterpret_cast<uint64_t *>(&pair));
}

//  Chooses one of the pairs in the barrier that doesn't have a 0 value;
//  chosen pair is returned in pair as result.
//  If there are no nonzero values, a pair with value 0 is returned.
//  Preconditions:  all threads in block participate.
//  Postcondition: every thread will have the same pair.
__device__
static
void
collectAnyPairPriorityNonzero(pair_t& __restrict__ pair, 
                              GmpCudaBarrier& __restrict__ bar)
{
  __shared__ pair_t sharedPair[WARP_SZ];
  
  bar.collect(*reinterpret_cast<uint64_t*>(&pair)); // Only low gridDim.x threads have "good" values.
  
  __syncthreads();  // protect shared memory against last call to this function.
  
  int warpLane = threadIdx.x % WARP_SZ;
  
  if (findAnyNonZero(pair, threadIdx.x < gridDim.x) == warpLane && threadIdx.x < gridDim.x)
    sharedPair[threadIdx.x / WARP_SZ] = pair;

  __syncthreads();

  int numWarps = (gridDim.x - 1) / WARP_SZ + 1;

  //  All warps do this and get common value for winner.
  pair = sharedPair[findAnyNonZero(sharedPair[warpLane], warpLane < numWarps)];
}

//  Calculate min of x into lane 0 of warp.
__device__
inline
void
minWarp(uint64_t &x)
{
#pragma unroll
  for (int i = WARP_SZ/2; i > 0; i /= 2)
    x = min(x, __shfl_down_sync(FULL_MASK, x, i));
}

//  Calculates abs(x), except that MOD_INFINITY == INT32_MIN is not changed.
__device__
static
inline
uint64_t
modAbs(int32_t x)
{
  return (x < 0) ? ~x + 1 : x;
}

//  Posts pair which achieves the minimum of the absolute value 
//  of all pairs in each threadblock to bar.
//  Precondition: modulus of each pair is odd and all threads participate.
//  Postcondition: bar is ready for collectMinPair to be called.
__device__
static
void
postMinPair(pair_t pair, GmpCudaBarrier& bar)
{
  __shared__ uint64_t sharedX[WARP_SZ];

  __syncthreads();  // protect shared memory against last call to this function.
    
  //  Prepare a long int composed of the absolute value of pair.value in the high bits and pair.modulus in the low bits.
  //  Store sign of pair.value in the low bit of pair.modulus, which should always be 1 since it's odd.
  uint64_t x = (modAbs(pair.value) << 32) | (pair.modulus - (pair.value >= 0)); 

  //  Find the smallest in each warp, and store in sharedX.
  minWarp(x);
  if (threadIdx.x % WARP_SZ == 0)
    sharedX[threadIdx.x / WARP_SZ] = x;
  __syncthreads();

  //  Now find the min of the values in sharedX.
  //  WARPS_PER_BLOCK must be a power of 2 <= WARP_SZ.
  if (threadIdx.x < WARP_SZ)
    {
      x = sharedX[threadIdx.x];
#pragma unroll
      for (int i = WARPS_PER_BLOCK/2; i > 0; i /= 2)
        x = min(x, __shfl_down_sync(FULL_MASK, x, i));        
    }

  bar.post(x);
}

//  Returns, in pair, the pair which achieves the global minimum of the absolute value 
//  of the value over all the pairs that have been posted to bar.
//  Precondition: postMinPair was previously called and all threads participate.
__device__
static
void
collectMinPair(pair_t& __restrict__ pair, GmpCudaBarrier& __restrict__ bar)
{
  uint64_t x;
  bar.collect(x);
  
  __shared__ uint64_t sharedX[WARP_SZ];
  
  __syncthreads();  // protect shared memory against last call to this function.
      
  int numWarps =  (gridDim.x - 1) / WARP_SZ + 1;

  if (threadIdx.x / WARP_SZ < numWarps)
    {
      if (threadIdx.x >= gridDim.x)
        x = UINT64_MAX;
      minWarp(x);
      if (threadIdx.x % WARP_SZ == 0)
        sharedX[threadIdx.x / WARP_SZ] = x;
    }

  __syncthreads();
  if (threadIdx.x < WARP_SZ)
    {
      x = (threadIdx.x < numWarps) ? sharedX[threadIdx.x] : UINT64_MAX;
#pragma unroll
      for (int i = WARPS_PER_BLOCK/2; i > 1; i /= 2)  //  assert(gridDim.x <= blockDim.x);
        x = min(x, __shfl_down_sync(FULL_MASK, x, i));  
      sharedX[threadIdx.x] = min(x, __shfl_down_sync(FULL_MASK, x, 1));                            
   }

  __syncthreads();
  x = sharedX[0];
  
  pair.modulus = static_cast<uint32_t>(x & MODULUS_MASK); 
  pair.value   = static_cast<int32_t>(x >> 32);
  //  Restore original sign.
  if (pair.modulus & 1)
    pair.value = ~pair.value + 1;  // Should leave MOD_INFINITY unchanged.
  pair.modulus |= 1;
}

//  Determines whether the modulus is equal to x.
__device__
static
inline
bool
equals(uint32_t x, modulus_t m)
{
  return (m.modulus == x);
}

//  Return a - b (mod m) in the range 0..m-1.
//  Precondition: a, b are both in the range 0..m-1.
__device__
static
inline
uint32_t
modSub(uint32_t a, uint32_t b, modulus_t m)
{
  return a - b + (-(a < b) & m.modulus);
}

//  Calculate x mod m, where x is 64 bits long.
__device__
static
inline
uint32_t
mod(uint64_t x, modulus_t m)
{
  return x - static_cast<uint64_t>(m.modulus) * (__umul64hi(m.inverse, x) >> (L - 1));
}

//  Return a * b (mod m) in the range 0..m-1.
//  Precondition: a, b are both in the range 0..m-1, and m is prime.
__device__
static
inline
uint32_t
modMul(uint32_t a, uint32_t b, modulus_t m)
{
  return mod(static_cast<uint64_t>(a) * b, m);
}

__device__
static
inline
uint32_t
fromSigned(int32_t x, modulus_t m)
{
  return (x < 0) ? x + m.modulus : x;
}

// Give x mod m as a signed value in the range [-modulus/2, modulus,2]
__device__
static
inline
int32_t
toSigned(uint32_t x, modulus_t m)
{
  return (x >= m.modulus/2) ? x - m.modulus : x;
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


//  Computes an approximation for x / y, when x, y >= 2^21.
//  Approximation could be too small by 1 or 2.
//  The estimate of q from multiplying by the reciprocal here could be too high or too low by 1;
//  make it too low by 1 or 2, by subtracting 1.0 BEFORE truncating toward zero.
__device__
static
inline
uint32_t
quasiQuo(uint32_t x, uint32_t y)
{ 
  return __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_rz(y)), -1.0f));
}

//  Assumes x >= RCP_THRESHOLD > y. (Recall that RCP_THRESHOLD == 2^RCP_THRESHOLD_EXPT.)
//  First computes i such that 2^RCP_THRESHOLD_EXPT > y * 2^i >= 2^(RCP_THRESHOLD_EXPT - 1.
//  Returns q = 2^i * q' such that x - q' * y * 2^i < 2 * y * 2^i,
//  i.e., x - q * y < y * 2^(i + 1) < 2*RCP_THRESHOLD.
__device__
static
inline
uint32_t
quasiQuoNorm(uint32_t x, uint32_t y)
{
  int i = __clz(y) - RCP_THRESHOLD_NORM_CLZ;
  return quasiQuo(x, y << i) << i;
}

  
//  Faster divide possible when x and y are close in size.
//  Precondition: 2^32 > x, y >= 2^RCP_THRESHOLD_EXPT, so 0 <= x / y < 2^RCP_THRESHOLD_NORM_CLZ.
//  Could produce a quotient that's too small by 1--but modInv can tolerate that.
__device__
static
inline
uint32_t
quasiQuoRem(uint32_t& x, uint32_t y)
{ 
//  Computes an approximation q for x / y, when x, y >= RCP_THRESHOLD.
//  q could be too small by 1 or 2.
//  The estimate of q from multiplying by the reciprocal here could be too high or too low by 1;
//  make it too low by 1 or 2, by subtracting 1.0 BEFORE truncating toward zero.
//  uint32_t q = __float2uint_rz(__fmaf_rz(__uint2float_rz(x), fastReciprocal(__uint2float_rz(y)), -1.0f));
  uint32_t q = quasiQuo(x, y);
  x -= q * y; 
  if (x >= y)  //  Now x < 3 * y.
    x -= y, q += 1;
  return q;               //  Now x < 2 * y, but unlikely that x >= y.
}

//  Return 1/v (mod u), assuming gcd(u,v) == 1.
//  Assumes u > v > 0.
//  Uses the extended Euclidean algorithm:
//  see Knuth, The Art of Computer Programming, vol. 2, 3/e,
//  Algorithm X on pp342-3.
template <bool CHECK_RCP>
__device__
static
uint32_t
modInv(uint32_t u, uint32_t v)
{
  constexpr bool QUASI_TRANSITION = (CUDA_ARCH != 700);
  
  uint32_t u2 = 0, u3 = u;
  uint32_t v2 = 1, v3 = v;
  
  //  When u3 and v3 are both large enough, divide with floating point hardware.
  while  (v3 >= RCP_THRESHOLD)
    {
      u2 += v2 * quasiQuoRem(u3, v3);
      if (u3 <  RCP_THRESHOLD)
        break;
      v2 += u2 * quasiQuoRem(v3, u3);
    }
    
  bool negateResult = (v3 > u3);
  if  (negateResult)
    {
      swap(u2, v2);
      swap(u3, v3);
    }

  //  u3 >= RCP_THRESHOLD > v3.
  //  Transition to both u3 and v3 small, so values are cast into floats.
  //  Although algorithm can tolerate a quasi-quotient here (i.e., possibly one less than
  //  the true quotient), the true quotient is about as fast as the quasi-quotient,
  //  so we decide which version to use when the compiler compiles to a specific architecture.
  uint32_t q = (QUASI_TRANSITION) ? quasiQuoNorm(u3, v3) : u3 / v3;
  float u3f = __uint2float_rz(u3 - q * v3);
  float v3f = __uint2float_rz(v3);
  if (QUASI_TRANSITION)
    q += quasiQuoRem<CHECK_RCP>(u3f, v3f);
  u2 += v2 * q;
   
  //  When u3 and v3 are both small enough, divide with floating point hardware.   
  //  At this point v3f > u3f.
  //  The loop will stop when u3f <= 1.0.
  //  If u3f == 1.0, result is in u2.
  //  If u3f == 0.0, then v3f == 1.0 and result is in v2.
  //  If u3f ==-1.0, result is in u2.
  while (u3f > 1.0f)
    {
      v2 += u2 * quasiQuoRem<CHECK_RCP>(v3f, u3f);
      u2 += v2 * quasiQuoRem<CHECK_RCP>(u3f, v3f);
    }
  
  //  If we are transitioning with a quasi quotient and don't check the reciprocal, 
  //  u3f == -1.0f is possible, in which case the result will need to be negated.
  if (QUASI_TRANSITION && !CHECK_RCP)
    negateResult ^= (u3f == -1.0f);
    
  negateResult ^= (v3f != 1.0f);  //  Update negateResult based on where the answer ended up.
  
  if (v3f != 1.0f)                //  Answer in u2--copy into v2.
    v2 = u2;
  if (negateResult)
    v2 = u - v2;
  return v2;
}

// Calculate u/v mod m, in the range [0,m-1]
template <bool CHECK_RCP>
__device__
static
inline
uint32_t
modDiv(uint32_t u, uint32_t v, modulus_t m)
{
  return modMul(u, modInv<CHECK_RCP>(m.modulus, v), m);
}

//  Calculate x mod m for a multiword unsigned integer x.
__device__
static
uint32_t
modMP(uint32_t x[], size_t xSz, modulus_t m)
{
  __shared__ uint32_t sharedX[WARP_SZ];
  uint64_t result = uint64_t{0};
  
  __syncthreads();  // protect shared memory against last call to this function.
  
  while (xSz > warpSize)
    {
      xSz -= warpSize;
      //  Copy a block of x to shared memory for processing.
      if (threadIdx.x < warpSize)
        sharedX[threadIdx.x] = x[threadIdx.x + xSz];
      __syncthreads();
      //  Process the block in shared memory.
      for (size_t i = warpSize; i-- != 0;  )
        result = mod(result << 32 | sharedX[i], m);
      __syncthreads();
    }
  //  Now xSz <= warpSize.  Copy remainder of x to shared memory and process.
  if (threadIdx.x < xSz)
    sharedX[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  for (size_t i = xSz; i-- != 0;  )
    result = mod(result << 32 | sharedX[i], m);
  return static_cast<uint32_t>(result);
}
 
// Initialize modulus for this thread by reading a modulus m from the list
// and computing its "inverse", mInverse == 2^(W + L - 1) / m + 1.
__device__
static
inline
modulus_t
getModulus(uint32_t* moduliList)
{
    uint32_t m = moduliList[blockDim.x * blockIdx.x + threadIdx.x];
    uint64_t D = static_cast<uint64_t>(m);
    constexpr uint64_t FC_hi = uint64_t{1} << (W - 1);
    uint64_t q = FC_hi / D;
    uint64_t r = FC_hi % D;
    return {m, uint64_t{1} + (q << L) + (r << L) / D};
}

//  Device kernel for the GmpCudaDevice::getGcdKernel method.
template <bool CHECK_RCP>
__global__
static
void
kernel(uint32_t* __restrict__ buf, size_t uSz, size_t vSz, 
       uint32_t* __restrict__ moduliList, GmpCudaBarrier bar)
{
  int totalModuliRemaining = blockDim.x * gridDim.x;
  int ubits = (uSz + 1) * 32;  // somewhat of an overestimate
  int vbits = (vSz + 1) * 32;  // same here
  
  //MGCD1: [Find suitable moduli]
  modulus_t q = getModulus(moduliList);

  //MGCD2: [Convert to modular representation]

  uint32_t uq, vq;
  uq = modMP(buf,       uSz, q);
  vq = modMP(buf + uSz, vSz, q);

  //MGCD3: [reduction loop]

  bool active = true;  //  Is the modulus owned by this thread active, or has it been retired?

  pair_t pair, myPair;
  myPair.modulus = q.modulus;
  myPair.value = (vq == 0) ? MOD_INFINITY : toSigned(modDiv<CHECK_RCP>(uq, vq, q), q);
  postMinPair(myPair, bar);
  collectMinPair(pair, bar);
  
  do
    {
      uint32_t p, tq;
      int tbits;
      if (equals(pair.modulus, q))  //  Deactivate this modulus.
        active = false, myPair.value = MOD_INFINITY;
      if (active)
        {
          p = pair.modulus;
          if (p > q.modulus)        //  Bring within range.
            p -= q.modulus;
          tq = modDiv<CHECK_RCP>(modSub(uq, modMul(fromSigned(pair.value, q), vq, q), q), p, q);
          myPair.value = (tq == 0) ? MOD_INFINITY : toSigned(modDiv<CHECK_RCP>(vq, tq, q), q);
        }
      postMinPair(myPair, bar);
      if (active)
        uq = vq, vq = tq;       
      totalModuliRemaining -= 1;
      tbits = ubits - (L - 1) + __ffs(abs(pair.value));
      ubits = vbits, vbits = tbits;
      if (totalModuliRemaining * (L - 2) <= ubits)  //  Ran out of moduli--means initial estimate was wrong.
        {
          if (blockIdx.x && threadIdx.x)
            return;
          buf[0] = GmpCudaDevice::GCD_KERNEL_ERROR, buf[1] = GmpCudaDevice::GCD_REDUX_ERROR;
          return;
        }        
      collectMinPair(pair, bar);
    }
  while (pair.value != MOD_INFINITY);
   
  //MGCD4: [Find SIGNED mixed-radix representation] Each "digit" is either positive or negative.

  pair_t* pairs = (pair_t *)buf + 1;

  myPair.value = (active) ? toSigned(uq, q) : 0;  //  Inactive threads should have low priority.

  postAnyPairPriorityNonzero(myPair, bar);

  collectAnyPairPriorityNonzero(pair, bar);

  do
    {
      if (equals(pair.modulus, q))  //  deactivate modulus.
        active = false, myPair.value = 0;
      if (active)
        {
          uint32_t p = pair.modulus;
          if (pair.modulus > q.modulus)  //  Bring within range.
            p -= q.modulus;
          uq = modDiv<CHECK_RCP>(modSub(uq, fromSigned(pair.value, q), q), p, q);
          myPair.value = toSigned(uq, q);
        }
      postAnyPairPriorityNonzero(myPair, bar);
      *pairs++ = pair;
      totalModuliRemaining -= 1;
      if (totalModuliRemaining <= 0)  //  Something went wrong.
        break;
      collectAnyPairPriorityNonzero(pair, bar);
    }
  while (pair.value != 0);

  if (blockIdx.x | threadIdx.x)  //  Final cleanup by just one thread.
    return;

  //  Return a count of all the nonzero pairs, plus one more "pair" that includes buf[0] itself.
  //  If there aren't enough moduli to recover the result, return error codes.
  if (pair.value != 0) 
    buf[0] = GmpCudaDevice::GCD_KERNEL_ERROR, buf[1] = GmpCudaDevice::GCD_RECOVERY_ERROR;
  else
    buf[0] = pairs - reinterpret_cast<pair_t*>(buf);   
}

//  Used to find a device name in devicesRcpNoCheck.
static 
int 
comparator(const void* s1, const void* s2Ptr)
{
  return strcmp(static_cast<const char*>(s1), *static_cast<char * const *>(s2Ptr));
}

//  Return the appropriate gcd kernel for a device to use, based on
//  whether there needs to be a check performed after the rcp.approx reciprocal
//  in quasiQuoRem.
const 
void* 
GmpCudaDevice::getGcdKernel(char* devName)
{
  void* key = bsearch(static_cast<const void*>(devName), static_cast<const void*>(devicesRcpNoCheck), 
                      sizeof(devicesRcpNoCheck)/sizeof(char*), sizeof(char*), &comparator);
  return reinterpret_cast<const void *>((key == NULL) ? &kernel<true> : &kernel<false>);
}

