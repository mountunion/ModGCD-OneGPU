/*  GmpCudaDevice.cu -- provides API to the GPU kernel.

  Implementation of the modular integer gcd algorithm using L <= 32 bit moduli.

  This version is for a single device and uses shuffle mechanism for the min operation.
  January 11, 2018.

  Runs in CUDA 9.

  Put GmpCudaDevice::gcd in its own file == GmpCudaDevice-gcd.cu
  Added capability to use more than warpSize SMs.
  January 22, 2018
  
  Modified to allow large grid sizes up to maximum occupancy.
  February 7-17, 2018.

  K. Weber--January, 2010             basic 16 bit version
            additional modifications: July, 2010
            further simplifications:  February, 2011
                                      Includes using float operations for modulus.
            reorganized:              March 8, 2011
            eliminated parallel
            conversion to standard
            rep:                      June 22, 2011
            final cleanup:            July, 2011

            modified to allow up
	          to 32 bit moduli.         June, 2012

            made object-oriented:     August, 2012

            more cleanup, including
            limiting to arch >= 2.0
            (anyPair uses __ballot):  January, 2013

            Bug fixed in barrier.
						Uses fixed number of
					  threads, but arbitrary
            number of moduli.
            Also overlaps communi-
            cation with computation.	March, 2014

            Further cleanup           July, 2014

  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009
*/

#include <cassert>
#include "GmpCudaDevice.h"
using namespace GmpCuda;
#include "moduli.h"
#include <iostream>

#ifdef USE_COOP_GROUPS
#include <cooperative_groups.h>
#endif

namespace  //  used only within this compilation unit, and only for device code.
{

  constexpr unsigned int FULL_MASK = 0xFFFFFFFF;             //  Used in sync functions.

  // Adjust WARPS_PER_BLOCK to change the block size--don't change BLOCK_SZ directly.
  // WARPS_PER_BLOCK must evenly divide WARP_SZ.
  constexpr int WARPS_PER_BLOCK = WARP_SZ / 4;               //  Provides most flexibility. 
  constexpr int BLOCK_SZ        = WARP_SZ * WARPS_PER_BLOCK;

  //  This type is used to pass back the gcd from the kernel as a list of pairs.
  typedef struct __align__(8) {uint32_t modulus; int32_t value;} pair_t;

  __shared__ GmpCudaGcdStats stats;
  
  inline
  unsigned int
  roundUp(unsigned int x, unsigned int b)
  {
    return ((x + (b - 1)) / b) * b;
  }

  //  Posts to the barrier one of the pair parameters from ALL threads such that value is positive.
  //  If no such value is found, a pair with a 0 value is posted.
  //  Preconditions:  all threads in block participate.
  template <bool STATS>
  __device__
  void
  postAnyPair(pair_t pair, GmpCudaBarrier &bar)
  {
     __shared__ pair_t sharedPair[WARP_SZ];

    __syncthreads();  // protect shared memory against last call to this function.

    if (STATS && threadIdx.x == 0)
      stats.anyPositiveCycles -= clock();

    int winner = max(0, __ffs(__ballot_sync(FULL_MASK, pair.value)) - 1);
    //  in case there is no winner, use the 0 from warpLane 0.
    if (winner == threadIdx.x % warpSize)
      sharedPair[threadIdx.x / warpSize] = pair;

    __syncthreads();

    int numWarps = (blockDim.x - 1) / warpSize + 1;

    if (threadIdx.x < numWarps)
       winner = max(0, __ffs(__ballot_sync(FULL_MASK, sharedPair[threadIdx.x].value)) - 1);

    if (STATS && threadIdx.x == 0)
      stats.anyBarrierCycles -= clock();
      
    bar.post(*(uint64_t *)(sharedPair + winner));
  }

  //  Chooses one of the value parameters from ALL threads such that value is positive.
  //  Chosen value is returned as result.
  //  If no such value is found, returns 0.
  //  Preconditions:  all threads in block participate.
  template <bool STATS>
  static
  __device__
  pair_t
  collectAnyPair(GmpCudaBarrier &bar)
  {
    pair_t pair;    
    __shared__ pair_t sharedPair[WARP_SZ];
    
    bar.collect(*reinterpret_cast<uint64_t*>(&pair)); // Only low gridDim.x threads have "good" values.
    
    __syncthreads();  // protect shared memory against last call to this function.
    
    if (STATS && threadIdx.x == 0)
      stats.anyBarrierCycles += clock();

    int winner;
    int warpLane = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;
    int numWarps = (gridDim.x -1) / warpSize + 1;
    if (threadIdx.x < gridDim.x)
      {
        //  Using ballot so that every multiprocessor (deterministically) chooses the same pair(s).
        winner = max(0, __ffs(__ballot_sync(FULL_MASK, pair.value)) - 1);
        //  in case there is no winner, use the 0 from warpLane 0.
        if (winner == warpLane)
          sharedPair[warpIdx] = pair;
      }

    __syncthreads();

    //  All warps do this and get common value for winner.
    //  Would it be faster to have 1 warp do this and put in shared memory for all?
    winner = max(0, __ffs(__ballot_sync(FULL_MASK, warpLane < numWarps && sharedPair[warpLane].value)) - 1);

    if (STATS && threadIdx.x == 0)
      stats.anyPositiveCycles += clock(), stats.mixedRadixIterations += 1;
      
    return sharedPair[winner];
  }

  //  Calculate min of x into lane 0 of warp.
  __device__
  void
  minWarp(uint64_t &x)
  {
#pragma unroll
    for (int i = WARP_SZ/2; i > 0; i /= 2)
      x = min(x, __shfl_down_sync(FULL_MASK, x, i));
  }

  //  Posts pair which achieves the minimum of the absolute value 
  //  of the value in the pairs on each SM to bar.
  //  Precondition: modulus of each pair is odd and all threads participate.
  //  Postcondition: bar is ready for collectMinPair to be called.
  template <bool STATS>
  __device__
  void
  postMinPair(pair_t pair, GmpCudaBarrier &bar)
  {
    __shared__ uint64_t sharedX[WARP_SZ];

    __syncthreads();  // protect shared memory against last call to this function.

    if (STATS && threadIdx.x == 0)
      stats.minPositiveCycles -= clock();
      
    //  Prepare a long int composed of the absolute value of pair.value in the high bits and pair.modulus in the low bits.
    //  Subtract 1 from pair.value to make 0 become "largest"; must restore at end.
    //  Store sign of pair.value in the low bit of pair.modulus, which should always be 1 since it's odd.
    uint64_t x = uint64_t{-1};
    
    if (pair.value != 0)
      {
        x = static_cast<uint64_t>(abs(pair.value) - 1);
        x <<= 32;
        x |= pair.modulus - (pair.value >= 0);
      }
 
    //  Find the smallest in each warp, and store in sharedX.
    minWarp(x);
    if (threadIdx.x % WARP_SZ == 0)
      sharedX[threadIdx.x / WARP_SZ] = x;
    __syncthreads();

    //  Now find the min of the values in sharedX.
    //  WARPS_PER_BLOCK must be a power of 2 <= WARP_SZ.
    if (threadIdx.x < WARPS_PER_BLOCK)
      {
        x = sharedX[threadIdx.x];
#pragma unroll
        for (int i = WARPS_PER_BLOCK/2; i > 0; i /= 2)
          x = min(x, __shfl_down_sync(FULL_MASK, x, i));        
      }

    if (STATS && threadIdx.x == 0)
      stats.minBarrierCycles -= clock();

    bar.post(x);
  }

  //  Returns the pair which achieves the global minimum of the absolute value 
  //  of the value in all the pairs that have been posted to bar.
  //  Precondition: postMinPair was previously called and all threads participate.
  template <bool STATS>
  __device__
  pair_t
  collectMinPair(GmpCudaBarrier& bar)
  {
    uint64_t x;
    bar.collect(x);
    
    if (STATS && threadIdx.x == 0)
      stats.minBarrierCycles += clock();

    __shared__ uint64_t sharedX[WARP_SZ];
    
    __syncthreads();  // protect shared memory against last call to this function.
        
    int numWarps =  (gridDim.x - 1) / WARP_SZ + 1;

    if (threadIdx.x / WARP_SZ < numWarps)
      {
        if (threadIdx.x >= gridDim.x)
          x = static_cast<uint64_t>(-1);
        minWarp(x);
        if (threadIdx.x % WARP_SZ == 0)
          sharedX[threadIdx.x / WARP_SZ] = x;
      }

    switch  (numWarps)
      {
        default:  // For the unlikely cases where 256 < gridDim.x.
          __syncthreads();
          if (threadIdx.x < WARP_SZ)
            {
              x = (threadIdx.x < numWarps) ? sharedX[threadIdx.x] : static_cast<uint64_t>(-1);
#pragma unroll
              for (int i = WARPS_PER_BLOCK/2; i > 1; i /= 2)  //  assert(gridDim.x <= blockDim.x);
                x = min(x, __shfl_down_sync(FULL_MASK, x, i));  
              sharedX[threadIdx.x] = min(x, __shfl_down_sync(FULL_MASK, x, 1));                            
           }
          break;
        //  Special cases will handle gridDim.x <= 256.
        case 5: case 6: case 7: case 8:
          __syncthreads();
          if (threadIdx.x < 8)
            {
              x = (threadIdx.x < numWarps) ? sharedX[threadIdx.x] : static_cast<uint64_t>(-1);
              x = min(x, __shfl_down_sync(0xFF, x, 4));        
              x = min(x, __shfl_down_sync(0xFF, x, 2));  
              sharedX[threadIdx.x] = min(x, __shfl_down_sync(0xFF, x, 1));      
            }
          break;
        case 4:
          __syncthreads();
          if (threadIdx.x < 2)
            {
              x = min(sharedX[threadIdx.x], sharedX[threadIdx.x + 2]);  
              sharedX[threadIdx.x] = min(x, __shfl_down_sync(0x3, x, 1));
            }
          break;
        case 3:
          __syncthreads();
          if (threadIdx.x == 0)
            sharedX[0] = min(min(x, sharedX[1]), sharedX[2]);
          break;
        case 2:
          __syncthreads();
          if (threadIdx.x == 0)
            sharedX[0] = min(x, sharedX[1]);
          break;
        case 1:
          break;
      }
      
    __syncthreads();
    x = sharedX[0];
    
    pair_t pair;
    pair.modulus = static_cast<uint32_t>(x & 0xFFFFFFFFL); 
    pair.value   = static_cast<int32_t>(x >> 32);
    //  Restore original value and sign.
    pair.value += 1;
    if (pair.modulus & 1)
      pair.value = -pair.value;
    pair.modulus |= 1;

    if (STATS && threadIdx.x == 0)
      stats.minPositiveCycles += clock(), stats.reductionIterations += 1;

    return pair;
  }

  __device__
  bool
  equals(uint32_t x, modulus_t &m)
  {
    return (m.modulus == x);
  }

  //  Return a - b (mod m) in the range 0..m-1.
  //  Precondition: a, b are both in the range 0..m-1.
  __device__
  uint32_t
  modSub(uint32_t a, uint32_t b, modulus_t m)
  {
    return a - b + (-(a < b) & m.modulus);
  }

  __device__
  uint32_t
  mod(uint64_t x, modulus_t m)
  {
    return x - static_cast<uint64_t>(m.modulus) * (__umul64hi(m.inverse, x) >> (L - 1));
  }

  //  Return a * b (mod m) in the range 0..m-1.
  //  Precondition: a, b are both in the range 0..m-1, and m is prime.
  __device__
  uint32_t
  modMul(uint32_t a, uint32_t b, modulus_t m)
  {
    return mod(static_cast<uint64_t>(a) * b, m);
  }

  __device__
  uint32_t
  fromSigned(int32_t x, modulus_t m)
  {
    return (x < 0) ? x + m.modulus : x;
  }

  __device__
  int32_t
  toSigned(uint32_t x, modulus_t m)
  {
    return (x >= m.modulus/2) ? x - m.modulus : x;
  }

  //  Return 1/v (mod m).  Requires 0 < v < m, and gcd(m,v) == 1.
  //  Loop should finish if m == 0 but result will be junk.
  //  Warning!! Loop may not finish if v == 0.
  __device__
  uint32_t
  modInv(uint32_t v, modulus_t m)
  {
    if (L < 23)
      {
        float x1, x2, y1, y2, q;

        //  xi * v == yi (mod m)
        //  One of the yi will be 1 at the end, since gcd(m,v) = 1.  The other will be 0.

        x1 = 0.0, y1 = m.modulus;
        x2 = 1.0, y2 = v;

        do
          {
            q = truncf(__fdividef(y1, y2));
            x1 -= x2*q;
            y1 -= y2*q;
            if (y1 == 0)
              return static_cast<uint32_t>(x2);              //  Answer in x2 is positive.
            q = truncf(__fdividef(y2, y1));
            x2 -= x1*q;
            y2 -= y1*q;
          }
        while (y2);
        return m.modulus + static_cast<int32_t>(x1);         //  Answer in x1 is negative.
      }
    else   //  Values are too large to fit in float.
      {
        int32_t  x1, x2;
        uint32_t y1, y2;

        x1 = 0, y1 = m.modulus;
        x2 = 1, y2 = v;

        const bool INT_DIVIDE_IS_FAST = false;
        if (INT_DIVIDE_IS_FAST)
          {
            do
              {
                uint32_t q;
                q = y1 / y2;
                x1 -= x2*q;
                y1 -= y2*q;
                if (y1 == 0)
                  return x2;         //  Answer in x2 is positive.
                q = y2 / y1;
                x2 -= x1*q;
                y2 -= y1*q;
              }
            while (y2);
            return m.modulus + x1;  //  Answer in x1 is negative.
          }
        else  //  Non-divergent code to reduce y1 and y2 to fit into floats.
          {
            y1 <<= (32 - L), y2 <<= (32 - L);
            int j = __clz(y2);                          //  First eliminate y1's MSB
            uint32_t y = y2 << j;
            int32_t  x = x2 << j;
            if (y <= y1)
              y1 = y1 - y, x1 = x1 - x;
            else
              y1 = y - y1, x1 = x - x1;
            j = __clz(y1);
            y = y1 << j, x = x1 << j;
            if ((int32_t)y2 < 0)                         //  y2's MSB is still set, so eliminate it.
              {
                if (y <= y2)
                  y2 = y2 - y, x2 = x2 - x;
                else
                  y2 = y - y2, x2 = x - x2;
              }
#pragma unroll
            for (int i = 32 - L + 1; i < 10; i += 1)     //  Eliminate more bits from y1 and y2, by shifts and subtracts, to fit them into floats.
              {
                y1 <<= 1, y2 <<= 1;

                j = __clz(y2);                           //  First eliminate y1's MSB
                y = y2 << j, x = x2 << j;
                if ((int32_t)y1 < 0 && y2)
                  {
                    if (y <= y1)
                      y1 = y1 - y, x1 = x1 - x;
                    else
                      y1 = y - y1, x1 = x - x1;
                  }

                j = __clz(y1);
                y = y1 << j, x = x1 << j;
                if ((int32_t)y2 < 0 && y1)
                  {
                    if (y <= y2)
                      y2 = y2 - y, x2 = x2 - x;
                    else
                      y2 = y - y2, x2 = x - x2;
                  }
              }

            if (y1 == 0)
              return fromSigned(x2, m);

            float f1, f2, q;

            f1 = y1 >> 9, f2 = y2 >> 9;

            if (f2 > f1)
              {
                q = truncf(__fdividef(f2, f1));
                x2 -= x1*(int32_t)q;
                f2 -= f1*q;
              }

            while (f2)
              {
                q = truncf(__fdividef(f1, f2));
                x1 -= x2*(int32_t)q;
                f1 -= f2*q;
                if (f1 == 0)
                  return fromSigned(x2, m);
                q = truncf(__fdividef(f2, f1));
                x2 -= x1*(int32_t)q;
                f2 -= f1*q;
              }
            return fromSigned(x1, m);
          }
      }
  }

  __device__
  uint32_t
  modDiv(uint32_t u, uint32_t v, modulus_t m)
  {
    return modMul(u, modInv(v, m), m);
  }

  __device__
  uint32_t
  modMP(uint32_t x[], size_t xSz, modulus_t m)
  {
    __shared__ uint32_t sharedX[WARP_SZ];
    unsigned long long int result = 0LL;
    
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

  //  Entry point into device-only code.
  template <bool STATS>
  __global__
  void
  kernel(uint32_t* buf, size_t uSz, size_t vSz, GmpCudaBarrier bar, struct GmpCudaGcdStats* gStats = NULL)
  {
    int totalModuliRemaining = blockDim.x * gridDim.x;
    int ubits = (uSz + 1) * 32;  // somewhat of an overestimate
    int vbits = (vSz + 1) * 32;  // same here
    
    //  The arithmetic used on the clock requires the exact same type size all the time.
    //  It uses the fact that the arithmetic is modulo 2^32.
    if (STATS && threadIdx.x == 0)
      {
        memset(&stats, 0, sizeof(stats));
        stats.blockDim = blockDim.x;
        stats.totalCycles = -clock();
      }

    //MGCD1: [Find suitable moduli]
    modulus_t q = moduliList[blockDim.x * blockIdx.x + threadIdx.x];

    //MGCD2: [Convert to modular representation]

    if (STATS && threadIdx.x == 0)
      stats.convertToModularCycles -= clock();

    uint32_t uq, vq;
    uq = modMP(buf,       uSz, q);
    vq = modMP(buf + uSz, vSz, q);

    if (STATS && threadIdx.x == 0)
      stats.convertToModularCycles += clock(), stats.reductionCycles -= clock();

    //MGCD3: [reduction loop]

    bool   active = true;  //  Is the modulus owned by this thread active, or has it been retired?

    pair_t pair, myPair;
    myPair.value = (vq) ? toSigned(modDiv(uq, vq, q), q) : 0;
    myPair.modulus = q.modulus;
    postMinPair<STATS>(myPair, bar);
    pair = collectMinPair<STATS>(bar);
    
    do
      {
        uint32_t p, tq;
        int tbits;
        if (equals(pair.modulus, q))  //  Deactivate this modulus.
          active = false, myPair.value = 0;
        if (active)
          {
            p = pair.modulus;
            if (p > q.modulus)        //  Bring within range.
              p -= q.modulus;
            tq = modDiv(modSub(uq, modMul(fromSigned(pair.value, q), vq, q), q), p, q);
            myPair.value = (tq) ? toSigned(modDiv(vq, tq, q), q) : 0;
          }
        postMinPair<STATS>(myPair, bar);
        if (active)
          uq = vq, vq = tq;       
        totalModuliRemaining -= 1;
        tbits = ubits - (L - 1) + __ffs(abs(pair.value));  //  Conservative estimate--THIS NEEDS REVIEWED
        ubits = vbits, vbits = tbits;
        pair = collectMinPair<STATS>(bar);
      }
    while (pair.value != 0 && totalModuliRemaining > ubits / (L - 1));
    
    if (STATS && threadIdx.x == 0)
      stats.reductionCycles += clock();
      
    if (pair.value != 0)  //  Ran out of moduli--means initial estimate was wrong.
      {
        if (blockIdx.x == 0 && threadIdx.x == 0)
          {
            buf[0] = 0, buf[1] = 0;
            if (STATS)
              {
                stats.totalCycles += clock();
                *gStats = stats;
              }
          }
        return;
      } 
      
    //MGCD4: [Find SIGNED mixed-radix representation] Each "digit" is either positive or negative.

    if (STATS && threadIdx.x == 0)
      stats.mixedRadixCycles -= clock();

    pair_t* pairs = (pair_t *)buf + 1;

    //pair.modulus = q.modulus;
    myPair.value = (active) ? toSigned(uq, q) : 0;

    postAnyPair<STATS>(myPair, bar);
    pair = collectAnyPair<STATS>(bar);

    do
      {
        if (equals(pair.modulus, q))  //  deactivate modulus.
          active = false, myPair.value = 0;
        if (active)
          {
            uint32_t p = pair.modulus;
            if (pair.modulus > q.modulus)  //  Bring within range.
              p -= q.modulus;
            uq = modDiv(modSub(uq, fromSigned(pair.value, q), q), p, q);
            myPair.value = toSigned(uq, q);
          }
        postAnyPair<STATS>(myPair, bar);
        *pairs++ = pair;
        totalModuliRemaining -= 1;
        ubits -= L - 1;
        pair = collectAnyPair<STATS>(bar);
      }
    while (pair.value != 0 && totalModuliRemaining > ubits / (L - 1));

    if (blockIdx.x | threadIdx.x)  //  Final cleanup by just one thread.
      return;

    //  Return a count of all the nonzero pairs, plus one more "pair" that includes buf[0] itself.
    //  If there aren't enough moduli to recover the result, return error codes.
    if (pair.value != 0) 
      buf[0] = 0, buf[1] = 1;
    else
      *buf = pairs - reinterpret_cast<pair_t*>(buf);
    
    if (STATS)
      {
        stats.mixedRadixCycles += clock();
        stats.totalCycles      += clock();
        *gStats = stats;
      }
  }
}

//  All that follows is host only code.

void
__host__
GmpCudaDevice::initGcdOccupancy()
{
  assert(BLOCK_SZ <= props.maxThreadsPerBlock);
  assert(cudaSuccess == cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gcdOccupancy, (collectStats) ? kernel<true> : kernel<false>, BLOCK_SZ, 0));
}

void
__host__
GmpCudaDevice::gcd(mpz_t g, mpz_t u, mpz_t v) throw (std::runtime_error)
{
  if(mpz_cmp(u, v) < 0)
    mpz_swap(u, v);

  size_t ubits = mpz_sizeinbase(u, 2);
  size_t vbits = mpz_sizeinbase(v, 2);

  //  Slightly overestimate size of parameters and size of result, which is a list of moduli pairs, to get size of buf.
  uint32_t buf[2*(std::max((ubits + vbits)/64, vbits/(L-1)) + 2)];

  //  Stage parameters into buf and zero fill rest of buf.
  size_t uSz, vSz;
  mpz_export(buf,       &uSz, -1, sizeof(uint32_t), 0, 0, u);
  mpz_export(buf + uSz, &vSz, -1, sizeof(uint32_t), 0, 0, v);
  memset(buf + uSz + vSz, 0, sizeof(buf) - (uSz + vSz) * sizeof(uint32_t));
  
  //  Number of moduli needed is approximated by a function of the number of bits in the larger input.
  const float MODULI_MULTIPLIER = 1.6 - 0.014 * L;  //  Heuristically obtained formula.
  float numModuliNeeded = ceil(MODULI_MULTIPLIER * ubits / logf(ubits));
  
  int blocksPerSM = static_cast<int>(ceil(numModuliNeeded / (props.multiProcessorCount * BLOCK_SZ)));
  
  gridSize = min(min(BLOCK_SZ, maxGridSize), min(blocksPerSM , gcdOccupancy) * props.multiProcessorCount);
     
  int numThreads = BLOCK_SZ * gridSize;

  if (numThreads < numModuliNeeded)
    throw std::runtime_error("Cannot allocate enough threads to support computation.");

  if (numThreads > NUM_MODULI)
    throw std::runtime_error("Not enough moduli available for given input.");

  //  Allocate some extra space in the global buffer, so that modMP can assume it can safely read a multiple of
  //  warpSize words to get the entirety (+ more) of either parameter.
  uint32_t* globalBuf;

  assert(cudaSuccess == cudaMalloc(&globalBuf, std::max(sizeof(buf), sizeof(uint32_t) * (uSz + roundUp(vSz, props.warpSize)))));

  //  Copy parameters to global memory.
  assert(cudaSuccess == cudaMemcpy(globalBuf, buf, sizeof(buf), cudaMemcpyHostToDevice));

  //  Execute a specific kernel, based on whether we are collecting statistics.

  barrier->reset();  //  Reset to use again.

#ifdef USE_COOP_GROUPS
  void* args[] = {&globalBuf, &uSz, &vSz, &*barrier, &stats};
  void* kPtr   = reinterpret_cast<void *>((collectStats) ? kernel<true> : kernel<false>);
  assert(cudaSuccess == cudaLaunchCooperativeKernel(kPtr, gridSize, BLOCK_SZ, args));
#else    
  if (collectStats)
    kernel<true> <<<gridSize, BLOCK_SZ>>>(globalBuf, uSz, vSz, *barrier, stats);
  else
    kernel<false><<<gridSize, BLOCK_SZ>>>(globalBuf, uSz, vSz, *barrier);
#endif

  assert(cudaSuccess == cudaDeviceSynchronize());

  // Copy result from global memory and convert from mixed-radix to standard representation.
  assert(cudaSuccess == cudaMemcpy(buf, globalBuf, 2*sizeof(pair_t), cudaMemcpyDeviceToHost));  // Just size and 0th mixed-radix digit read now.
  
  if (buf[0] == 0)
    {
      assert(cudaSuccess == cudaFree(globalBuf));
      switch(buf[1])
        {
          case 0:  throw std::runtime_error("Ran out of moduli in the reduction loop.");
          case 1:  throw std::runtime_error("Ran out of moduli in the recovery loop.");
          default: throw std::runtime_error("Unknown error in the gcd kernel.");
        }
    }

  if (buf[0] > 1)
    assert(cudaSuccess == cudaMemcpy(reinterpret_cast<pair_t*>(buf), globalBuf, buf[0] * sizeof(pair_t), cudaMemcpyDeviceToHost));

  pair_t* pairs = reinterpret_cast<pair_t*>(buf) + buf[0] - 1;  // point to most significant digit.

  mpz_set_si(g, pairs->value);

  while (--pairs != reinterpret_cast<pair_t*>(buf))
    {
      mpz_mul_ui(g, g, pairs->modulus);
      if (pairs->value < 0)
        mpz_sub_ui(g, g, -pairs->value);
      else
        mpz_add_ui(g, g,  pairs->value);
    }

  mpz_abs(g, g);

  assert(cudaSuccess == cudaFree(globalBuf));
}
