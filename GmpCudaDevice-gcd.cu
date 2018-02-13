/*  GmpCudaDevice.cu -- provides API to the GPU kernel.

  Implementation of the modular integer gcd algorithm using L <= 32 bit moduli.

  This version is for a single device and uses shuffle mechanism for the min operation.
  January 11, 2018.

  Updated to run in CUDA 9.

  Backported to CUDA 8 to run on Owens cluster in Ohio Supercomputer Center.
  January 15, 2018.

  Put GmpCudaDevice::gcd in its own file == GmpCudaDevice-gcd.cu
  Added capability to use more than warpSize SMs.
  January 22, 2018

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

namespace  //  used only within this compilation unit, and only for device code.
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000

  constexpr unsigned int FULL_MASK = 0xffffffff;
  constexpr int BLOCK_SZ = WARP_SZ * WARP_SZ / 2;

#else

//  This is needed to support code older than CUDA 9.

#define __syncwarp(X)
#define __ballot_sync(X, Y) __ballot(Y)
#define __shfl_xor_sync(X,Y,Z) __shfl_xor_uint64_t(Y, Z)

__device__ inline uint64_t __shfl_xor_uint64_t(uint64_t x, int laneMask)
{
   int2 tmp = *reinterpret_cast<int2*>(&x);
   tmp.x = __shfl_xor(tmp.x, laneMask);
   tmp.y = __shfl_xor(tmp.y, laneMask);
   return *reinterpret_cast<int64_t*>(&tmp);
}

#endif

  constexpr size_t MAX_MODULI_PER_THREAD = 1;

  //  This type is used to pass back the gcd from the kernel as a list of pairs.
  typedef struct __align__(8) {uint32_t modulus; int32_t value;} pair_t;

  //  Various ways to access dynamically allocated shared memory.
  //  The code is expecting at least warpSize * sizeof(uint64_t) bytes
  //  of shared memory to be allocated dynamically.
  extern __shared__ uint64_t sharedUint64[];
  extern __shared__ pair_t   sharedPair[];
  extern __shared__ uint32_t sharedUint32[];

  //  This function is used to calculate a pointer to the "end"
  //  of sharedUint64[], where space for the statistics is
  //  allocated when statistics are called for.
  __device__
  inline
  GmpCudaGcdStats *
  getSharedStats()
  {
     return (struct GmpCudaGcdStats *)(sharedUint64 + warpSize);
  }

      __device__
  inline
  GmpCudaGcdStats *
  initSharedStats(int numModuli)
  {
    GmpCudaGcdStats * sPtr = getSharedStats();
    sPtr->modPerThread = numModuli;
    sPtr->blockDim = blockDim.x;
    sPtr->convertToModularCycles = sPtr->reductionCycles = sPtr->minPositiveCycles = sPtr->minBarrierCycles
                                 = sPtr->mixedRadixCycles = sPtr->anyBarrierCycles = sPtr->anyPositiveCycles
                                 = sPtr->reductionIterations = sPtr->mixedRadixIterations = 0;
    sPtr->totalCycles = -clock();
    return sPtr;
  }

  inline
  unsigned int
  roundUp(unsigned int x, unsigned int b)
  {
    return ((x + (b - 1)) / b) * b;
  }

  //  Chooses one of the value parameters from ALL threads such that value is positive.
  //  Chosen value is returned as result.
  //  If no such value is found, returns 0.
  //  Preconditions:  all threads in block participate.
  template <bool STATS>
  __device__
  void
  postAnyPair(pair_t pair, GmpCudaBarrier &bar)
  {
    if (STATS && threadIdx.x == 0)
      getSharedStats()->anyPositiveCycles -= clock();

    int winner = max(0, __ffs(__ballot_sync(FULL_MASK, pair.value)) - 1);
    //  in case there is no winner, use the 0 from warpLane 0.
    if (winner == threadIdx.x % warpSize)
      sharedPair[threadIdx.x / warpSize] = pair;

    __syncthreads();

    int numWarps = (blockDim.x - 1) / warpSize + 1;

    if (threadIdx.x < numWarps)
       winner = max(0, __ffs(__ballot_sync(FULL_MASK, sharedPair[threadIdx.x].value)) - 1);

    if (STATS && threadIdx.x == 0)
      getSharedStats()->anyBarrierCycles -= clock();

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
    bar.collect(*reinterpret_cast<uint64_t*>(&pair)); // Only low gridDim.x threads have "good" values.

    if (STATS && threadIdx.x == 0)
      getSharedStats()->anyBarrierCycles += clock();

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
      getSharedStats()->anyPositiveCycles += clock(), getSharedStats()->mixedRadixIterations += 1;

    return sharedPair[winner];
  }

  //  calculate min into all lanes of a warp.
  __device__
  void
  minWarp(uint64_t &x)
  {
//      switch (warpSize)
//        {
//          default: assert(false);
//          case 32:
            x = min(x, __shfl_xor_sync(FULL_MASK, x, 16));
            x = min(x, __shfl_xor_sync(FULL_MASK, x,  8));
            x = min(x, __shfl_xor_sync(FULL_MASK, x,  4));
            x = min(x, __shfl_xor_sync(FULL_MASK, x,  2));
            x = min(x, __shfl_xor_sync(FULL_MASK, x,  1));
//        }
  }

  template <bool STATS>
  __device__
  void
  postMinPair(pair_t pair, GmpCudaBarrier &bar)
  {
    if (STATS && threadIdx.x == 0)
      getSharedStats()->minPositiveCycles -= clock();

    //  Prepare a long int composed of the absolute value of pair.value in the high bits and pair.modulus in the low bits.
    //  Subtract 1 from pair.value to make 0 become "largest"; must restore at end.
    //  Store sign of pair.value in the low bit of pair.modulus, which should always be 1 since it's odd.
    uint64_t x;
    x = static_cast<uint32_t>(abs(pair.value)) - 1;
    x <<= 32;
    x |= pair.modulus - (pair.value >= 0);

    //  Find the smallest in each warp, and store in sharedUint64.
    minWarp(x);
    if (threadIdx.x % warpSize == 0)
      sharedUint64[threadIdx.x / warpSize] = x;
    __syncthreads();

    //  Now use the low warp to find the min of the values in sharedUint64.
    int numWarps = (blockDim.x - 1) / warpSize + 1;
    if (threadIdx.x < warpSize)
      {
        x = (threadIdx.x < numWarps) ? sharedUint64[threadIdx.x] : static_cast<uint64_t>(-1);
        minWarp(x);
      }

    if (STATS && threadIdx.x == 0)
      getSharedStats()->minBarrierCycles -= clock();

    bar.post(x);
  }

  //  Returns the x where the global minimum of x is achieved, for ALL threads.
  //  Precondition: x != 0 && all threads participate && blockDim.x >= 2 * warpSize.
  template <bool STATS>
  __device__
  pair_t
  collectMinPair(GmpCudaBarrier& bar)
  {
    uint64_t x;
    bar.collect(x);

    if (STATS && threadIdx.x == 0)
      getSharedStats()->minBarrierCycles += clock();

    int warpLane = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;
    int numWarps =  (gridDim.x - 1) / warpSize + 1;
    if (warpIdx < numWarps)
      {
        if (threadIdx.x >= gridDim.x)
          x = static_cast<uint64_t>(-1);
        minWarp(x);
        if (warpLane == 0)
          sharedUint64[warpIdx] = x;
      }

    __syncthreads();

    //  There should only be a few values to find the min of now.
    //  If there is more than one, put the min of these in slot 0.
    switch  (numWarps)
      {
        default:
          if (threadIdx.x < warpSize)
            {
              x = (threadIdx.x < numWarps) ? sharedUint64[threadIdx.x] : static_cast<uint64_t>(-1);
              minWarp(x);
              __syncwarp();
              if (threadIdx.x == 0)
                {
                    *sharedUint64 = x;
                }
            }
          __syncthreads();
          break;
        case 2:
          if (threadIdx.x == 0)
            *sharedUint64 = min(sharedUint64[0], sharedUint64[1]);
          __syncthreads();
          break;
        case 1:
          break;
      }

    pair_t pair = *sharedPair;  // sharedPair is an alias for sharedUint64

    //  Restore original value and sign.
    pair.value += 1;
    if (pair.modulus & 1)
      pair.value = -pair.value;
    pair.modulus |= 1;

    if (STATS && threadIdx.x == 0)
      getSharedStats()->minPositiveCycles += clock(), getSharedStats()->reductionIterations += 1;

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
    unsigned long long int result = 0LL;
    while (xSz > warpSize)
      {
        xSz -= warpSize;
        //  Copy a block of x to shared memory for processing.
        if (threadIdx.x < warpSize)
          sharedUint32[threadIdx.x] = x[threadIdx.x + xSz];
        __syncthreads();
        //  Process the block in shared memory.
        for (size_t i = warpSize; i-- != 0;  )
          result = mod(result << 32 | sharedUint32[i], m);
        __syncthreads();
      }
    //  Now xSz <= warpSize.  Copy remainder of x to shared memory and process.
    if (threadIdx.x < xSz)
      sharedUint32[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    for (size_t i = xSz; i-- != 0;  )
      result = mod(result << 32 | sharedUint32[i], m);
    __syncthreads();
    return static_cast<uint32_t>(result);
  }

  //  Entry point into device-only code.
  template <bool STATS>
  __global__
  void
  kernel(uint32_t* buf, size_t uSz, size_t vSz, int numModuli, GmpCudaBarrier bar, struct GmpCudaGcdStats* stats = NULL)
  {
    struct GmpCudaGcdStats * sPtr;
    int totalModuliRemaining = numModuli * blockDim.x * gridDim.x;
    int ubits = (uSz + 1) * 32;  // somewhat of an overestimate
    int vbits = (vSz + 1) * 32;  // same here
    
    //  The arithmetic used on the clock requires the exact same type size all the time.
    //  It uses the fact that the arithmetic is modulo 2^32.
    if (STATS && threadIdx.x == 0)
        sPtr = initSharedStats(numModuli);

    //MGCD1: [Find suitable moduli]
    modulus_t q[MAX_MODULI_PER_THREAD];
    for (int i = numModuli, d = blockDim.x * gridDim.x, m = blockDim.x * blockIdx.x + threadIdx.x; i > 0; m += d)
      q[--i] = moduliList[m];  //  q[0] holds smallest modulus.

    //MGCD2: [Convert to modular representation]

    if (STATS && threadIdx.x == 0)
      sPtr->convertToModularCycles -= clock();

    uint32_t uq[MAX_MODULI_PER_THREAD], vq[MAX_MODULI_PER_THREAD];
    for (int i = 0; i < numModuli; i += 1)
      {
        uq[i] = modMP(buf,       uSz, q[i]);
        vq[i] = modMP(buf + uSz, vSz, q[i]);
      }

    if (STATS && threadIdx.x == 0)
      sPtr->convertToModularCycles += clock(), sPtr->reductionCycles -= clock();

    //MGCD3: [reduction loop]

//    if (blockIdx.x == 0 && threadIdx.x == 0)
 //     printf("###\n");

    pair_t pair;
    int t = numModuli - 1;  //  The top of a stack of moduli for this thread.

    pair.value = (vq[t]) ? toSigned(modDiv(uq[t], vq[t], q[t]), q[t]) : 0;
    pair.modulus = q[t].modulus;
    postMinPair<STATS>(pair, bar);
    pair = collectMinPair<STATS>(bar);
    
    do
      {
        pair_t newPair = {q[0].modulus, 0};
        if (t >= 0 && equals(pair.modulus, q[t]))  //  deactivate this modulus.
          {
            t -= 1;
          }
        totalModuliRemaining -= 1;
        if (t >= 0)
          {
            uint32_t p = pair.modulus;
            if (p > q[t].modulus)  //  Bring within range.
              p -= q[t].modulus;
            uint32_t tq = modDiv(modSub(uq[t], modMul(fromSigned(pair.value, q[t]), vq[t], q[t]), q[t]), p, q[t]);
            uq[t] = vq[t];
            vq[t] = tq;
            newPair.value = (vq[t]) ? toSigned(modDiv(uq[t], vq[t], q[t]), q[t]) : 0;
            newPair.modulus = q[t].modulus;
          }
        postMinPair<STATS>(newPair, bar);
        int tbits = ubits - (L - 1) + __ffs(abs(pair.value));  //  Conservative estimate--THIS NEEDS REVIEWED
        ubits = vbits;
        vbits = tbits;
        for (int i = 0; i < t; i += 1)
	        {
            uint32_t p = pair.modulus;
            if (p > q[i].modulus)  //  Bring within range.
              p -= q[i].modulus;
            uint32_t tq = modDiv(modSub(uq[i], modMul(fromSigned(pair.value, q[i]), vq[i], q[i]), q[i]), p, q[i]);
            uq[i] = vq[i];
            vq[i] = tq;
	        }
        pair = collectMinPair<STATS>(bar);
      }
    while (pair.value != 0 && totalModuliRemaining > ubits / (L - 1));
    
    if (pair.value != 0)  //  Ran out of moduli--means initial estimate was wrong.
      {
        if (blockIdx.x == 0 && threadIdx.x == 0)
          *buf = 0;
        return;
      } 

    //MGCD4: [Find SIGNED mixed-radix representation] Each "digit" is either positive or negative.

    if (STATS && threadIdx.x == 0)
      sPtr->reductionCycles += clock(), sPtr->mixedRadixCycles -= clock();

    pair_t* pairs = (pair_t *)buf + 1;

    if (t >= 0)
      {
        pair.value = toSigned(uq[t], q[t]);
        pair.modulus = q[t].modulus;
      }
    else
      {
        pair.value = 0;
        pair.modulus = q[0].modulus;
      }

    postAnyPair<STATS>(pair, bar);
    pair = collectAnyPair<STATS>(bar);

    do
      {
        *pairs++ = pair;
        pair_t newPair = {q[0].modulus, 0};
        if (t >= 0 && equals(pair.modulus, q[t]))  //  deactivate modulus.
          t -= 1;
        if (t >= 0)
          {
            uint32_t p = pair.modulus;
            if (pair.modulus > q[t].modulus)  //  Bring within range.
              p -= q[t].modulus;
            uq[t] = modDiv(modSub(uq[t], fromSigned(pair.value, q[t]), q[t]), p, q[t]);
            newPair.value = toSigned(uq[t], q[t]);
            newPair.modulus = q[t].modulus;
          }
        postAnyPair<STATS>(newPair, bar);
        for (int i = 0; i < t; i += 1)
          {
            uint32_t p = pair.modulus;
            if (pair.modulus > q[i].modulus)  //  Bring within range.
              p -= q[i].modulus;
            uq[i] = modDiv(modSub(uq[i], fromSigned(pair.value, q[i]), q[i]), p, q[i]);
          }
        pair = collectAnyPair<STATS>(bar);
//        cnt += 1;
//        if (blockIdx.x == 0 && threadIdx.x == 0)
//            printf("-->%d\n", cnt);
      }
    while (pair.value);

//    if (blockIdx.x == 0 && threadIdx.x == 0)
//      printf("---\n");

    if (blockIdx.x | threadIdx.x)
      return;

    *buf = pairs - reinterpret_cast<pair_t*>(buf);  // count of all the nonzero pairs, plus one more "pair" that includes buf[0] itself.

    if (STATS)
      {
        sPtr->mixedRadixCycles += clock();
        sPtr->totalCycles += clock();
        *stats = *sPtr;
      }
  }
}
//  All that follows is host only code.

void
__host__
GmpCudaDevice::initMaxGridSize()
{
  assert(BLOCK_SZ <= props.maxThreadsPerBlock);
  size_t sharedSz = sizeof(uint64_t) * WARP_SZ;
  if (collectStats)
    sharedSz += sizeof(struct GmpCudaGcdStats);
  int numBlocksPerSM;
  assert(cudaSuccess == cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, (collectStats) ? kernel<true> : kernel<false>, BLOCK_SZ, sharedSz));
  maxGridSize = props.multiProcessorCount * numBlocksPerSM;
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
  mpz_export(buf      , &uSz, -1, sizeof(uint32_t), 0, 0, u);
  mpz_export(buf + uSz, &vSz, -1, sizeof(uint32_t), 0, 0, v);
  memset(buf + uSz + vSz, 0, sizeof(buf) - (uSz + vSz) * sizeof(uint32_t));
  
  size_t sharedSz = sizeof(uint64_t) * WARP_SZ;
  if (collectStats)
    sharedSz += sizeof(struct GmpCudaGcdStats);

  //  Number of moduli needed is approximated by a function of the number of bits in the larger input.
  const float MODULI_MULTIPLIER = 1.6 - 0.014 * L;  //  Heuristically obtained formula.
  float numModuliNeeded = ceil(MODULI_MULTIPLIER * ubits / logf(ubits));
  
  int blockDim = BLOCK_SZ;
  
  gridSize = min(maxGridSize, static_cast<int>(ceil(numModuliNeeded/blockDim)));  //  grid size used to launch this kernel.
  if (gridSize > blockDim)
    gridSize = blockDim;
   
  //std::cout << "Block size = " << blockDim << "Grid size == " << gridSize << std::endl;
  
  int numThreads = blockDim * gridSize;    
  int modPerThread = static_cast<int>(ceil(numModuliNeeded / numThreads));
  if (modPerThread < 1)
    modPerThread = 1;

  if (numThreads * modPerThread > NUM_MODULI)
    throw std::runtime_error("Not enough moduli available for given input.");

  if (modPerThread > MAX_MODULI_PER_THREAD)
    throw std::runtime_error("Can't allocate enough per-thread storage for moduli needed");

  //  Allocate some extra space in the global buffer, so that modMP can assume it can safely read a multiple of
  //  warpSize words to get the entirety (+ more) of either parameter.
  uint32_t* globalBuf;

  assert(cudaSuccess == cudaMalloc(&globalBuf, std::max(sizeof(buf), sizeof(uint32_t) * (uSz + roundUp(vSz, props.warpSize)))));

  //  Copy parameters to global memory.
  assert(cudaSuccess == cudaMemcpy(globalBuf, buf, sizeof(buf), cudaMemcpyHostToDevice));

  //  Execute a specific kernel, based on whether we are collecting statistics.

  if (collectStats)
    kernel<true> <<<gridSize, blockDim, sharedSz>>>
      (globalBuf, uSz, vSz, modPerThread, *barrier, stats);
  else
    kernel<false><<<gridSize, blockDim, sharedSz>>>
      (globalBuf, uSz, vSz, modPerThread, *barrier);

  assert(cudaSuccess == cudaDeviceSynchronize());

  // Copy result from global memory and convert from mixed-radix to standard representation.
  assert(cudaSuccess == cudaMemcpy(buf, globalBuf, 2*sizeof(pair_t), cudaMemcpyDeviceToHost));  // Just size and 0th mixed-radix digit read now.
  
  assert(buf[0] != 0);  //  buf[0] == 0 means we ran out of moduli.

  if (buf[0] > 1)
    assert(cudaSuccess == cudaMemcpy(reinterpret_cast<pair_t*>(buf), globalBuf, buf[0] * sizeof(pair_t), cudaMemcpyDeviceToHost));//

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
