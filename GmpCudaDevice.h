/*  GmpCudaDevice.h -- provides declarations for the GmpCudaDevice class.

  Constructor and other methods are declared in GmpCudaDevice.cu.
  GmpCudaDevice::gcd is declared in GmpCudaDevice-gcd.cu.

  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009
           
  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  See GmpCudaDevice.cu for more information.
*/

#include <gmp.h>
#include <stdint.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <sstream>
#include "GmpCudaBarrier.h"

//  Uncomment the following line if you want to use cooperative groups
//  to perform grid-wide synchronization provided by CUDA 9.
//  Otherwise, a simple custom busy-wait barrier is used.

//#define USE_COOP_GROUPS

namespace GmpCuda
{
  constexpr int WARP_SZ = 32;  // GmpCudaDevice checks to see whether this is true.
  constexpr int L = 32;
  constexpr int W = 64;
  constexpr int NUM_MODULI = 1 << 17;
  typedef struct {uint32_t modulus; uint64_t inverse;} modulus_t;
#ifdef __CUDACC__
  extern __device__ const modulus_t moduliList[];
#endif
  struct GmpCudaGcdStats
  {
    uint32_t
      blockDim,
      reductionIterations,
      mixedRadixIterations,
      convertToModularCycles,
      reductionCycles,
      minPositiveCycles,
      mixedRadixCycles,
      anyPositiveCycles,
      anyBarrierCycles,
      minBarrierCycles,
      totalCycles;
    int clockRateInKHz;
  };

  class GmpCudaDevice
  {
  private:
    GmpCudaBarrier * barrier;
    struct GmpCudaGcdStats * stats;
    struct cudaDeviceProp props;
    int deviceNum;
    int gridSize;
    int maxGridSize;
    bool collectStats;
    int gcdOccupancy;
    void initGcdOccupancy();
  public:
    GmpCudaDevice(int);
    ~GmpCudaDevice();
    void gcd(mpz_t g, mpz_t u, mpz_t v) throw (std::runtime_error);
    struct GmpCudaGcdStats getStats() const;
    void inline setCollectStats(bool b){collectStats = b;}
    bool inline getCollectStats() const {return collectStats;}
    int inline getGridSize() const {return gridSize;}
    int inline getClockRate() const {return props.clockRate;}
  };
};
