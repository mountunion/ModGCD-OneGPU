//  GmpCudaDevice.h

#include <gmp.h>
#include <stdint.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <sstream>
#include "GmpCudaBarrier.h"

namespace GmpCuda
{
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
      int modPerThread;
  };

  class GmpCudaDevice
  {
  private:
    GmpCudaBarrier * barrier;
    struct GmpCudaGcdStats * stats;
    struct cudaDeviceProp props;
    int deviceNum;
    int gridSize;
    bool collectStats;
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
