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
#include "GmpCudaConstants.h"

namespace GmpCuda
{
  extern const uint32_t moduli[];
  extern __global__ void gcdKernel(uint32_t*, size_t, size_t, uint32_t*, GmpCudaBarrier);
  typedef struct __align__(8) {uint32_t modulus; int32_t value;} pair_t;
  class GmpCudaDevice
  {
  private:
    GmpCudaBarrier * barrier;
    uint32_t* moduliList;
    int deviceNum;
    int maxGridSize;
  public:
    GmpCudaDevice(int);
    ~GmpCudaDevice();
    void gcd(mpz_t g, mpz_t u, mpz_t v) throw (std::runtime_error);
    int inline getMaxGridSize() const {return maxGridSize;}
  };
};
