/*  GmpCudaModuli.h -- provides declarations for moduli.

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
#include <stdint.h>

namespace GmpCuda
{
  constexpr int L = 32;
  constexpr int W = 64;
  constexpr int NUM_MODULI = 1 << 17;
  typedef struct {uint32_t modulus; uint64_t inverse;} modulus_t;
#ifdef __CUDACC__
  extern __device__ const modulus_t moduliList[];
#endif
};
