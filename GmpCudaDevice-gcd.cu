/*  GmpCudaDevice-gcd.cu -- provides GmpCudaDevice::gcd method.

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

//  Enforce use of CUDA 9 at compile time.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#else
#error Requires CUDA 9 or more recent
#endif

#include <cassert>
#include <cuda_runtime.h>
#include "GmpCuda.h"
using namespace GmpCuda;

// Round x up to the next larger multiple of b >= 2.
// Precondition: T must be an integral type, and x >= 0.
template <typename T>
inline
T
roundUp(T x, int b)
{
  return ((x - 1) / b + 1) * b;
}
  
//  Calculate the number of moduli needed to compute the GCD.
//  Number of moduli needed is approximated by a function of the number of bits in the larger input,
//  then rounded up to the next multiple of GCD_BLOCK_SZ.
inline
int
numModuliNeededFor(size_t numBits)
{
  constexpr float C_L = 1.6 - 0.015 * L; 
  return roundUp(static_cast<int>(ceil(C_L * numBits / logf(numBits))), GmpCudaDevice::GCD_BLOCK_SZ);
}

void
__host__
GmpCudaDevice::gcd(mpz_t g, mpz_t u, mpz_t v) // throw (std::runtime_error)
{
  assert(cudaSuccess == cudaSetDevice(0));

  if(mpz_cmp(u, v) < 0)
    mpz_swap(u, v);

  size_t ubits = mpz_sizeinbase(u, 2);
  size_t vbits = mpz_sizeinbase(v, 2);
  
  int numModuliNeeded = numModuliNeededFor(ubits);
  if (numModuliNeeded > NUM_MODULI)
    throw std::runtime_error("Not enough moduli available for given input.");
    
  int gridSize = numModuliNeeded/GCD_BLOCK_SZ;    
  if (gridSize > maxGridSize)
    throw std::runtime_error("Cannot allocate enough threads to support computation.");

  size_t numb = 8 * sizeof(uint32_t);
  size_t uSz = (ubits + numb - 1)/numb;
  size_t vSz = (vbits + numb - 1)/numb;

  //  Slightly overestimate size of parameters and size of result, which is a list of moduli pairs, to get size of buf. 
  //  Allocate some extra space in buf, so that modMP can assume it can safely read a multiple of
  //  warpSize words to get the entirety (+ more) of either parameter.
  size_t bufSz = (2*std::max((ubits + vbits)/64, vbits/(L-1)) + 2) * sizeof(uint32_t);
         bufSz = std::max(bufSz, sizeof(uint32_t) * (uSz + roundUp(vSz, WARP_SZ)));

  uint32_t* buf;
  assert(cudaSuccess == cudaMallocManaged(&buf, bufSz));

  //  Stage parameters into buf and zero fill rest of buf.
  mpz_export(buf,       &uSz, -1, sizeof(uint32_t), 0, 0, u);
  mpz_export(buf + uSz, &vSz, -1, sizeof(uint32_t), 0, 0, v);
  memset(buf + uSz + vSz, 0, bufSz - (uSz + vSz) * sizeof(uint32_t));

  barrier->reset();  //  Reset to use again.


  //  Launch kernels on all devices, using separate streams.
  int devIdx[devCount];
  cudaStream_t stream[devCount];
  for (int i = 0; i < devCount; i += 1)
    {
      devIdx[i] = i;
      assert(cudaSuccess == cudaSetDevice(i));
      assert(cudaSuccess == (cudaStreamCreate(&stream[i])));
      void* args[] = {&buf, &uSz, &vSz, &moduliList, barrier, devIdx + i, &devCount};
      assert(cudaSuccess == (*kernelLauncher)(gcdKernel, gridSize, GCD_BLOCK_SZ, args, 0, stream[i]));
    }

  //  Synchronize all streams and destroy streams.
  for (int i = 0; i < devCount; i += 1)
    {
      assert(cudaSuccess == cudaSetDevice(i));
      cudaStreamSynchronize(stream[i]);
      assert(cudaSuccess == cudaStreamDestroy(stream[i]));
    }

  // Convert from mixed-radix to standard representation.
  
  if (buf[0] == GCD_KERNEL_ERROR)
    {
      assert(cudaSuccess == cudaFree(buf));
      switch(buf[1])
        {
          case GCD_REDUX_ERROR:    throw std::runtime_error("Ran out of moduli in the reduction loop.");
          case GCD_RECOVERY_ERROR: throw std::runtime_error("Ran out of moduli in the recovery loop.");
          default:                 throw std::runtime_error("Unknown error in the gcd kernel.");
        }
    }

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

  assert(cudaSuccess == cudaSetDevice(0));
  assert(cudaSuccess == cudaFree(buf));
}
