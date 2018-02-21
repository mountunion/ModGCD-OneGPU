/*  GmpCudaBarrier.cu -- provides a device wide synchronization barrier.
    
  Implemented in January, 2018.

  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009
           
  See GmpCudaDevice.cu for more information.

*/

//  Enforce use of CUDA 9 at compile time.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#else
#error Requires CUDA 9 or more recent
#endif

#include <cassert>
#include <cuda_runtime.h>
#include "GmpCudaBarrier.h"
#include <iostream>
using namespace GmpCuda;

//  Initialize the base pointer for the global barrier.
//  barrier points to a quadruple circular buffer in global memory used for sharing
//  nonzero uint64_t data among multiprocessors.
GmpCudaBarrier::GmpCudaBarrier(int gridSize) : copy(false), row(0)
{
  assert(cudaSuccess == cudaMallocPitch(&barrier, &pitch, gridSize * sizeof(uint64_t), 4));
  assert(cudaSuccess == cudaMemset(const_cast<char *>(barrier), 0xff, pitch * 4));
}

GmpCudaBarrier::~GmpCudaBarrier()
{
#if !defined(__CUDA_ARCH__)
  if (copy)
    return;
  assert(cudaSuccess == cudaFree(const_cast<char *>(barrier)));
#endif
}

//  Whenever a GmpCudaBarrier object is moved or copied, mark copy
//  so the destructor will not try to deallocate the global memory.
GmpCudaBarrier& GmpCudaBarrier::operator= (const GmpCudaBarrier& orig)
{
  pitch   = orig.pitch;
  barrier = orig.barrier;
  row     = orig.row;
  copy    = true;
  return *this;
}

//  On host, need to initialize the second row of the barrier to all zeros
//  via this method.
void GmpCudaBarrier::reset()
{
#if !defined(__CUDA_ARCH__)
  assert(cudaSuccess == cudaMemset(const_cast<char *>(barrier) + pitch, 0, pitch));
#endif
}

GmpCudaBarrier& GmpCudaBarrier::operator= (GmpCudaBarrier&& orig)
{
  *this = orig;
  return *this;
}

GmpCudaBarrier::GmpCudaBarrier(      GmpCudaBarrier&& orig){*this = orig;}
GmpCudaBarrier::GmpCudaBarrier(const GmpCudaBarrier&  orig){*this = orig;}
