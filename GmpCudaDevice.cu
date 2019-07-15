/*  GmpCudaDevice.cu -- provides constructor for GmpCudaDevice objects.

  Implementation of the modular integer gcd algorithm using L <= 32 bit moduli.
  
  Reference: Weber, Trevisan, Martins 2005. A Modular Integer GCD algorithm
             Journal of Algorithms 54, 2 (February, 2005) 152-167.

             Note that there is an error in Fig. 2, which shows that the
             final result can be recovered as the mixed radix representation
             is calculated.  In actuality, all the mixed radix digits and moduli
             must be computed before the actual GCD can be recovered.

  This version is for a single device.

  Runs in CUDA 9.
  
  Based on initial work by
  Authors:  Justin Brew, Anthony Rizzo, Kenneth Weber
            Mount Union College
            June 25, 2009

  Further revisions by 
  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  History:  Basic 16 bit version      January, 2010 
  
            Additional modifications  July, 2010
            
            Further simplifications   February, 2011
            including using float 
            operations for modulus
            
            Reorganized               March 8, 2011
            
            Eliminated parallel       June 22, 2011
            conversion to standard
            representation
                                  
            "Final" cleanup           July, 2011

            Modified to allow up      June, 2012
	          to 32 bit moduli         

            Made object-oriented      August, 2012

            More cleanup              January, 2013
            limited to arch >= 2.0
            (anyPair uses __ballot)

            Bug fixed in barrier      March, 2014
						Uses fixed number of
					  threads, but arbitrary
            number of moduli.
            Also overlaps communi-
            cation with computation.

            Further cleanup           July, 2014
            
            Ported to CUDA 9.         January 11, 2018
            Uses shuffle mechanism
            for the min operation
            and ballot mechanism
            to select a nonzero
            value

            Put GmpCudaDevice::gcd    January 22, 2018
            in its own file named
            GmpCudaDevice-gcd.cu
            Added capability to use 
            more than warpSize SMs.
            
            Split out GmpCudaBarrier  January, 2018
            in files GmpCudaBarrier.h
            and GmpCudaBarrier.cu

            Modified to allow large   February 17, 2018
            grid sizes up to maximum 
            occupancy.  
            
            Corrected errors in       May 2, 2018
            modInv.
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

//  Initialize all available CUDA device(s).  Assume all CUDA devices are homogeneous for now (FIXME).  
//  (Remember that the available devices can be set using a CUDA environment variable.
//  Also initializes the global barrier and the moduli list.
GmpCudaDevice::GmpCudaDevice(void)
{
  int devCount;
  assert(cudaSuccess == cudaGetDeviceCount(&devCount));
  
  int deviceNum = 0;
  assert(cudaSuccess == cudaSetDevice(deviceNum)); 
  //  assert(cudaSuccess == cudaGetDevice(&deviceNum));

  //  Initialize the device properties values.
  struct cudaDeviceProp props;
  assert(cudaSuccess == cudaGetDeviceProperties(&props, deviceNum));

  assert(props.warpSize == WARP_SZ);  //  Assume a fixed warp size of 32 for the forseeable future.
  
  assert(GCD_BLOCK_SZ <= props.maxThreadsPerBlock);

  //  The kernel launcher we want to use depends on whether the device supports cooperative launch.
  kernelLauncher = (props.cooperativeLaunch == 1)
    ? static_cast<cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t)>(&cudaLaunchCooperativeKernel)
    : static_cast<cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t)>(&cudaLaunchKernel);
    
  //  The gcd kernel we want to use depends on whether the device has a "good" rcp.approx instruction.
  gcdKernel = getGcdKernel(props.name);

  //  Limit the grid, and thus, the barrier size also.
  int gcdOccupancy;
  assert(cudaSuccess == cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gcdOccupancy, gcdKernel, GCD_BLOCK_SZ, 0));
  maxGridSize = min(GCD_BLOCK_SZ, props.multiProcessorCount * gcdOccupancy);    
    
  barrier = new GmpCudaBarrier(maxGridSize);
  
  //  Copy moduli to device.
  size_t maxModuliBytes = maxGridSize * GCD_BLOCK_SZ * sizeof(uint32_t);
  assert(cudaSuccess == cudaMalloc(&moduliList, maxModuliBytes));
  assert(cudaSuccess == cudaMemcpy(moduliList, moduli, maxModuliBytes, cudaMemcpyHostToDevice));
}

GmpCudaDevice::~GmpCudaDevice()
{
  assert(cudaSuccess == cudaFree(moduliList));
  delete barrier;
}
