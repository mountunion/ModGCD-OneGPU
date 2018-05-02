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
#include "GmpCudaDevice.h"
using namespace GmpCuda;
#include <iostream>

//  Initialize the CUDA device.  The device to use can be set by cudaSetDevice.
//  If 0 < n < the device's number of SMs,
//  the device's number of SMs is changed to n.
//  Also initializes the global barrier.
GmpCudaDevice::GmpCudaDevice(int n)
{

  collectStats = false;  //  default.

  deviceNum = 0;
  assert(cudaSuccess == cudaGetDevice(&deviceNum));

  //  Initialize the device properties values.
  assert(cudaSuccess == cudaGetDeviceProperties(&props, deviceNum));

  assert(props.warpSize == WARP_SZ);  //  Assume a fixed warp size of 32 for the forseeable future.

  assert(cudaSuccess == cudaMalloc(&stats, sizeof(struct GmpCudaGcdStats)));

  initGcdOccupancy();
  
  //  Limit the grid--and the barrier size--to the number of SMs * kernel occupancy.
  maxGridSize = props.multiProcessorCount * gcdOccupancy;
  if (0 < n && n < maxGridSize)
    maxGridSize = n;
    
  gridSize = maxGridSize;

  barrier = new GmpCudaBarrier(maxGridSize);
}

GmpCudaDevice::~GmpCudaDevice()
{
  assert(cudaSuccess == cudaFree(stats));
  delete barrier;
}

struct GmpCudaGcdStats GmpCudaDevice::getStats() const
{
  struct GmpCudaGcdStats s;
  assert(collectStats && cudaSuccess == cudaMemcpy(&s, stats, sizeof(s), cudaMemcpyDeviceToHost));
  s.clockRateInKHz = props.clockRate;
  return s;
}
