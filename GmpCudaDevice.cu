/*  GmpCudaDevice.cu -- provides API to the GPU kernel.

  Implementation of the modular integer gcd algorithm using L <= 32 bit moduli.

  This version is for a single device and uses shuffle mechanism for the min operation.
  January 11, 2018.

  Updated to run in CUDA 9.

  Backported to CUDA 8 to run on Owens cluster in Ohio Supercomputer Center.
  January 15, 2018.

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
#include <iostream>
//#include <bitset>

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

  // We assume warp size will always be a power of 2, even if it changes
  // for newer architectures.
 // std::bitset<8*sizeof(int)> tmp(props.warpSize);
 
  assert(props.warpSize == WARP_SZ);  //  Assume a fixed warp size of 32 for the forseeable future.

  assert(cudaSuccess == cudaMalloc(&stats, sizeof(struct GmpCudaGcdStats)));

  //  Limit the grid--and the barrier size--to the number of SMs * kernel occupancy.
  initMaxGridSize();
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
