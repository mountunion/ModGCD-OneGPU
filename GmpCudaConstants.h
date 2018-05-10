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
  // Adjust WARPS_PER_BLOCK to change the block size--don't change BLOCK_SZ directly.
  // WARPS_PER_BLOCK must evenly divide WARP_SZ.
  constexpr int WARP_SZ         = 32;  // GmpCudaDevice checks to see whether this is true.
  constexpr int WARPS_PER_BLOCK = WARP_SZ / 4;               //  Provides most flexibility. 
  constexpr int BLOCK_SZ        = WARP_SZ * WARPS_PER_BLOCK; 
  constexpr int NUM_MODULI      = BLOCK_SZ * BLOCK_SZ;  //  Largest possible is 68181070.
  constexpr int L               = 32;
  constexpr int W               = 64;
};
