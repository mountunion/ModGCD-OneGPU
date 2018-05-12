/*  GmpCuda.h -- provides all declarations for the GmpCuda namespace.


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

//  Uncomment the following line if you want to use cooperative groups
//  to perform grid-wide synchronization provided by CUDA 9.
//  Otherwise, a simple custom busy-wait barrier is used.
//#define USE_COOP_GROUPS

#ifdef USE_COOP_GROUPS
#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif
#endif

namespace GmpCuda
{
  class GmpCudaBarrier
  {
#ifdef __CUDACC__
  private:
    volatile char * barrier;
    unsigned int row;
    size_t pitch;
    bool copy;
    __device__ volatile inline uint64_t * barRow(unsigned int r)
    {
      return reinterpret_cast<volatile uint64_t *>(barrier + (r % 4) * pitch);
    }
  public:
    __host__                            GmpCudaBarrier(int gridSize);  // can only be originally constructed on host.
    __host__ __device__                 ~GmpCudaBarrier();
    __host__ __device__ GmpCudaBarrier& operator=     (const GmpCudaBarrier&  orig);
    __host__ __device__ GmpCudaBarrier& operator=     (      GmpCudaBarrier&& orig);
    __host__ __device__                 GmpCudaBarrier(      GmpCudaBarrier&& orig);
    __host__ __device__                 GmpCudaBarrier(const GmpCudaBarrier&  orig);
    
    __host__            void            reset();

    //  Inlining these device functions for execution speed.

    //  Only allow low gridDim.x threads on each multiprocessor to participate.
    //  Values in threads whose threadIdx.x == 0 will be shared with all multiprocessors.
    //  Values shared MUST be nonzero.
    __device__ inline void post(uint64_t x)
    {
      if (threadIdx.x < gridDim.x)
        {
          row += 1;
          if (threadIdx.x == 0)
            {
              barRow(row + 1)[blockIdx.x] = uint64_t{0};
              barRow(row    )[blockIdx.x] = x;
            }
        }
    }

    //  Only allow low gridDim.x threads on each multiprocessor to participate.
    //  Collect gridDim.x results in out variable of the low gridDim.x threads.
    //  No __syncthreads() done here--caller generally should.
    __device__ inline void collect(uint64_t& out)
    {
#if defined(USE_COOP_GROUPS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      cooperative_groups::this_grid().sync();   
      if (threadIdx.x < gridDim.x)
        {
          volatile uint64_t * bar = barRow(row) + threadIdx.x;
          out = *bar;
        }
#else
      if (threadIdx.x < gridDim.x)
        {
          volatile uint64_t * bar = barRow(row) + threadIdx.x;
          do
            out = *bar;
          while (!out);
        }
#endif
    }
#endif
  };

  class GmpCudaDevice
  {
  private:
    typedef void (*GcdKernelPtr_t)(uint32_t*, size_t, size_t, uint32_t*, GmpCudaBarrier);
    static GcdKernelPtr_t getGcdKernelPtr(void);
    GmpCudaBarrier* barrier;
    uint32_t* moduliList;
    int deviceNum;
    int maxGridSize;
#if defined(USE_COOP_GROUPS)
    bool deviceSupportsCooperativeLaunch;
#endif
  public:
    // Adjust WARPS_PER_BLOCK to change the block size--don't change BLOCK_SZ directly.
    // WARPS_PER_BLOCK must evenly divide WARP_SZ.
    static constexpr int WARP_SZ         = 32;  // GmpCudaDevice checks to see whether this is true.
    static constexpr int WARPS_PER_BLOCK = WARP_SZ / 4;               //  Provides most flexibility. 
    static constexpr int BLOCK_SZ        = WARP_SZ * WARPS_PER_BLOCK; 
    GmpCudaDevice(int);
    ~GmpCudaDevice();
    void gcd(mpz_t g, mpz_t u, mpz_t v) throw (std::runtime_error);
    int inline getMaxGridSize() const {return maxGridSize;}
  };
  
  //  For L == 32, the largest possible NUM_MODULI is 68181070.
  constexpr int NUM_MODULI = GmpCudaDevice::BLOCK_SZ * GmpCudaDevice::BLOCK_SZ; 
  constexpr int L          = 32;
  constexpr int W          = 64;
  
  extern const uint32_t moduli[];
}
