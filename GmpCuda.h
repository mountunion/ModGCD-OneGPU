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
#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif

namespace GmpCuda
{
  constexpr int WARP_SZ = 32; // GmpCudaDevice checks to see whether this is true.

#ifdef __CUDACC__  
  class GmpCudaBarrier
  {
  private:
    //  Set USE_COOP_GROUPS_IF_AVAILABLE to true if you want to use cooperative groups
    //  to perform grid-wide synchronization provided by CUDA 9.
    //  Otherwise, a simple custom busy-wait barrier is used.
    static constexpr bool USE_COOP_GROUPS_IF_AVAILABLE = false;
  
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      if (USE_COOP_GROUPS_IF_AVAILABLE)
        cooperative_groups::this_grid().sync();
      constexpr bool SPIN_WAIT = !USE_COOP_GROUPS_IF_AVAILABLE;
#else
      constexpr bool SPIN_WAIT = true;
#endif
      if (threadIdx.x < gridDim.x)
        {
          volatile uint64_t * bar = barRow(row) + threadIdx.x;
          do
            out = *bar;
          while (SPIN_WAIT && !out);
        }
    }
  };
#endif

  class GmpCudaDevice
  {
  private:
#if defined(__CUDACC__)
    typedef cudaError_t (*launcher_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
#else
    typedef void GmpCudaBarrier;
    typedef int (*launcher_t)(void);
#endif
    GmpCudaBarrier* barrier;
    uint32_t* moduliList;
    int deviceNum;
    int maxGridSize;
    static const void* gcdKernel;
    launcher_t kernelLauncher;
  public:
    static constexpr int GCD_BLOCK_SZ = WARP_SZ << 3; // Must be a power of 2 and a multiple of WARP_SZ.
    static constexpr int MAX_THREADS  = GCD_BLOCK_SZ * GCD_BLOCK_SZ;
    //  Error codes used by gcdKernel to tell gcd what's going on.
    static constexpr uint32_t GCD_KERNEL_ERROR   = 0; //  Error in the gcd kernel.
    static constexpr uint32_t GCD_REDUX_ERROR    = 0; //  Error in reduction phase.
    static constexpr uint32_t GCD_RECOVERY_ERROR = 1; //  Error in recovery phase.
#if defined(__CUDACC__)
    //  pair_t is used to pass result back from gcdKernel to gcd.
    typedef struct __align__(8) {uint32_t modulus; int32_t value;} pair_t;
#endif
    GmpCudaDevice(int devNo = 0);
    ~GmpCudaDevice();
    void gcd(mpz_t g, mpz_t u, mpz_t v) throw (std::runtime_error);
    int inline getMaxGridSize() const {return maxGridSize;}
  };
  
  constexpr int L          = 32;
  constexpr int W          = 64;
  constexpr int NUM_MODULI = GmpCudaDevice::MAX_THREADS;
  // For L == 32, the largest possible NUM_MODULI is 68181070.
  
  extern const uint32_t moduli[];  
}
