/*  GmpCudaBarrier.h -- provides declarations for the GmpCudaBarrier class.

  Constructor and other methods are declared in GmpCudaBarrier.cu.
 
  Implemented in January, 2018.

  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009
           
  See GmpCudaDevice.cu for more information.
*/

//  Uncomment the following line if you want to use cooperative groups
//  to perform grid-wide synchronization provided by CUDA 9.
//  Otherwise, a simple custom busy-wait barrier is used.

//#define USE_COOP_GROUPS

#include <stdint.h>
#ifdef USE_COOP_GROUPS
#include <cooperative_groups.h>
#endif

namespace GmpCuda
{
  class GmpCudaBarrier
  {
  private:
    volatile char * barrier;
    unsigned int row;
    size_t pitch;
    bool copy;
    __device__ volatile inline uint64_t * barRow(unsigned int r)
    {
#ifdef __CUDACC__
      return reinterpret_cast<volatile uint64_t *>(barrier + (r % 4) * pitch);
#endif
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
#ifdef __CUDACC__
      if (threadIdx.x < gridDim.x)
        {
          row += 1;
          if (threadIdx.x == 0)
            {
              barRow(row + 1)[blockIdx.x] = uint64_t{0};
              barRow(row    )[blockIdx.x] = x;
            }
        }
#endif
    }

    //  Only allow low gridDim.x threads on each multiprocessor to participate.
    //  Collect gridDim.x results in out variable of the low gridDim.x threads.
    //  No __syncthreads() done here--caller generally should.
    __device__ inline void collect(uint64_t& out)
    {
#ifdef __CUDACC__
#ifdef USE_COOP_GROUPS
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
#endif
    }
  };
}
