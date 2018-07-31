/*  certifyQuoRemQuasi.cu

    This program will certify that quoRem<QUASI>(xf, yf) works correctly, 
    as long as xf and yf are integers, 1 <= xf < 2 * FLOAT_THRESHOLD, 
    and 1 <= yf < FLOAT_THRESHOLD, by testing the function on all possible 
    input satisfying the preconditions.
    
    K. Weber
    20-July, 2018.
*/

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include "quoRem.h"

__device__ inline void checkRange(bool* fail, uint32_t xInit, uint32_t xLimit, float yf, float limit)
{
  for (uint32_t x = xInit; x < xLimit; x += 1)
    {
      float rf;
      float qf = quoRem<QUASI>(rf, __uint2float_rz(x), yf);
      if (0.0f <= rf && rf < limit)
        continue;
      *fail = true;
      printf("Failed for x == %u and yf == %f: qf == %f, rf = %f\n", x, yf, qf, rf);
    }
}

__global__ void kernel(bool* fail)
{
  //  Check y == 1 first. (Possibly not necessary--analysis indicates it should always be OK.)
  float yf = __uint2float_rz(1);
  for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x  + 1; x < 2 *FLOAT_THRESHOLD; x += blockDim.x * gridDim.x)
    {
      float rf;
      float qf = quoRem<QUASI>(rf, __uint2float_rz(x), yf);
      if (rf == 0.0f)
        continue;
      *fail = true;
      printf("Failed for x == %u and yf == %f: qf == %f, rf = %f\n", x, yf, qf, rf);
    }
  //  Now test all divisors 1 < y < FLOAT_THRESHOLD.
  for (uint32_t y = blockIdx.x * blockDim.x + threadIdx.x + 2; y < FLOAT_THRESHOLD; y += blockDim.x * gridDim.x)
    {
      yf = __uint2float_rz(y);
      //  Now make sure quoRem<QUASI> satisfies postconditions.
      // for 1 <= x < 2 * y && x != y, then 0 <= rf < yf.
      checkRange(fail, 1,         y, yf, yf); // Possibly not necessary--analysis indicates it should always be OK.
      checkRange(fail, y + 1, 2 * y, yf, yf); // Possibly not necessary--analysis indicates it should always be OK.
      // for 2 * y <= x < 2 * FLOAT_THRESHOLD, then 0 <= rf < 2 * yf.
      checkRange(fail, 2 * y, 2 * FLOAT_THRESHOLD, yf, 2 * yf);  // Necessary.  Analysis indicates q could be too high.
    }
}

int main(void)
{
  printf("Starting\n");
  fflush(0);
  bool fail = false;
  bool* globalFail;
  assert(cudaSuccess == cudaMalloc(&globalFail, sizeof(fail)));
  assert(cudaSuccess == cudaMemcpy(globalFail, &fail, sizeof(fail), cudaMemcpyHostToDevice));
  struct cudaDeviceProp props;
  assert(cudaSuccess == cudaGetDeviceProperties(&props, 0));
  kernel<<<props.multiProcessorCount,1024>>>(globalFail);
  assert(cudaSuccess == cudaDeviceSynchronize());
  assert(cudaSuccess == cudaMemcpy(&fail, globalFail, sizeof(fail), cudaMemcpyDeviceToHost));
  printf("Device %s: %s.\n", props.name, fail ? "FAIL" : "PASS");
  printf("All done\n");
}
