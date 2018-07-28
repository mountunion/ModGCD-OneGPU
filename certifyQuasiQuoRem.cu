/*  certifyQuasiQuoRem.cu

    This program will certify that quasiQuoRem<false>(xf, yf) works correctly, 
    as long as xf and yf are integers and 1 <= xf, yf < 2^22,
    by testing the function on all possible input satisfying the preconditions.
    
    K. Weber
    20-July, 2018.
*/

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include "quasiQuoRem.h"

__global__ void kernel(bool* fail)
{
  constexpr uint32_t FLOAT_THRESHOLD = 1 << FLOAT_THRESHOLD_EXPT;

  for (uint32_t y = blockIdx.x * blockDim.x + threadIdx.x  + 1; y < FLOAT_THRESHOLD; y += blockDim.x * gridDim.x)
    {
      float yf = __uint2float_rz(y);
      
      //  Now make sure quasiQuoRem<false> satisfies postconditions.
      for (uint32_t x = 1; x < 2 * y; x += 1)
        {
          if (x > 1 && x == y)
            continue;
          float xf = __uint2float_rz(x);
          float qf = quasiQuoRem<false>(xf, xf, yf);
          if (yf > xf && xf >= 0.0f)
            continue;
          *fail = true;
          printf("Failed for x == %u and y == %u: qf == %f, xf = %f\n", x, y, qf, xf);
        }
      float yf2 = yf + yf;
      for (uint32_t x = 2 * y; x < 2 * FLOAT_THRESHOLD; x += 1)
        {
          float xf = __uint2float_rz(x);
          float qf = quasiQuoRem<false>(xf, xf, yf);
          if (yf2 > xf && xf >= 0.0f)
            continue;
          *fail = true;
          printf("Failed for x == %u and y == %u: qf == %f, xf = %f\n", x, y, qf, xf);
        }
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
