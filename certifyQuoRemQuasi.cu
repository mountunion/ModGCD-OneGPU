/*  certifyQuoRemQuasi.cu

    This program will certify that quoRem<QUASI>(rf, xf, yf) works correctly.
    Preconditions: 
      xf and yf are integers
      xf == yf == 1.0f || xf != yf
      1 <= xf < FLOAT_THRESHOLD * 2
      1 <= yf < FLOAT_THRESHOLD
    Postcondition tested:
      quoRem<QUASI>(rf, xf, yf) returns in rf a value satisfying 
        0.0f <= rf < yf     when 0      <  xf < yf * 2
        0.0f <= rf < yf * 2 when yf * 2 <= xf < FLOAT_THRESHOLD * 2
      for all possible input xf and yf satisfying the preconditions.
    
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
      float xf = __uint2float_rz(x);
      float qf = quoRem<QUASI>(rf, xf, yf);
      if (0.0f <= rf && rf < limit)
        continue;
      *fail = true;
      printf("Failed for xf == %f and yf == %f: qf == %f, rf = %f\n", xf, yf, qf, rf);
    }
}

__global__ void kernel(bool* fail)
{
  for (uint32_t y = blockIdx.x * blockDim.x + threadIdx.x + 1; y < FLOAT_THRESHOLD; y += blockDim.x * gridDim.x)
    {
      float yf = __uint2float_rz(y);
      checkRange(fail, 1, max(y, 2), yf, yf); //  Will include check for quoRem<QUASI>(rf, 1.0f, 1.0f).
      checkRange(fail, y + 1, y * 2, yf, yf);
      checkRange(fail, y * 2, FLOAT_THRESHOLD * 2, yf, 2 * yf);
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
  assert(cudaSuccess == cudaFree(globalFail));
  printf("Device %s: %s.\n", props.name, fail ? "FAIL" : "PASS");
  printf("All done\n");
}
