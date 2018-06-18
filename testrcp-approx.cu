#include <cuda_runtime.h>
#include <cassert>
#include <stdio.h>
#include <stdint.h>

__device__
inline
float
fastReciprocal(float yf)
  {
    float rf;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rf) : "f"(yf));
    return rf;
  }
  
template <bool RCP_CAN_BE_HIGH>
__device__
inline
uint32_t
quasiQuoRem(float& xf, float yf)
{
  float qf = truncf(__fmul_rz(xf, fastReciprocal(yf)));
  xf = __fmaf_rz(qf, -yf, xf); 
  if (RCP_CAN_BE_HIGH)  //  Have to check to see if the approximation was one too high.
    {
      if (xf < 0.0f)
        xf += yf, qf -= 1.0f;
    }
  return __float2uint_rz(qf);
}


__global__ void kernel(bool* fail)
{
  uint32_t LIMIT = 3 * (1 << 22); // 3 * (1 << 22);

  for (uint32_t y = blockIdx.x * blockDim.x + threadIdx.x  + 1; y < LIMIT; y += blockDim.x * gridDim.x)
    {
      float yf = __uint2float_rz(y);
      for (uint32_t x = 1; x < LIMIT; x += 1)
        {
          float xf = __uint2float_rz(x);
          float qf = quasiQuoRem<false>(xf, yf);
          uint32_t rem = __float2uint_rz(xf);
          if (0.0 <= xf && rem < 2 * y)
            continue;
          *fail = false, printf("Failed for x == %u and y == %u: rem == %u, xf = %f\n", x, y, rem, xf);
          return;
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
