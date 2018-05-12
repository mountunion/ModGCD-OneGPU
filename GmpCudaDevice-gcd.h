/*  GmpCudaDevice-gcd.h

    These are declarations shared between the host side and the gpu side of the
    GmpCudaDevice::gcd method.
    
    K. Weber
    May 12, 2018.
*/

//  Error codes used by gcdKernel to tell gcd what's going on.
constexpr uint32_t GCD_KERNEL_ERROR   = 0; //  Error in the gcd kernel.
constexpr uint32_t GCD_REDUX_ERROR    = 0; //  Error in reduction phase.
constexpr uint32_t GCD_RECOVERY_ERROR = 1; //  Error in recovery phase.

//  Used to pass result back from gcdKernel to gcd.
typedef struct __align__(8) {uint32_t modulus; int32_t value;} pair_t;


