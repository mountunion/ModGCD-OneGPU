# ModGCD-OneGPU
Modular Integer GCD for one GPU

See [An Implementation of the Modular Integer GCD Algorithm on a Single GPU](https://github.com/mountunion/ModGCD-OneGPU/blob/master/ModGCD-OneGPU.pdf) for more information on the overall project, including references to additional resources.

Requires CUDA 9 development kit or more recent.  (See https://developer.nvidia.com/cuda-downloads for most recent.)

## Build commands

The following command will build executables testmodgcd, testmodgcd22, testmodgcd27`and testmodgcd32.  It has been tested on Ubuntu 16.04.
> `make`

The following will clean out all but the `createModuli` executable and the `moduli` folder:
> `make clean`

The following will clean out everything but original source files:
> `make distclean`

The following command is used to build a somewhat more portable executable, that is used for timing on several similar systems:
> `make static`
