# ModGCD-OneGPU
Modular Integer GCD for one GPU

See http://raider.mountunion.edu/~weberk/Spring2018Sabbatical/sabb-proposal-f2016.pdf for more information on the overall project, including references to additional resources.

Requires CUDA 9 development kit or more recent.  (See https://developer.nvidia.com/cuda-downloads for most recent.)

## Build commands

The following command will build executables `testmodgcd22` and `testmodgcd32`.  It works on linux--tested on Ubuntu 16.04.
> `make`

The following will clean out all but the `createModuli` executable and the `moduli` folder:
> `make clean`

The following will clean out everything but original source files:
> `make distclean`
