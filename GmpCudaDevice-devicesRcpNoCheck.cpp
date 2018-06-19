#include "GmpCuda.h"

//  This array needs to be ordered according to strcmp rules.
const char GmpCuda::GmpCudaDevice::devicesRcpNoCheck[][256] = 
{
  "GeForce GTX 1080",
  "GeForce GTX 980 Ti",
};

const size_t GmpCuda::GmpCudaDevice::NUM_DEVICES_RCP_NO_CHECK = sizeof(devicesRcpNoCheck)/256;
