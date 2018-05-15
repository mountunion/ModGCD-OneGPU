##  Make for ModGCD-OneGPU.
##  Known to work on Ubuntu 16.04.
##  May require adjustments for your system.
##  Requires an NVidia CUDA device with compute capability 5.2 or higher.
##  Requires CUDA 9 or more recent to be installed on your system.
##  If Gnu MP is not installed on your system, you will also need to provide a Gnu MP library.
##
##  J. Brew jbrew5662@gmail.com
##  K. Weber weberk@mountunion.edu
##  February 26, 2018

##  To create executables for which the C++ runtime library and the Gnu MP library are
##  statically linked, execute make as follows:
##
##      make static
##
##  This is known to work on Ubuntu 16.04 with the gnu development toolchain.
##  Linking this way makes the executable more portable.

GMPL=-lgmp

CUDA_ARCH=-arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70

CXX=g++
CXXFLAGS=--std c++11 -O2 -m64

NVCC=nvcc
NVCCFLAGS= $(CXXFLAGS) --device-c 

LD=nvcc
LDFLAGS=$(CUDA_ARCH)

GCD_KERN_FLAGS=--use_fast_math $(CUDA_ARCH) -maxrregcount 32

.PHONY: all clean distclean

all: testmodgcd testmodgcd-nogpu

##
## Used to generate executables for the timing reported in paper(s).
## The same executable can be run on all three target systems.
##
static:
	echo "Making portable executables "
	$(MAKE) clean
	$(MAKE) GMPL=-l:libgmp.a LDFLAGS="-Xcompiler -static-libstdc++ $(LDFLAGS)"

testmodgcd: testmodgcd.o GmpCudaDevice-gcd.o GmpCudaDevice-gcdKernel.o GmpCudaDevice.o GmpCudaBarrier.o GmpCudaModuli.o
	$(LD) $(LDFLAGS) $^ -o $@ $(GMPL)

##
##  This target will only use Gnu MP and no GPU.
##  
testmodgcd-nogpu: testmodgcd.cpp GmpCuda.h
	$(CXX) $(CXXFLAGS) -DNO_GPU $< -o $@ $(GMPL)

testmodgcd.o: testmodgcd.cpp GmpCuda.h
	$(CXX) $(CXXFLAGS) -c $<

GmpCudaBarrier.o: GmpCudaBarrier.cu GmpCuda.h
	$(NVCC) $(NVCCFLAGS) -c $<

GmpCudaDevice.o: GmpCudaDevice.cu GmpCuda.h
	$(NVCC) $(NVCCFLAGS) -c $<

GmpCudaDevice-gcd.o: GmpCudaDevice-gcd.cu GmpCuda.h
	$(NVCC) $(NVCCFLAGS) -c $<

GmpCudaDevice-gcdKernel.o: GmpCudaDevice-gcdKernel.cu GmpCuda.h
	$(NVCC) $(NVCCFLAGS) $(GCD_KERN_FLAGS) -c $<

createModuli: createModuli.cpp GmpCuda.h
	$(CXX) $(CXXFLAGS) $< $(GMPL) -o $@

GmpCudaModuli.cpp: createModuli GmpCuda.h
	./createModuli > $@

GmpCudaModuli.o: GmpCudaModuli.cpp GmpCuda.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm *.o testmodgcd testmodgcd-nogpu || true

distclean: clean
	rm createModuli || true
	rm -rf GmpCudaModuli.cu || true
	rm -rf tests || true
