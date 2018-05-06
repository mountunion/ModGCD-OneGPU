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
#CUDA_ARCH=-arch=sm_52 -gencode=arch=compute_61,code=sm_61
CXX=g++
CXXFLAGS=--std c++11 -O2 -m64

NVCC=nvcc
NVCCFLAGS= -g -O2 --std c++11 --use_fast_math -m64 $(CUDA_ARCH)

LD=nvcc
LDFLAGS=$(CUDA_ARCH)

GCD_KERN_FLAGS=-maxrregcount 32 --device-c

.PHONY: all clean distclean

all: testmodgcd testmodgcd32

##
## Used to generate executables for the timing reported in paper(s).
## The same executable can be run on all three target systems.
##
static:
	echo "Making portable executables "
	$(MAKE) clean
	$(MAKE) GMPL=-l:libgmp.a LDFLAGS="-Xcompiler -static-libstdc++ $(LDFLAGS)" CXXFLAGS="-static-libstdc++ $(CXXFLAGS)"

testmodgcd32: testmodgcd.o GmpCudaDevice-gcd32.o GmpCudaDevice.o GmpCudaBarrier.o
	$(LD) $(LDFLAGS) $^ -o $@ $(GMPL)

##
##  This target will only use Gnu MP and no GPU.
##  
testmodgcd: testmodgcd.cpp
	$(CXX) $(CXXFLAGS) -DNO_GPU $^ -o $@ $(GMPL)

GmpCudaDevice.h: GmpCudaBarrier.h
	touch $@

testmodgcd.o: testmodgcd.cpp GmpCudaDevice.h
	$(NVCC) $(NVCCFLAGS) -c $<

GmpCudaBarrier.o: GmpCudaBarrier.cu GmpCudaBarrier.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

GmpCudaDevice.o: GmpCudaDevice.cu GmpCudaDevice.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

GmpCudaDevice-gcd32.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/32bit/moduli.h
	$(NVCC) -I moduli/32bit $(NVCCFLAGS) $(GCD_KERN_FLAGS) -c $< -o $@

moduli/32bit/moduli.h: createModuli
	mkdir -p moduli/32bit
	./createModuli 32 > $@

createModuli: createModuli.cpp
	$(CXX) $(CXXFLAGS) $^ $(GMPL) -o $@

clean:
	rm *.o testmodgcd testmodgcd[0-9][0-9] || true

distclean: clean
	rm createModuli || true
	rm -rf moduli || true
	rm -rf tests || true
