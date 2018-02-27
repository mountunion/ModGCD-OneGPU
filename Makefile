##  Make for ModGCD-OneGPU.
##  Known to work on Ubuntu 16.04.
##  May require adjustments for your system.
##  Requires an NVidia CUDA device with compute capability 5.2 or higher.
##  Requires CUDA 9 or more recent to be installed on your system.
##  If Gnu MP is not installed on your system, you will also need to provide a Gnu MP library.
##
##  K. Weber weberk@mountunion.edu
##  J. Brew jbrew5662@gmail.com
##  February 26, 2018

GMPL=-lgmp

CUDA_ARCH=-arch=compute_52 -code=sm_52,sm_60,sm_61

CXX=g++
CXXFLAGS=--std c++11 -O2 -m64

NVCC=nvcc
NVCCFLAGS= $(GMPI) -g -O2 --std c++11 --use_fast_math -m64 $(CUDA_ARCH)

LD=nvcc
LDFLAGS=$(CUDA_ARCH)

GCD_KERN_FLAGS=-maxrregcount 32 --device-c

.PHONY: all clean distclean

all: testmodgcd22 testmodgcd27 testmodgcd32

testmodgcd22: testmodgcd.o GmpCudaDevice-gcd22.o GmpCudaDevice.o GmpCudaBarrier.o
	$(LD) $(LDFLAGS) $^ -o $@ $(GMPL)

testmodgcd27: testmodgcd.o GmpCudaDevice-gcd27.o GmpCudaDevice.o GmpCudaBarrier.o
	$(LD) $(LDFLAGS) $^ -o $@ $(GMPL)

testmodgcd32: testmodgcd.o GmpCudaDevice-gcd32.o GmpCudaDevice.o GmpCudaBarrier.o
	$(LD) $(LDFLAGS) $^ -o $@ $(GMPL)

GmpCudaDevice.h: GmpCudaBarrier.h
	touch $@

testmodgcd.o: testmodgcd.cpp GmpCudaDevice.h
	$(NVCC) $(NVCCFLAGS) -c $<

GmpCudaBarrier.o: GmpCudaBarrier.cu GmpCudaBarrier.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

GmpCudaDevice.o: GmpCudaDevice.cu GmpCudaDevice.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

GmpCudaDevice-gcd22.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/22bit/moduli.h
	$(NVCC) -I moduli/22bit $(NVCCFLAGS) $(GCD_KERN_FLAGS) -c $< -o $@

GmpCudaDevice-gcd27.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/27bit/moduli.h
	$(NVCC) -I moduli/27bit $(NVCCFLAGS) $(GCD_KERN_FLAGS) -c $< -o $@

GmpCudaDevice-gcd32.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/32bit/moduli.h
	$(NVCC) -I moduli/32bit $(NVCCFLAGS) $(GCD_KERN_FLAGS) -c $< -o $@

moduli/22bit/moduli.h: createModuli
	mkdir -p moduli/22bit
	ulimit -s 32768 && ./createModuli 22 > $@

moduli/27bit/moduli.h: createModuli
	mkdir -p moduli/27bit
	ulimit -s 32768 && ./createModuli 27 > $@

moduli/32bit/moduli.h: createModuli
	mkdir -p moduli/32bit
	ulimit -s 32768 && ./createModuli 32 > $@

createModuli: createModuli.cpp
	$(CXX) $(CXXFLAGS) $^ $(GMPL) -o $@

clean:
	rm *.o testmodgcd22 testmodgcd32 || true

distclean: clean
	rm createModuli || true
	rm -rf moduli || true
	rm -rf tests || true
