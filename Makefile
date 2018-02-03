##
##  If GMP is not installed on your system, you will need to provide a GMP library.
##  An easy way to do this is to download the source from http://gmplib.org/ and build it anywhere in the filesystem for which you have permissions.
##  Then point the GMPDIR variable to that directory.

##  These definitions assume GMP is installed on system
GMPDIR=
GMPI=
GMPL=-lgmp

CUDA_ARCH=-arch=sm_60
#-arch=compute_20 ## forces JIT compilation for all GPU architectures >= 2.0

CXX=nvcc
CXXFLAGS= $(GMPI) -g -O2 --std c++11 --use_fast_math -m64 $(CUDA_ARCH)

.PHONY: all clean distclean

all: testmodgcd22 testmodgcd32

testmodgcd22: testmodgcd.o GmpCudaDevice-gcd22.o GmpCudaDevice.o GmpCudaBarrier.o
	$(CXX) $(CXXFLAGS)  $^ -o $@ $(GMPL)

testmodgcd32: testmodgcd.o GmpCudaDevice-gcd32.o GmpCudaDevice.o GmpCudaBarrier.o
	$(CXX) $(CXXFLAGS)  $^ -o $@ $(GMPL)

GmpCudaDevice.h: GmpCudaBarrier.h
	touch $@

testmodgcd.o: testmodgcd.cpp GmpCudaDevice.h
	$(CXX) $(CXXFLAGS) -c $<

GmpCudaBarrier.o: GmpCudaBarrier.cu GmpCudaBarrier.h
	$(CXX) $(CXXFLAGS) -c --device-c $< -o $@

GmpCudaDevice.o: GmpCudaDevice.cu GmpCudaDevice.h
	$(CXX) $(CXXFLAGS) -c --device-c $< -o $@

GmpCudaDevice-gcd22.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/22bit/moduli.h
	$(CXX) $(CXXFLAGS) -I moduli/22bit -c --device-c $< -o $@

GmpCudaDevice-gcd32.o: GmpCudaDevice-gcd.cu GmpCudaDevice.h moduli/32bit/moduli.h
	$(CXX) $(CXXFLAGS) -I moduli/32bit -c --device-c $<  -o $@

moduli/22bit/moduli.h: createModuli
	mkdir -p moduli/22bit
	ulimit -s 32768 && ./createModuli 22 > $@

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
