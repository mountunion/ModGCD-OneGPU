/*  testmodgcd.cpp -- test program to give statistics on the modular gcd algorithm.

  Implementation of the modular integer gcd algorithm.

  First, modgcdInit() must be called to initialize some tables needed by modgcd.
  After that, modgcd(g, u, v) can be called to compute the gcd of u and v and put it in g.

  Runs in CUDA 9.
  
  Based on initial work by
  Authors:  Justin Brew, Anthony Rizzo, Kenneth Weber
            Mount Union College
            June 25, 2009

  Further revisions by 
  K. Weber  University of Mount Union
            weberk@mountunion.edu

            made object-oriented:     August, 2012
            
            made linux-oriented       August, 2015
            
            added compilation macro   March 2, 2018
            to turn off GPU code
            
  J. Brew   jbrew5662@gmail.com
  
            extended test capability  February, 2018
            to read test data from
            files

  Invocation:

    testmgcd* [ s ] [ r ] [ i increment ] [ g grid_size ] num_bits [ gcd_size ]

  where
    *           is the size of the moduli in bits,
    s           is optional; it causes statistics on each modular gcd call to be printed.
    r           is optional; it causes the test to use random numbers. If left off the test will use test input in /tests. 
    If input does not exist in /tests for num_bits then random numbers will be generated and stored for future tests.
    Default is false.
    i increment is optional; it gives the increment to be added to num_bits.
		If it is left off, only one size is tested.
    g grid_size is optional; it can vary from 1 to the number of MPs on the device.
		Default is the number of MPs on the device.
    num_bits    is the number of bits of the pseudorandomly generated test data, u and v.
    gcd_size    is optional; if not included in the invocation. Default is 1.

  Description:
  u and v are pseudorandomly generated positive integers having num_bits bits, with a gcd of at least
  gcd_size bits.  The variable num_bits starts at num_bits and is optionally incremented by increment, up to the physical limit of
  the number of threads available on the CUDA device used.


*/

#ifndef __linux__
#error Only linux supported
#endif

#if !defined(NO_GPU)
#include "GmpCudaDevice.h"
#endif
#include <iostream>
#include <iomanip>
#include <libgen.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <gmp.h>

inline time_t monotonicTime(void)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC_RAW, &now);
  return now.tv_sec * 1000 * 1000 * 1000 + now.tv_nsec;
}

const int NUM_REPS = 10;

using std::cin;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;

#if !defined(NO_GPU)
struct MyGmpCudaGcdStats : public GmpCuda::GmpCudaGcdStats
{
  MyGmpCudaGcdStats(const GmpCudaGcdStats& rhs) : GmpCudaGcdStats(rhs) {}
  float cyclesToMS(uint32_t cycles){return cycles * uint64_t{1000000} / clockRateInKHz / 1e6;}
  void printLine(int i, int g)
  {
     cout << fixed << setprecision(3)
           << i                                  << "\t"
           << blockDim                           << "\t"
           << blockDim * g                       << "\t"
           << cyclesToMS(convertToModularCycles) << "\t"
           << reductionIterations                << "\t"
           << cyclesToMS(reductionCycles)        << "\t"
           << cyclesToMS(minPositiveCycles)      << "\t"
           << cyclesToMS(minBarrierCycles)       << "\t"
           << mixedRadixIterations               << "\t"
           << cyclesToMS(mixedRadixCycles)       << "\t"
           << cyclesToMS(anyPositiveCycles)      << "\t"
           << cyclesToMS(anyBarrierCycles)       << "\t"
           << cyclesToMS(totalCycles)            << endl;
  }
};
#endif

int
main(int argc, char *argv[])
{
  unsigned int num_bits;
  unsigned int num_g_bits = 1;
  unsigned int increment = 0;
  unsigned int gridSize = 0;
  bool collectStats = false;
  bool random = false;
  bool newFile = false;

  if (argc == 1)
    {
      cout << "Usage: " << basename(argv[0]) << " [ s ] [ r ] [ i increment ] [ g grid_size ] num_bits [ gcd_size ]" << endl;
      return 0;
    }

  if (argv[1][0] == 's')
    {
      collectStats = true;
      argc -= 1;
      argv += 1;
    }
  if (argv[1][0] == 'r')
    {
      random = true;
      argc -= 1;
      argv += 1;
    }
  if (argv[1][0] == 'i')
    {
      sscanf(argv[2], "%d", &increment);
      argc -= 2;
      argv += 2;
    }
  if (argv[1][0] == 'g')
    {
      sscanf(argv[2], "%d", &gridSize);
      argc -= 2;
      argv += 2;
    }
  switch (argc)
    {
      default: exit(1);
      case 3: sscanf(argv[2], "%u", &num_g_bits);
      case 2: sscanf(argv[1], "%u", &num_bits);
    }


  int cnt;
  int devNo = 0;
#if defined(NO_GPU)
  cout << "Executing tests only on CPU" << endl;
#else
  cudaGetDeviceCount(&cnt);
  if ( cudaSuccess != cudaSetDevice(devNo)) {
    cout << "Error on cuda set device" << endl;
    exit(2);
  }
  cout << "Using device " << devNo << endl;
  cout << "Cuda device count = " << cnt << endl;
#endif

  time_t ttime;

  ttime = -monotonicTime();
#if !defined(NO_GPU)
  GmpCuda::GmpCudaDevice dev(gridSize);
#endif
  ttime += monotonicTime();

#if defined(NO_GPU)
  collectStats = false;
#else
  dev.setCollectStats(collectStats);

  cout << endl
       << "GMP version " << gmp_version
       << "; initialization time: " << ttime/1e6 << " ms."
       << endl
       << "Grid size = " << dev.getGridSize() << endl;
#endif

  mpz_t u, v, g, mod_g;
  mpz_init(g);
  mpz_init(mod_g);
  mpz_init(u);
  mpz_init(v);

  gmp_randstate_t state;
  gmp_randinit_mt(state);

  do
    {
      cout << "***************************** "
           << "Input size = " << num_bits << "; gcd bits = " << num_g_bits
           << " *****************************" << endl;
      if (collectStats)
        cout << "Test"    << "\t"
             << "block"   << "\t"
             << "number"  << "\t"
             << "convert" << "\t"
             << "reduct"  << "\t"
             << "reduct"  << "\t"
             << "min"     << "\t"
             << "min bar" << "\t"
             << "mixed"   << "\t"
             << "mixed"   << "\t"
             << "any"     << "\t"
             << "any bar" << "\t"
             << "total"   << endl
             << "number"  << "\t"
             << "size"    << "\t"
             << "moduli"  << "\t"
             << "time-ms" << "\t"
             << "iter"    << "\t"
             << "time-ms" << "\t"
             << "time-ms" << "\t"
             << "time-ms" << "\t"
             << "iter"    << "\t"
             << "time-ms" <<  "\t"
             << "time-ms" << "\t"
             << "time-ms" << "\t"
             << "time-ms" << endl;
      time_t atime = time_t{0};
      time_t mtime = time_t{0};
      int numErrs = 0;

      FILE * pFile;

      if(!random)
      {
        const std::string folder = "tests";
        struct stat sb;

        if (!(stat(folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
        {
            //create directory
            cout << "Tests Folder Does not exist. Creating tests\n";
            const int dir_err = mkdir("tests", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (-1 == dir_err)
            {
                printf("Error creating directory!\n");
                exit(3);
            }
        }

        std::string testFile;
        testFile = folder + "/" + std::to_string(num_bits) + "-" + std::to_string(num_g_bits);
        
        if (!(stat(testFile.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)))
        {
            //create file
            cout << "Tests File Does not exist. Creating tests/" << num_bits << "-" << num_g_bits << "\n";
            
            pFile = fopen (testFile.c_str(),"a+");
            if (pFile==NULL)
            {              
              printf("Error creating file!\n");
              exit(4);
            }

            newFile = true;
        } 
        else 
        {
          pFile = fopen (testFile.c_str(),"r");
        }
      }

      for (int i = 0; i < NUM_REPS; i += 1)
        {
          if(random)
          {
            do
              mpz_urandomb(u, state, num_bits - num_g_bits);
            while (mpz_cmp_ui(u, 0) == 0);
            do
              mpz_urandomb(v, state, num_bits - num_g_bits);
            while (mpz_cmp_ui(v, 0) == 0);
            do
              mpz_urandomb(g, state, num_g_bits);
            while (mpz_cmp_ui(g, 0) == 0);
            mpz_mul(u, g, u);
            mpz_mul(v, g, v);
          }
          else
          {
            if(newFile)
            {
              do
                mpz_urandomb(u, state, num_bits - num_g_bits);
              while (mpz_cmp_ui(u, 0) == 0);
              do
                mpz_urandomb(v, state, num_bits - num_g_bits);
              while (mpz_cmp_ui(v, 0) == 0);
              do
                mpz_urandomb(g, state, num_g_bits);
              while (mpz_cmp_ui(g, 0) == 0);
              mpz_mul(u, g, u);
              mpz_mul(v, g, v);

              mpz_out_str (pFile, 62, u);
              fprintf(pFile, "\n");
              
              mpz_out_str (pFile, 62, v);
              fprintf(pFile, "\n");
              
              fflush(pFile);
            }
            else
            {
              mpz_inp_str (u, pFile, 62);
              mpz_inp_str (v, pFile, 62);
            }
          }
          mtime -= monotonicTime();
          mpz_gcd(g, u, v);
          mtime += monotonicTime();
          
#if !defined(NO_GPU)
          try
            {
              atime -= monotonicTime();
              dev.gcd(mod_g, u, v);
              atime += monotonicTime();
            }
          catch (std::runtime_error e)
            {
              cout << e.what() << endl;
              return 0;
            }

          if (collectStats)
            {
              struct MyGmpCudaGcdStats s(dev.getStats());
              s.printLine(i, dev.getGridSize());
            }

          if (mpz_cmp(g, mod_g)!= 0)
            {
              char uS[mpz_sizeinbase(u, 16) + 1], vS[mpz_sizeinbase(v, 16) + 1];
              char gS[mpz_sizeinbase(g, 16) + 1], mod_gS[mpz_sizeinbase(mod_g, 16) + 1];
              mpz_get_str(uS, 16, u);
              mpz_get_str(vS, 16, v);
              mpz_get_str(gS, 16, g);
              mpz_get_str(mod_gS, 16, mod_g);
              cout << "Test " << i << " error\n";
              cout << "GMP_GCD = " << gS << endl;
              cout << "MOD_GCD = " << mod_gS << ", size = " << mpz_size(mod_g) << endl;
              numErrs += 1;
            }
#endif
        }

      if(!random)
      {
        fclose (pFile);
      }

      cout << fixed << setprecision(3);
#if !defined(NO_GPU)
      if (numErrs == 0)
        cout << "Modular gcd correct\n";
      cout << "MOD time(avg): " << atime/NUM_REPS/1e6 << " ms\n";
#endif

      cout << "GMP time     : " << mtime/NUM_REPS/1e6 << " ms\n";
#if !defined(NO_GPU)
      cout << "MOD/GMP ratio: " << (double)atime/(double)mtime << endl;
#endif

      num_bits += increment;
    }
  while (increment > 0);

  cout << endl;
  return 0;
}
