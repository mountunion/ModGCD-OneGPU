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

    testmgcd [ r ] [ i increment ] [ f final_bits ] num_bits [ gcd_size ]

  where
    r            is optional; it causes the test to use random numbers. If left off the test will use test input in /tests. 
                 If input does not exist in /tests for num_bits then random numbers will be generated and stored for future tests.
                 Default is false.
    i increment  is optional; it gives the increment to be added to num_bits.
		             If it is left off, only one size is tested.
    f final_bits is optional; it gives the limiting size of num_bits.
		             If it is left off, the tests will continue until the program runs out of resources (most likely threads on the gpu).
    num_bits     is the number of bits of the pseudorandomly generated test data, u and v.
    gcd_size     is optional; if not included in the invocation. Default is 1.

  Description:
  u and v are pseudorandomly generated positive integers having num_bits bits, with a gcd of at least
  gcd_size bits.  The variable num_bits starts at num_bits and is optionally incremented by increment, up to the physical limit of
  the number of threads available on the CUDA device used.


*/

#ifndef __linux__
#error Only linux supported
#endif

#include <cassert>
#include <iostream>
#include <iomanip>
#include <libgen.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <climits>
#include <sys/stat.h>
#include "GmpCuda.h"

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

int
main(int argc, char *argv[])
{
  unsigned int num_bits;
  unsigned int num_g_bits = 1;
  unsigned int increment = 0;
  unsigned int final_bits = 0;  //  Stop after 1 test.
  bool random = false;
  bool newFile = false;

  if (argc == 1)
    {
      cout << "Usage: " << basename(argv[0]) << " [ r ] [ i increment ] [ f final ] num_bits [ gcd_size ]" << endl;
      return 0;
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
      final_bits = UINT_MAX;  //  if no f value specified, continue...
    }
    
  if (argv[1][0] == 'f')
    {
      sscanf(argv[2], "%d", &final_bits);
      argc -= 2;
      argv += 2;
    }
    
  switch (argc)
    {
      default: exit(1);
      case 3: sscanf(argv[2], "%u", &num_g_bits);
      case 2: sscanf(argv[1], "%u", &num_bits);
    }

  cout << "GMP version " << gmp_version << "." << endl;
  
#if defined(NO_GPU)
  cout << "Executing tests only on CPU." << endl;
#else
  time_t ttime;
  try
    {
      // Take the work of setting the device out of initialization timing.
      GmpCuda::GmpCudaDevice::setDevice(0); 
    }
  catch (std::runtime_error e)  //  Some error thrown on dev.
    {
      cout << e.what() << endl;
      return 0;
    }
  ttime = -monotonicTime();
  GmpCuda::GmpCudaDevice dev;
  ttime += monotonicTime();
  cout << "Max grid size = " << dev.getMaxGridSize() << endl
       << "Initialization time: " << ttime/1e6 << " ms." << endl;
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
           << "Input size = " << num_bits << " (" << num_bits/1024.0 << " Kibit); gcd bits = " << num_g_bits
           << " *****************************" << endl;
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
      
      bool warmup = true;

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
              if (warmup)  // Warm up GPU for timing.
                {
                  dev.gcd(mod_g, u, v);
                  warmup = false;
                }
              atime -= monotonicTime();
              dev.gcd(mod_g, u, v);
              atime += monotonicTime();
            }
          catch (std::runtime_error e)
            {
              cout << e.what() << endl;
              return 0;
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
  while (num_bits <= final_bits);

  cout << endl;
  return 0;
}
