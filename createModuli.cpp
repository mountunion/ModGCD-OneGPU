/* createModuli.cpp -- Program to generate moduli.h.

  Prepares a declaration for an array of moduli and ancillary data needed to compute inverses.
  
  Uses Sieve of Eratosthenes to find primes.

  See Cavagnino & Werbrouck
      Efficient Algorithms for Integer Division by Constants Using Multiplication
      The Computer Journal
      Vol. 51 No. 4, 2008.
      
  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009
           
  Further revisions by 
  K. Weber  University of Mount Union
            weberk@mountunion.edu
            
  See GmpCudaDevice.cu for revision history.
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdint.h>
#include <gmp.h>
#include "GmpCudaModuli.h"

using namespace std;
using namespace GmpCuda;

//  Symbolic constant for 2^L - 1.
constexpr uint32_t TWO_L_1 = static_cast<uint32_t>((uint64_t{1} << L) - 1);

//  sieve[i] represents the integer x == (2^L - 1) - 2*i.
static inline uint32_t integerAt(size_t i)
{
  return TWO_L_1 - 2 * i;
}

//  Return the index of the largest multiple of d in the range 2^(L-1)...2^L.
//  Note that odd integer x's position in the sieve is at
//  i == ((2^L - 1) - x) / 2.
static inline size_t sieveIndexOfLargestOddMultiple(uint32_t d)
{
  uint32_t r = TWO_L_1 % d;
  if (r % 2)
    r += d;
  return r / 2; // == ((2^L - 1) - ((2^L - 1) - r)) / 2;       
}

//  Return largest odd integer less than sqrt(x).
static inline uint32_t oddSqrt(uint32_t x)
{
  uint32_t d = static_cast<uint32_t>(floor(sqrt(static_cast<double>(x))));
  if (d % 2 == 0)
    d -= 1;
  return d;
}

//  Returns true iff d == k * p, 
//  where p is a prime, 3 <= p <= 29, and k >= 2 is an integer.
//  Precondition: d is odd.
static inline bool isMultipleOfSmallPrime(uint32_t d)
{
  uint32_t g1 = d;
  uint32_t g2 = uint32_t{3}*5*7*11*13*17*19*23*29;
  while (g1 != g2)
    {
      if (g1 > g2)
        {
          g1-= g2;
          while (g1 % 2 == 0)
            g1 /= 2;
        }
      else
        {
          g2 -= g1;
          while (g2 % 2 == 0)
            g2 /= 2;
        }
    }
  switch (g1)  // g1 == g2 == gcd(d, 3*5*7*11*13*17*19*23*29).
    {
      case  1: return false;
      case  3: 
      case  5: 
      case  7: 
      case 11: 
      case 13: 
      case 17: 
      case 19: 
      case 23: 
      case 29: return (d > g1);
      default: return true;
    }
}

int main(void)
{
  if (L < 2 || L > 32)
    {
      cerr << "L = " << L << " is invalid.";
      return 1;
    }

  //  Mark odd composite values in the range 2^(L-1) to 2^L, for L >= 2.
  //  Only odd values are represented in the sieve, since all even
  //  values > 2 are composite. 
  //  Sieve needs to be large enough to represent all 2^(L-2) odd integers
  //  in the range 2^(L-1)...2^L.
  constexpr uint32_t SIEVE_SZ = 1 << (L - 2);
  static char  sieve[SIEVE_SZ];                
  memset(sieve, 0,   SIEVE_SZ);
  for (uint32_t d = oddSqrt(TWO_L_1); d > 2; d -= 2)
    {
      if (isMultipleOfSmallPrime(d))  // Ignore, since will be handled by other values of d.
        continue;
      for (size_t i = sieveIndexOfLargestOddMultiple(d); i < SIEVE_SZ; i += d)
        sieve[i] = !0;
    }
  
  size_t numPrimes = 0;
  size_t numUsable = 0;
  mpz_t FC, J, DJ_FC, Qcr, Ncr;
  mpz_init_set_ui(FC,  2);
  mpz_pow_ui(FC, FC, W + L - 1);  //  FC <-- 2^(W + L - 1)
  mpz_init(J);
  mpz_init(DJ_FC);
  mpz_init(Qcr);
  mpz_init(Ncr);
  
  filebuf fbmod, fbinv;
  fbmod.open("GmpCudaModuli.cu",        std::ios::out);
  fbinv.open("GmpCudaInverseModuli.cu", std::ios::out);
  ostream omod(&fbmod), oinv(&fbinv);
  
  omod   << "//  AUTOMATICALLY GENERATED by createModuli: do not edit" << endl
         << endl
         << "//  A list of " << L << "-bit primes, selected so that DBM_a(N, J) will always be accurate." << endl
         << "//  See Cavagnino & Werbrouck," << endl
         << "//      Efficient Algorithms for Integer Division by Constants Using Multiplication," << endl
         << "//      The Computer Journal, Vol. 51 No. 4, 2008." << endl
         << endl
         << "#include \"GmpCudaModuli.h\"" << endl
         << "__device__ const uint32_t GmpCuda::moduliList[] = " << endl
         << "{" << endl;
  
  oinv   << "//  AUTOMATICALLY GENERATED by createModuli: do not edit" << endl
         << endl
         << "//  Companion inverses to moduli in GmpCudaModuli.cu."    << endl
         << endl
         << "#include \"GmpCudaModuli.h\""                             << endl
         << "__device__ const uint64_t GmpCuda::inverseModuliList[] = "<< endl
         << "{" << endl;
  
  //  Harvest primes from sieve and print out any usable ones.
  for (size_t i = 0; i < SIEVE_SZ; i += 1)
    {
      if (sieve[i])
        continue;                       //  Composite--ignore.       
      uint32_t D = integerAt(i);        //  D is prime.
      numPrimes += 1;
      mpz_fdiv_q_ui(J, FC, D);
      mpz_add_ui(J, J, 1);              //  J <-- FC / D + 1
      mpz_mul_ui(DJ_FC, J, D);
      mpz_sub(DJ_FC, DJ_FC, FC);        //  DJ_FC <-- D * J - FC
      mpz_cdiv_q(Qcr, J, DJ_FC);        //  Qcr <-- ceil(J / DJ_FC)
      mpz_mul_ui(Ncr, Qcr, D);
      mpz_sub_ui(Ncr, Ncr, 1);          //  Ncr <-- Qcr * D - 1
      if (mpz_sizeinbase(Ncr, 2) <= W)
        continue;                       //  Not usaable as modulus
      uint64_t Dinverse;
      mpz_export(&Dinverse, NULL, -1, sizeof(uint64_t), 0, 0, J);
      numUsable += 1;
      if (numUsable <= NUM_MODULI)
        {
          omod << "\t" << D        << "," << endl;
          oinv << "\t" << Dinverse << "," << endl;
        }
    }

  omod << "};" << endl;
  oinv << "};" << endl;
  
  fbmod.close();
  fbinv.close();
  
  cout << "There are " << numPrimes << " " << L << "-bit primes; "
       << numUsable << " are usable as moduli." << endl;
       
  if (numUsable >= NUM_MODULI)
    return 0;  
    
  cerr << "Error: "<< NUM_MODULI << " usable moduli are required." << endl;
  return 2;
}
