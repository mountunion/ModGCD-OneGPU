/* createModuli.cpp -- Program to generate moduli.h.

  Implementation of the modular integer gcd algorithm.

  First, mod_init() must be called to initialize some tables needed by mod_gcd.
  After that, mod_gcd(g, u, v) can be called to compute the gcd of u and v and put it in g.

  K. Weber--January, 2010
            additional modifications:       July, 2010
            further simplifications:        February, 2011
                                            Includes using float operations for modulus.
            reorganized:                    March 8, 2011
            eliminated parallel conversion
               to standard
              rep:                          June 22, 2011
            final cleanup:                  July, 2011
            modified to calculate inverses  July, 2012
            modified to read L from cin	    Jan, 2013


  Based on initial work by
  Authors: Justin Brew, Anthony Rizzo, Kenneth Weber
           Mount Union College
           June 25, 2009

*/
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdint.h>
#include <gmp.h>

static const int W = 64;
static const size_t MAX_NUM_MODULI = 1 << 19;  //  can go up to 1<<20 with ulimit = 32768

using namespace std;

static void mark_composite(uint32_t limit, char* sieve, size_t sieve_sz)
{
  uint32_t d = static_cast<uint32_t>(floor(sqrt((double)limit)));

  if ((d&1) == 0)
    d -= 1;                /*  Make d odd.  */

  while (d > 2)
    {
      uint32_t g = uint32_t{3}*5*7*11*13*17*19*23*29;
      uint32_t tmp = d;
      while (g != tmp)
          if (g > tmp)
            {
              g -= tmp;
              while ((g&1) == 0)
                  g >>= 1;
            }
          else
            {
              tmp -= g;
              while ((tmp&1) == 0)
                  tmp >>= 1;
            }
      switch (g)
        {
          size_t i, end;
          case  3: case  5: case  7: case 11:
          case 13: case 17: case 19: case 23: case 29:
              if (g != d)
                  break;
          case 1:
              i = limit % d;
              if (i & 1)
                  i += d;
              i >>= 1;
              end = (limit - d)>>1;
              if (sieve_sz < end)
              end = sieve_sz;
              while (i < end)
                  sieve[i] = !0, i += d;
        }
      d -= 2;
    }
}

/*
    Generate a list of the N largest k-bit odd primes.  If there will
    not be enough values to generate the whole list whatever is
    generated will be returned in LIST; it is assumed that LIST has at
    least N slots.  The return value of the function is how many slots were
    left empty in LIST.  The larger primes will be placed at the lower
    indices of LIST.
*/

size_t primes(uint32_t * list, size_t n, int k)
{
  uint32_t limit = (uint64_t{1} << k) - 1;
  uint32_t lowerLimit = 1 << (k - 1);
  char* sieve;
  size_t sieve_sz, inv_density;

  if ((limit&1) == 0)                /*  If limit is even,  */
      limit -= 1;                /*  make limit next smaller odd number.  */

  if (n == 0 || limit < 3)
      return n;

  /*  inv_density is the inverse of the density of primes in ODD
      integers at limit;

          inv_density == (number of odd integers)/(number of primes)
                      ~= (limit/2)/(limit/ln(limit))
                      == ln(limit)/2
  */
  inv_density = ceil(log((double)limit)/2.0);

  /*  The sieve should be approximately large enough to find n primes.  */
  sieve_sz = (n > static_cast<uint32_t>(-1)/inv_density) ? static_cast<uint32_t>(-1) : n * inv_density;

  /*  Now try to acquire memory on the stack for the sieve.
      The sieve needs only to index odd numbers.
  */
  while ((sieve = (char *)alloca(sieve_sz)) == NULL)
    {
      sieve_sz /= 2;               /*  Ask for half as much.  */
      if (sieve_sz == 0)
          return n;                /*  Could not obtain any memory.  */
    }

  /*  Now strike non-primes from the sieve and harvest primes.  */
  do
    {
      size_t i;
      uint32_t d;

      if (sieve_sz > limit/2)
          sieve_sz = limit/2;

      memset(sieve, 0, sieve_sz);

      mark_composite(limit, sieve, sieve_sz);

      /*  Harvest primes.  */
      for (i = 0; i < sieve_sz; i++)
        {
          if (sieve[i])
            continue;                         /*  Composite.  */
          *list = limit - (i << 1);
          if (*list++ < lowerLimit)
            return n;
          n -= 1;
          if (n == 0)
            return n;
        }

      limit -= 2*sieve_sz;
      if (sieve_sz/n > inv_density)
          sieve_sz = n * inv_density;
    }
  while (limit >= 3);

  return n;
}

int main(int argc, char *argv[])
{
  int L;
  if (argc < 2)
    {
      cerr << "No L value provided" << endl;
      exit(1);
    }

  L = atoi(argv[1]);

  if (L < 2)
    {
      cerr << "Size requested is too small." << endl;
      exit(1);
    }

  if (L > 32)
    {
      cerr << "Size requested is too large." << endl;
      exit(1);
    }

  cout << "//  AUTOMATICALLY GENERATED by createModuli: do not edit" << endl
       << endl;

  uint32_t moduliList[MAX_NUM_MODULI];
  uint64_t mInvList[MAX_NUM_MODULI];
  size_t   mListSize = MAX_NUM_MODULI - primes(moduliList, MAX_NUM_MODULI, L);

  size_t mListSizeOriginal = mListSize;
  mListSize = 0;
  mpz_t FC, J, DJ_FC, Qcr, Ncr;
  mpz_init_set_ui(FC,  2);
  mpz_pow_ui(FC, FC, W + L - 1);  //  FC <-- 2^(W + L - 1)
  mpz_init(J);
  mpz_init(DJ_FC);
  mpz_init(Qcr);
  mpz_init(Ncr);
  for (size_t i = 0; i < mListSizeOriginal; i += 1)
    {
      uint32_t D = moduliList[i];
      mpz_fdiv_q_ui(J, FC, D);
      mpz_add_ui(J, J, 1);          //  J <-- FC / D + 1
      mpz_mul_ui(DJ_FC, J, D);
      mpz_sub(DJ_FC, DJ_FC, FC);    //  DJ_FC <-- D * J - FC
      mpz_cdiv_q(Qcr, J, DJ_FC);    //  Qcr <-- ceil(J / DJ_FC)
      mpz_mul_ui(Ncr, Qcr, D);
      mpz_sub_ui(Ncr, Ncr, 1);      //  Ncr <-- Qcr * D - 1
      if (mpz_sizeinbase(Ncr, 2) > W)
        {
          moduliList[mListSize] = moduliList[i];    //  Compress list.
          mpz_export(mInvList + mListSize++, NULL, -1, sizeof(uint64_t), 0, 0, J);
        }
    }

  cout   << "//  A list of " << L << "-bit primes, selected so that DBM_a(N, J) will always be accurate." << endl
         << "//  See Cavagnino & Werbrouck," << endl
         << "//      Efficient Algorithms for Integer Division by Constants Using Multiplication," << endl
         << "//      The Computer Journal, Vol. 51 No. 4, 2008." << endl
         << endl
         << "#include <stdint.h>" << endl
         << "typedef struct {uint32_t modulus; " << ((W == 32) ? "uint32_t" : "uint64_t") << " inverse;} modulus_t;" << endl
         << "static const int L = " << L << ";" << endl
         << "static const int W = " << W << ";" << endl
         << "static const int NUM_MODULI = " << mListSize << ";" << endl
         << "static __device__ const modulus_t moduliList[] = " << endl
         << "{" << endl;
  for (size_t i = 0; i < mListSize; i += 1)
    cout << "\t{" << moduliList[i] << ", " << mInvList[i] << "}," << endl;
  cout   << "};" << endl;
}
