/*=======================================================================*/
/* Approximates pi with the n-point quadrature rule 4 / (1+x**2)         */
/* applied to the integral of x from 0 to 1.                             */
/*=======================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const double M_pi = 3.14159265358979323846; /* reference value */

double calc_pi (unsigned n) {
  double h   = 1.0 / n;
  double sum = 0.0;
  
  int i;
  double partials[2] = {0,0};
  #pragma omp parallel for
  for (int thread = 0; thread < 2; thread++)
  {
    
    for (i = 0; i < n/2; i++)
    {
      double x = 0.5*thread + (i + 0.5) * h;
      partials[thread] += 4.0 / (1.0 + x * x);
    }
  }
  for (int thread = 0; thread < 2; thread++){
    sum = sum + partials[thread];
  }
  return h * sum;
}

int main(int argc, char* argv[]) {
  int n;

  if ( argc != 2 ) {
    fprintf(stderr, "usage: pi <num_iterations>\n");
    return 1;
  }

  n = atoi(argv[1]);

  if ( n > 0 ) {
    double pi = calc_pi(n);
    double err = pi - M_pi;
    printf("Calculated pi is %19.15f\n", pi);
    printf("Referenced pi is %19.15f\n", M_pi);
    printf("  Error in pi is %19.15f (%f%%)\n", err, err*100/M_pi);
  }

  return 0;
}


