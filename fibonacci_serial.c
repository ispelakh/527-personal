

#include <stdio.h>
#include <omp.h>


int fib(int n)
{
  int x, y, res;
  if (n < 2) 
	  return n;
#pragma omp parallel
    {
      #pragma omp single 
        {
          #pragma omp task shared(x)
  x = fib(n - 1);
          #pragma omp task shared(y)
  y = fib(n - 2);
          #pragma omp taskwait
            res = x+y;
        }
    }
  return res;

}


int main()
{
  int n,fibonacci;
  double starttime;
  printf("Please insert n, to calculate fib(n): \n");
  scanf("%d",&n);
  starttime=omp_get_wtime();

  fibonacci=fib(n);

  printf("fib(%d)=%d \n",n,fibonacci);
  printf("calculation took %lf sec\n",omp_get_wtime()-starttime);
  return 0;
}
