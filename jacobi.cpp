#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <omp.h>
#include <openacc.h>
#include <nvtx3/nvToolsExt.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

void initialize(double* A, double* Anew, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    /*
    for(int i = 0; i < m; i++){
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
    /**/


    int ind;
    double a, b;
    double step;

    a = 10, b = 20;
    step = (b - a) / (m - 1);
    for (int i = 0; i < m; i++)
    {
        ind = OFFSET(0, i, m);
        Anew[ind] = A[ind] = a + step * i;
    }

    a = 30, b = 20;
    step = (b - a) / (m - 1);
    for (int i = 0; i < m; i++)
    {
        ind = OFFSET(n-1, i, m);
        Anew[ind] = A[ind] = a + step * i;
    }

    a = 10, b = 30;
    step = (b - a) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        ind = OFFSET(i, 0, m);
        Anew[ind] = A[ind] = a + step * i;
    }

    a = 20, b = 20;
    step = (b - a) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        ind = OFFSET(i, m-1, m);
        Anew[ind] = A[ind] = a + step * i;
    }

}

double calcNext(double* A, double* Anew, int m, int n)
{
    double error = 0.0;
    
    #pragma acc parallel loop collapse(2) present(A, Anew) reduction(max:error)
    for(int j = 1; j < n-1; j++)
    {
        for(int i = 1; i < m-1; i++)
        {
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)]
                                   + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
            error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]));
        }
    }
    return error;
}

void swap(double* A, double* Anew, int m, int n)
{
    #pragma acc parallel loop collapse(2) present(A, Anew)
    for(int j = 1; j < n-1; j++)
    {
        for(int i = 1; i < m-1; i++)
        {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];    
        }
    }
}

int main(int argc, char** argv)
{
    int n = 512;
    int m = 512;
    const int iter_max = 1000000;

    if(argc > 1) n = atoi(argv[1]);
    if(argc > 2) m = atoi(argv[2]);

    const double tol = 1.0e-6;
    double error = 1.0;

    auto A_sh = std::shared_ptr<double[]>(new double[n*m]);
    auto Anew_sh = std::shared_ptr<double[]>(new double[n*m]);

    double* A = A_sh.get();
    double* Anew = Anew_sh.get();

    nvtxRangePushA("init");
    initialize(A, Anew, m, n);
    nvtxRangePop();

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    double st = omp_get_wtime();
    int iter = 0;

    nvtxRangePushA("ACC Copy");
    #pragma acc data copy(A[0:n*m], Anew[0:n*m])
    {
        nvtxRangePop();

        nvtxRangePushA("while");
        while(error > tol && iter < iter_max)
        {
            nvtxRangePushA("calc");
            error = calcNext(A, Anew, m, n);
            nvtxRangePop();

            nvtxRangePushA("swap");
            //swap(A, Anew, m, n);
            #pragma acc data present(A, Anew)
            std::swap(A, Anew);
            nvtxRangePop();

            if(iter % 10000 == 0) printf("%5d, %0.6f\n", iter, error);
            iter++;
        }
        nvtxRangePop();
    }
    
    double runtime = omp_get_wtime() - st;
    printf(" total: %f s\n", runtime);

    return 0;
}