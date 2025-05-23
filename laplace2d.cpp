#include <math.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <openacc.h>
#include <nvtx3/nvToolsExt.h>
 
#define OFFSET(x, y, m) (((x)*(m)) + (y))
 
void initialize(double *restrict A, double *restrict Anew, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
 
    for(int i = 0; i < m; i++){
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
}

double calcNext(double *restrict A, double *restrict Anew, int m, int n)
{
    double error = 0.0;
    #pragma acc parallel loop present(A, Anew) collapse(2)
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            Anew[OFFSET(j, i, m)] = 0.25 * ( A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)]
                                           + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
            //error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
        }
    }
    return 1; // return error
}

double calcError(double *restrict A, double *restrict Anew, int m, int n)
{
    double error = 0.0;

    #pragma acc parallel loop copy(error) present(A, Anew) collapse(2)
    for( int j = 1; j < n-1; j++)
        for( int i = 1; i < m-1; i++ )
            error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));

    return error;
}

void swap(double *restrict A, double *restrict Anew, int m, int n)
{
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];    
        }
    }
}
 
void deallocate(double *restrict A, double *restrict Anew)
{
    free(A);
    free(Anew);
}