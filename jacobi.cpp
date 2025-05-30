#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <memory>
#include <omp.h>
#include <openacc.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#define OFFSET(x, y, m) (((x)*(m)) + (y))

struct SimulationParams {
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    double tol = 1.0e-6;
};

SimulationParams parse_args(int argc, char** argv) {
    SimulationParams params;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show this help message")
        ("size-x,n", po::value<int>(&params.n)->default_value(512)) 
        ("size-y,m", po::value<int>(&params.m)->default_value(512))
        ("iterations,i", po::value<int>(&params.iter_max)->default_value(1000000))
        ("tolerance,t", po::value<double>(&params.tol)->default_value(1.0e-6));

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            exit(0);
        }
    } catch (const po::error& e) {
        std::cout << "Error: " << e.what() << "\n";
        std::cout << desc << "\n";
        exit(1);
    }

    return params;
}

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
    
    #pragma acc parallel loop collapse(2) present(A, Anew)  // reduction(max:error)
    for(int j = 1; j < n-1; j++)
    {
        for(int i = 1; i < m-1; i++)
        {
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)]
                                   + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
            //error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]));
        }
    }
    return error;
}

double calcError(double* A, double* Anew, int m, int n, double* diff)
{
    double error = 0.0;
    /*
    #pragma acc parallel loop collapse(2) present(A, Anew) reduction(max:error)
    for(int j = 1; j < n-1; j++)
    {
        for(int i = 1; i < m-1; i++)
        {
            error = fmax(error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]));
        }
    }
    /**/

    ///*
    cublasHandle_t handle;
    cublasCreate(&handle);

    #pragma acc data present(A[0:n*m], Anew[0:n*m], diff[0:n*m])
    {
        #pragma acc host_data use_device(A, Anew, diff)
        {
            double alpha = 1.0;
            double beta = -1.0;
            cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, 
                       &alpha, Anew, m,
                       &beta,  A,    m,
                               diff, m);
            
            int result;
            cublasIdamax(handle, n*m, diff, 1, &result);
            
            double max_diff;
            cublasGetVector(1, sizeof(double), &diff[result-1], 1, &max_diff, 1);
            error = max_diff;
        }
    }

    cublasDestroy(handle);
    /**/

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
    SimulationParams params = parse_args(argc, argv);

    int n = params.n;
    int m = params.m;
    const int iter_max = params.iter_max;
    const double tol = params.tol;

    double error = 1.0;

    auto A_sh = std::shared_ptr<double[]>(new double[n*m]);
    auto Anew_sh = std::shared_ptr<double[]>(new double[n*m]);
    auto diff_ptr = std::shared_ptr<double[]>(new double[n*m]);

    double* A = A_sh.get();
    double* Anew = Anew_sh.get();
    double* diff = diff_ptr.get();

    nvtxRangePushA("init");
    initialize(A, Anew, m, n);
    nvtxRangePop();

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    double st = omp_get_wtime();
    int iter = 0;

    nvtxRangePushA("ACC Copy");
    #pragma acc data copy(A[0:n*m], Anew[0:n*m], diff[0:n*m])
    {
        nvtxRangePop();

        nvtxRangePushA("while");
        while(error > tol && iter < iter_max)
        {
            nvtxRangePushA("calc");
            //error = calcNext(A, Anew, m, n);
            calcNext(A, Anew, m, n);
            nvtxRangePop();

            if(iter % 1000 == 0) 
            {
                nvtxRangePushA("error calc");
                error = calcError(A, Anew, m, n, diff);
                nvtxRangePop();
                printf("%5d, %0.6f\n", iter, error);
            }

            nvtxRangePushA("swap");
            //swap(A, Anew, m, n);
            #pragma acc data present(A, Anew)
            std::swap(A, Anew);
            nvtxRangePop();

            iter++;
        }
        nvtxRangePop();
    }
    
    double runtime = omp_get_wtime() - st;
    printf(" total: %f s\n", runtime);

    printf("Printing matrix in file...\n");
    std::ofstream out_file;
    out_file.open("out_matxr.txt");

    out_file << (A[0]);
    for (int i = 1; i < m*n; i++)
    {
        out_file << "," << (A[i]);
    }

    printf("Result in: out_matxr.txt\n");

    return 0;
}