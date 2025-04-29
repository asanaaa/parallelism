#include <boost/program_options.hpp>
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include <memory> 
#include <nvtx3/nvToolsExt.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

void initialize(double *A, double *Anew, int m, int n) {
   memset(A, 0, sizeof(double)*m*n);
   memset(Anew, 0, sizeof(double)*m*n);

   double topLeft = 10.0;
   double topRight = 20.0;
   double bottomRight = 30.0;
   double bottomLeft = 20.0;

   // Верхняя граница
   for (int j = 0; j < m; ++j) {
       double alpha = static_cast<double>(j) / (m - 1);
       A[OFFSET(0, j, m)] = Anew[OFFSET(0, j, m)] = (1 - alpha) * topLeft + alpha * topRight;
   }

   // Нижняя граница
   for (int j = 0; j < m; ++j) {
       double alpha = static_cast<double>(j) / (m - 1);
       A[OFFSET(n - 1, j, m)] = Anew[OFFSET(n - 1, j, m)] = (1 - alpha) * bottomLeft + alpha * bottomRight;
   }

   // Левая граница
   for (int i = 0; i < n; ++i) {
       double alpha = static_cast<double>(i) / (n - 1);
       A[OFFSET(i, 0, m)] = Anew[OFFSET(i, 0, m)] = (1 - alpha) * topLeft + alpha * bottomLeft;
   }

   // Правая граница
   for (int i = 0; i < n; ++i) {
       double alpha = static_cast<double>(i) / (n - 1);
       A[OFFSET(i, m - 1, m)] = Anew[OFFSET(i, m - 1, m)] = (1 - alpha) * topRight + alpha * bottomRight;
   }
}

namespace po = boost::program_options;

int main(int argc, char** argv) {
    int n, m, iter_max;
    double tol;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("size,n", po::value<int>(&n)->default_value(512), "size (n x n)")
        ("tol,t", po::value<double>(&tol)->default_value(1.0e-6), "tolerance")
        ("max_iter,i", po::value<int>(&iter_max)->default_value(1000000), "maximum number of iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    m = n;

    std::unique_ptr<double[]> A_smart = std::make_unique<double[]>(n*m);
    std::unique_ptr<double[]> Anew_smart = std::make_unique<double[]>(n*m);
    std::unique_ptr<double[]> error_smart = std::make_unique<double[]>(n*m);

    nvtxRangePushA("init");
    initialize(A_smart.get(), Anew_smart.get(), m, n);
    nvtxRangePop();
 
    std::cout << "Jacobi relaxation Calculation: " << n << " x " << m << " mesh\n";
 
    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
    int iter = 0;
    double error = 1.0;

    auto cublas_deleter = [](cublasHandle_t* handle) {
        if (handle && *handle) {
            cublasDestroy(*handle);
            delete handle;
        }
    };

    cublasStatus_t stat;
    std::unique_ptr<cublasHandle_t, decltype(cublas_deleter)> cublasHandlePtr(new cublasHandle_t, cublas_deleter);
    cublasCreate(cublasHandlePtr.get());

    int idx_max;
    auto A = A_smart.get();
    auto Anew = Anew_smart.get();
    auto error_array = error_smart.get();

    nvtxRangePushA("while");
    #pragma acc data copy(A[0:n*m], Anew[0:n*m], error_array[0:n*m])
    { 
        while (fabs(error) > tol && iter < iter_max) {

            #pragma acc parallel loop collapse(2) present(A, Anew)
            for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < m - 1; ++j) {
                    Anew[OFFSET(i, j, m)] = 0.25 * (A[OFFSET(i, j+1, m)] + A[OFFSET(i, j-1, m)]
                                                    + A[OFFSET(i+1, j, m)] + A[OFFSET(i-1, j, m)]);
                    error_array[OFFSET(i, j, m)] = Anew[OFFSET(i, j, m)];
                }
            }

            if (iter % 1000 == 0){
                #pragma acc host_data use_device(A, Anew, error_array)
                {
                    double alpha = -1;
                    cublasDaxpy(*cublasHandlePtr, n*m, &alpha, A, 1, error_array, 1);
                }
                #pragma acc parallel loop present(error_array)
                for (int i = 0; i < n; ++i) {
                    error_array[OFFSET(i, 0, m)] = 0.0;
                    error_array[OFFSET(i, m-1, m)] = 0.0;
                }
                #pragma acc parallel loop present(error_array)
                for (int j = 0; j < m; ++j) {
                    error_array[OFFSET(0, j, m)] = 0.0;
                    error_array[OFFSET(n-1, j, m)] = 0.0;
                }
                #pragma acc host_data use_device(A, Anew, error_array)
                {
                    stat = cublasIdamax(*cublasHandlePtr, n*m, error_array, 1, &idx_max);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cout << "cublasIdamax failed\n";
                    }

                    if (idx_max > 0 && idx_max <= n * m) {
                        idx_max -= 1;
                        cudaMemcpy(&error, &error_array[idx_max], sizeof(double), cudaMemcpyDeviceToHost);
                    } else {
                        std::cerr << "Warning: cublasIdamax returned index " << (idx_max+1) << "\n";
                        error = 0.0;
                    }
                }

            }

            double* temp = A;
            A = Anew;
            Anew = temp;

            if(iter % 10000 == 0)
                std::cout << iter << ", error = " << error << "\n";

            iter++;
        }
    }
    nvtxRangePop();
 
    std::chrono::steady_clock::time_point fn = std::chrono::steady_clock::now();
    std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(fn - st);
    
    std::cout << "iterations: " << iter << "\n";
    std::cout << "error: " << error << "\n";
    std::cout << "time: " << runtime.count() << " seconds\n";
 
    return 0;
 }
