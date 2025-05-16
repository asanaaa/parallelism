#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <boost/program_options.hpp>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

#define OFFSET(x, y, m) (((x)*(m)) + (y))


namespace po = boost::program_options;
// cuda unique_ptr
template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

// new
template <typename T>
T *cuda_new(std::size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

// delete
template <typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}

__global__ void grid_kernel(double* A, double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < size-1 && j > 0 && j < size-1) {
        Anew[OFFSET(i, j, size)] = 0.25 * (A[OFFSET(i, j+1, size)] + A[OFFSET(i, j-1, size)]
                                                    + A[OFFSET(i+1, j, size)] + A[OFFSET(i-1, j, size)]);
    }
}

__global__ void error_kernel(double* A, double* Anew, double* block_max_errors, int size) {
    double inner_error = 0.0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int idx = thread_id; idx < size * size; idx += total_threads) {
        int i = idx / size;
        int j = idx % size;
        if (i > 0 && i < size-1 && j > 0 && j < size-1) {
            inner_error = fmax(inner_error, fabs(Anew[OFFSET(i, j, size)] - A[OFFSET(i, j, size)]));
        }
    }

    double block_max = BlockReduce(temp_storage).Reduce(inner_error, cub::Max());
    if (threadIdx.x == 0) 
        block_max_errors[blockIdx.x] = block_max;
}

// __global__ void copy_kernel(double* A, const double* Anew, int size) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i >= 1 && i < size - 1 && j >= 1 && j < size - 1) {
//         A[i * size + j] = Anew[i * size + j];
//     }
// }

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

 int main(int argc, char** argv) {
    int n, m, iter_max;
    double tol;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("size,n", po::value<int>(&n)->default_value(1024), "size (n x n)")
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
    double error = 1.0;
    int iteration = 0;

    std::unique_ptr<double[]> A = std::make_unique<double[]>(n*m);
    std::unique_ptr<double[]> Anew = std::make_unique<double[]>(n*m);

    // std::unique_ptr<double[]> errort = std::make_unique<double[]>(n*m);

    initialize(A.get(), Anew.get(), m, n);

    // double* device_A, *device_Anew;
    // cudaMalloc(&device_A, n*m*sizeof(double));
    // cudaMalloc(&device_Anew, n*m*sizeof(double));

    double *device_A = cuda_new<double>(n*m);
    cuda_unique_ptr<double> d_matA(device_A,
                                  cuda_delete<double>);

    double *device_Anew = cuda_new<double>(n*m);
    cuda_unique_ptr<double> d_matB(device_Anew,
                                  cuda_delete<double>);

    cudaMemcpy(device_A, A.get(), n*m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Anew, Anew.get(), n*m*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((m + threads - 1) / threads, (m + threads - 1) / threads);

    int num_blocks = 1024;
    double* d_block_max;
    cudaMalloc(&d_block_max, sizeof(double) * num_blocks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // cudaGraph_t graph;
    // cudaGraphExec_t instance;
    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // for (int i = 0; i < 1000; i++) {
    //     grid_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, n);
    //     copy_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, n);
    // }
    // grid_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, n);
    // cudaStreamEndCapture(stream, &graph);
    // cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    std::unique_ptr<double[]> h_block_max = std::make_unique<double[]>(num_blocks);

    std::cout << "Jacobi relaxation Calculation: " << n << " x " << m << " mesh\n";

    const auto start{std::chrono::steady_clock::now()};

    while (error > tol && iteration < iter_max) {
        // cudaGraphLaunch(instance, stream);
        // cudaStreamSynchronize(stream);
        for(int i = 0; i < 1000; i++){
            grid_kernel<<<grid, block, 0, stream>>>(device_A, device_Anew, n);
            double* temp = device_A;
            device_A = device_Anew;
            device_Anew = temp;
        }
        error_kernel<<<num_blocks, 256, 0, stream>>>(device_A, device_Anew, d_block_max, n);

        cudaMemcpy(h_block_max.get(), d_block_max, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < num_blocks; ++i) 
            if (h_block_max[i] > error)
                error = h_block_max[i];

        iteration+=1000;
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "iterations: " << iteration << "\n";
    std::cout << "error: " << error << "\n";
    std::cout << "time: " << elapsed_seconds.count() << " seconds\n";

    cudaFree(device_A);
    cudaFree(device_Anew);
    cudaFree(d_block_max);
    // cudaGraphDestroy(graph);
    // cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    return 0;
}
