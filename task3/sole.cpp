#include<iostream>
#include<fstream>
#include<vector>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<chrono>
#include <omp.h>

using namespace std::chrono;

double norm(std::vector<double> v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

void solve_serial(std::vector<double> a, std::vector<double> b, int n, double eps, const double r, int iterations=12000) {
    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);

    steady_clock::time_point t1 = steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> Ax(n, 0.0);
        std::vector<double> residual(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Ax[i] += a[i*n + j] * x_cur[j];
            }
            residual[i] = Ax[i] - b[i];
            x_new[i] = x_cur[i] - (r*residual[i]);

            if(std::isnan(x_new[i])){
                std::cout << x_cur[i]<<std::endl;
                return;
            }
        }
        
        if(iter % 1000 == 0)
            std::cout << "Complete " << iter << " iteration.\n";

        if(iter % 10000 == 0){
            std::cout << x_new[0] << std::endl;
        }

        double norm_b = norm(b, n);

        double norm_res = norm(residual, n);
        double error = norm_res / norm_b;
        
        if (error < eps) {
            steady_clock::time_point t2 = steady_clock::now();
            duration<double> t = duration_cast<duration<double>>(t2 - t1);
            std::cout << x_new[0] << std::endl;
            std::cout << "Solution take " << t.count() << " seconds.\n";
            break;
        }

        x_cur.swap(x_new);
    }
}

double solve_parallel_1(std::vector<double> a, std::vector<double> b, int n, double eps, const double r, int nthreads, int iterations=12000) {
    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);
    omp_set_num_threads(nthreads);
    steady_clock::time_point t1 = steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> Ax(n, 0.0);
        std::vector<double> residual(n);

        // Распараллеливаем вычисление Ax
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double temp = 0.0;
            for (int j = 0; j < n; ++j) {
                temp += a[i * n + j] * x_cur[j];
            }
            Ax[i] = temp;
        }

        // Распараллеливаем вычисление residual и обновление x_new
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            residual[i] = Ax[i] - b[i];
            x_new[i] = x_cur[i] - (r * residual[i]);
        }

        if (iter % 1000 == 0)
            std::cout << "Complete " << iter << " iteration.\n";

        if (iter % 10000 == 0)
            std::cout << x_new[0] << std::endl;

        double norm_b = norm(b, n);
        double norm_res = norm(residual, n);

        double error = norm_res / norm_b;

        if (error < eps) {
            steady_clock::time_point t2 = steady_clock::now();
            duration<double> t = duration_cast<duration<double>>(t2 - t1);
            return t.count();
            break;
        }

        x_cur.swap(x_new);
    }
}


double solve_parallel_2(std::vector<double> a, std::vector<double> b, int n, double eps, const double r, int nthreads, int iterations=12000) {

    omp_set_num_threads(nthreads);
    double error;

    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);

    steady_clock::time_point t1 = steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> Ax(n, 0.0);
        std::vector<double> residual(n);

        // Распараллеливаем вычисление Ax
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double temp = 0.0;
            for (int j = 0; j < n; ++j) {
                temp += a[i * n + j] * x_cur[j];
            }
            Ax[i] = temp;
            residual[i] = temp - b[i];
            x_new[i] = x_cur[i] - (r * residual[i]);
        }

        double norm_b = norm(b, n);
        double norm_res = norm(residual, n);

        double error = norm_res / norm_b;

        if (error < eps) {
            t = omp_get_wtime() - t;
            std::cout << x_new[0] << std::endl;
            std::cout << "Solution take " << t << " seconds.\n";
            steady_clock::time_point t2 = steady_clock::now();
            duration<double> t = duration_cast<duration<double>>(t2 - t1);
            return t.count();
            break;
        }
        x_cur.swap(x_new);
    }
}

int main() {
    int n = 1998;

    std::vector<double> a(n*n);
    std::vector<double> b(n, n+1);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i==j) 
                a[i * n + j] = 2.0;
            else
                a[i * n + j] = 1.0;
        }
    }
    int nthreads = omp_get_max_threads();
    std::cout << nthreads << "\n"; 
    std::cout << "Starting to compute...\n";
    // solve_serial(a, b, n, 1e-05, 0.001);
    double t[10] = {0.0};

    for(int i = 0; i < 5; ++i){
        t[0] += solve_parallel_2(a, b, n, 1e-05, 0.001, 80);
        t[1] += solve_parallel_2(a, b, n, 1e-05, 0.001, 60);
        t[2] += solve_parallel_2(a, b, n, 1e-05, 0.001, 40);
        t[3] += solve_parallel_2(a, b, n, 1e-05, 0.001, 30);
        t[4] += solve_parallel_2(a, b, n, 1e-05, 0.001, 20);
        t[5] += solve_parallel_2(a, b, n, 1e-05, 0.001, 16);
        t[6] += solve_parallel_2(a, b, n, 1e-05, 0.001, 8);
        t[7] += solve_parallel_2(a, b, n, 1e-05, 0.001, 7);
        t[8] += solve_parallel_2(a, b, n, 1e-05, 0.001, 4);
        t[9] += solve_parallel_2(a, b, n, 1e-05, 0.001, 2);
    }
    std::ofstream out;
    out.open("MyRes.txt", std::ios::app);
    for(int i = 0; i< 10; ++i){
        out << "Time of " << i << " threads: " << t[i]/5 << "\n";
    }
    out.close();

    return 0;
}
