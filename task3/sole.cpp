#include<iostream>
#include<fstream>
#include<vector>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include <omp.h>

double norm(std::vector<double> v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

void solve_serial(std::vector<double> a, std::vector<double> b, int n, int iterations, double eps, const double r) {
    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);

    double t = omp_get_wtime();

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
        // if (norm_b == 0 || norm_res == 0) {
        //     std::cerr << "Ошибка: деление на ноль в вычислении ошибки!" << std::endl;
        //     return;
        // }
        double error = norm_res / norm_b;
        
        if (error < eps) {
            t = omp_get_wtime() - t;
            // std::cout << error << std::endl;
            // std::cout << eps << std::endl;
            // std::cout << norm_res << std::endl;
            // std::cout << norm_b << std::endl;
            std::ofstream out;
            out.open("MyRes.txt", std::ios::app);
            out << "Time of 1 thread: " << t << "\n";
            out << "Error is " << 1-x_new[0] << "\n";
            out.close();
            std::cout << x_new[0] << std::endl;
            std::cout << "Solution take " << t << " seconds.\n";
            break;
        }

        x_cur.swap(x_new);
    }
}

double solve_parallel_1(std::vector<double> a, std::vector<double> b, int n, int iterations, double eps, const double r, int nthreads) {
    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);
    omp_set_num_threads(nthreads);
    double t = omp_get_wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> Ax(n, 0.0);
        std::vector<double> residual(n);

        // Распараллеливаем вычисление Ax
        #pragma omp parallel for schedule(auto)
        for (int i = 0; i < n; ++i) {
            double temp = 0.0;
            for (int j = 0; j < n; ++j) {
                temp += a[i * n + j] * x_cur[j];
            }
            Ax[i] = temp;
        }

        // Распараллеливаем вычисление residual и обновление x_new
        #pragma omp parallel for schedule(auto)
        for (int i = 0; i < n; ++i) {
            residual[i] = Ax[i] - b[i];
            x_new[i] = x_cur[i] - (r * residual[i]);

            // if (std::isnan(x_new[i])) {
            //     std::cout << "nan detected at iteration " << iter << std::endl;
            //     return;
            // }
        }

        if (iter % 1000 == 0)
            std::cout << "Complete " << iter << " iteration.\n";

        if (iter % 10000 == 0)
            std::cout << x_new[0] << std::endl;

        double norm_b = norm(b, n);
        double norm_res = norm(residual, n);

        // if (norm_b == 0 || norm_res == 0) {
        //     std::cerr << "Error: div by zero!" << std::endl;
        //     return;
        // }

        double error = norm_res / norm_b;

        if (error < eps) {
            t = omp_get_wtime() - t;
            // std::ofstream out;
            // out.open("MyRes.txt", std::ios::app);
            // out << "Time of " << nthreads << " threads: " << t << "\n";
            // out << "Error is " << fabs(1-x_new[0]) << "\n";
            // out.close();
            // std::cout << x_new[0] << std::endl;
            // std::cout << "Solution take " << t << " seconds.\n";
            break;
        }

        x_cur.swap(x_new);
    }
    return t;
}


double solve_parallel_2(std::vector<double> a, std::vector<double> b, int n, int iterations, double eps, const double r, int nthreads) {
    std::vector<double> x_cur(n, 0.0);
    std::vector<double> x_new(n);

    double t = omp_get_wtime();
    bool done = 0;
    omp_set_num_threads(nthreads);
    double error;
    
    while(!done) {
        double sumAx = 0.0;
        double sumb = 0.0;
        std::vector<double> Ax(n, 0.0);
        std::vector<double> residual(n);
        #pragma omp parallel
        {
            int threadid = omp_get_thread_num();
            int items_per_thread = n /nthreads;
            int low_bound = threadid * items_per_thread;
            int up_bound = (threadid == nthreads - 1) ? (n -1) : (low_bound + items_per_thread - 1);
            
            // Распараллеливаем вычисление Ax
            
            for (int i = low_bound; i < up_bound; ++i) {
                double temp = 0.0;
                for (int j = 0; j < n; ++j) {
                    temp += a[i * n + j] * x_cur[j];
                }
                #pragma omp atomic write
                    Ax[i] = temp;
            }
            // Распараллеливаем вычисление residual и обновление x_new
            
            for (int i = low_bound; i < up_bound; ++i) {
                #pragma omp critical
                {
                    residual[i] = Ax[i] - b[i];
                    x_new[i] = x_cur[i] - (r * residual[i]);
    
                    sumAx += residual[i] * residual[i];
                    sumb += b[i] * b[i];
                }
            }

            // if (iter % 1000 == 0 && omp_get_thread_num() == 0)
            //     std::cout << "Complete " << iter << " iteration.\n";

            // if (iter % 10000 == 0 && omp_get_thread_num() == 0)
            //     std::cout << x_new[0] << std::endl;
            #pragma omp barrier
            #pragma omp single
            {
                sumAx = sqrt(sumAx);
                sumb = sqrt(sumb);
                error = sumAx / sumb;
                if(error < eps){
                    t = omp_get_wtime() - t;
                    // std::cout << error << std::endl;
                    // std::cout << eps << std::endl;
                    // std::cout << norm_res << std::endl;
                    // std::cout << norm_b << std::endl;

                    // std::ofstream out;
                    // out.open("MyRes.txt", std::ios::app);
                    // out << "Time of " << nthreads << " threads: " << t << "\n";
                    // out << "Error is " << error << "\n";
                    // out.close();

                    // for(int i =0;i<10;++i)
                    //     std::cout << x_new[i] << std::endl;
                    std::cout << x_new[0] << std::endl;
                    std::cout << "Solution take " << t << " seconds.\n";
                    done = 1;
                }
            }
            
            #pragma omp single
            {
                x_cur.swap(x_new);
            }
        }
    }
    return t;
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
    // solve_serial(a, b, n, 10000000, 1e-05, 0.001);
    double t[10] = {0.0};

    for(int i = 0; i < 10; ++i){
        t[0] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 80);
        t[1] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 60);
        t[2] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 40);
        t[3] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 30);
        t[4] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 20);
        t[5] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 16);
        t[6] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 8);
        t[7] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 7);
        t[8] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 4);
        t[9] += solve_parallel_2(a, b, n, 12000, 1e-05, 0.001, 2);
    }
    std::ofstream out;
    out.open("MyRes.txt", std::ios::app);
    for(int i = 0; i< 10; ++i){
        out << "Time of " << i << " threads: " << t[i]/10 << "\n";
    }
    out.close();

    return 0;
}
