#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


double func(double x) {
    return exp(-x * x);
}

double run_serial() {
    const double a = -4.0;
    const double b = 4.0;
    const int n = 40000000;

    double h = (b - a) / n;
    double s = 0.0;
    double t = omp_get_wtime();
    
    for (int i = 0; i < n; i++)
        s += func(a + h * (i + 0.5));
    
    s *= h;
    t = omp_get_wtime() - t;
    return t;
}

double run_parallel(int nthreads) {
    const double a = -4.0;
    const double b = 4.0;
    const int n = 40000000;

    double h = (b - a) / n;
    double s = 0.0;
    double t = omp_get_wtime();
    
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        double sloc = 0.0;
        
        for (int i = tid; i < n; i += total_threads)
            sloc += func(a + h * (i + 0.5));
        
        #pragma omp atomic
        s += sloc;
    }
    
    s *= h;
    t = omp_get_wtime() - t;
    return t;
}

int main() {
    double t[8] = {0.0};
    FILE *output_time = fopen("output_2.txt", "a");

    for (int i = 0; i < 15; ++i) {
        t[0] += run_serial();
        t[1] += run_parallel(2);
        t[2] += run_parallel(4);
        t[3] += run_parallel(7);
        t[4] += run_parallel(8);
        t[5] += run_parallel(16);
        t[6] += run_parallel(20);
        t[7] += run_parallel(40);
    }
    
    for (int i = 0; i < 8; ++i) {
        fprintf(output_time, "%lf\n", t[i] / 15);
    }
    
    fclose(output_time);
    return 0;
}
