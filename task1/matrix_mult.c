#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

double matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int nthreads) {
    double t;
    #pragma omp parallel num_threads(nthreads)
    {
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int low_bound = threadid * items_per_thread;
        int up_bound = (threadid == nthreads - 1) ? (m - 1) : (low_bound + items_per_thread - 1);

        for (int i = low_bound; i < up_bound; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
        }
        #pragma omp single
        {
            for (int j = 0; j < n; j++) {
                b[j] = j;
            }
        }
        t = omp_get_wtime();
        for (int i = low_bound; i <= up_bound; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++) {
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
    t = omp_get_wtime() - t;
    return t;
}



double run_serial(int m, int n) {
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; j++) {
        b[j] = j;
    }
    
    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    t = omp_get_wtime() - t;
    
    free(a);
    free(b);
    free(c);
    return t;
}

double run_parallel(int m, int n, int nthreads) {
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);
    
    // double t = omp_get_wtime();
    double t = matrix_vector_product_omp(a, b, c, m, n, nthreads);
    // t = omp_get_wtime() - t;
    
    free(a);
    free(b);
    free(c);
    return t;
}

int main() {
    double t[16] = {0.0};
    FILE *output_time = fopen("output.txt", "a");
    
    for (int i = 0; i < 15; ++i) {
        t[0] += run_serial(20000, 20000);
        t[1] += run_parallel(20000, 20000, 2);
        t[2] += run_parallel(20000, 20000, 4);
        t[3] += run_parallel(20000, 20000, 7);
        t[4] += run_parallel(20000, 20000, 8);
        t[5] += run_parallel(20000, 20000, 16);
        t[6] += run_parallel(20000, 20000, 20);
        t[7] += run_parallel(20000, 20000, 40);

        t[8] += run_serial(40000, 40000);
        t[9] += run_parallel(40000, 40000, 2);
        t[10] += run_parallel(40000, 40000, 4);
        t[11] += run_parallel(40000, 40000, 7);
        t[12] += run_parallel(40000, 40000, 8);
        t[13] += run_parallel(40000, 40000, 16);
        t[14] += run_parallel(40000, 40000, 20);
        t[15] += run_parallel(40000, 40000, 40);
    }
    
    for (int i = 0; i < 16; ++i) {
        fprintf(output_time, "%lf\n", t[i]/15);
    }

    fclose(output_time);
    return 0;
}
