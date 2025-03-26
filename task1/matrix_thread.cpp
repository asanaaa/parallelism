#include <iostream>
#include <fstream>
#include<vector>
#include <thread>
#include<chrono>


// Функция параллельной инициализации матрицы
void initializeMatrix(std::vector<double>& a, int m, int n, int low_bound, int up_bound) {
    for (int i = low_bound; i < up_bound; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
        }
    }
}

// Функция параллельной инициализации вектора
void initializeVector(std::vector<double>& b, int n) {
    for (int j = 0; j < n; j++) {
        b[j] = j;
    }
}

void matrix_vector_product_threads(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, int low_bound, int up_bound, int n) {
    for (int i = low_bound; i < up_bound; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }

}

int main() {
    int nthreads = 40;
    int m = 40000;
    int n = 40000;
    std::vector<double> a;//инициализация происходит сразу надо исправить
    a.reserve(m*n);
    std::vector<double> b;
    b.reserve(n);
    std::vector<double> c;
    c.reserve(m);

    std::vector<std::thread> threads;

    // Параллельная инициализация матрицы A
    for (int i = 0; i < nthreads; ++i) {
        int threadid = i;
        int items_per_thread = m / nthreads;
        int low_bound = threadid * items_per_thread;
        int up_bound = (threadid == nthreads - 1) ? (m - 1) : (low_bound + items_per_thread - 1);

        threads.emplace_back(initializeMatrix, std::ref(a), m, n, low_bound, up_bound);
    }

    // Запускаем отдельный поток для вектора B
    std::thread vectorThread(initializeVector, std::ref(b), n);

    // Ожидание завершения инициализации
    for (auto& thread : threads) {
        thread.join();
    }
    vectorThread.join();

    threads.clear();

    // Засекли время
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for(int i = 0; i < nthreads; ++i){
        int threadid = i;
        int items_per_thread = m / nthreads;
        int low_bound = threadid * items_per_thread;
        int up_bound = (threadid == nthreads - 1) ? (m - 1) : (low_bound + items_per_thread - 1);

        threads.emplace_back(matrix_vector_product_threads, std::ref(a), std::ref(b), std::ref(c), low_bound, up_bound, n);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> t = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::ofstream out;
    out.open("MyRes.txt", std::ios::app);
    out << "Time of " << nthreads << " threads: " << t.count() << "\n";
    out.close();

    return 0;
}
