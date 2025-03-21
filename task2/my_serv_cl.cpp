#include <iostream>
#include <fstream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <optional>
#include <experimental/random>

#define myType double

std::mutex cout_mutex; // Мьютекс для синхронизации вывода

template<typename T>
T fun_sin(T arg) {
    return std::sin(arg);
}

template<typename T>
T fun_sqrt(T arg) {
    return std::sqrt(arg);
}

template<typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}


template <typename T>
class Server {
public:
    Server() {}

    ~Server() {
        stop();
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            results.clear(); // Очищаем все futures
        }
    }

    void start() {
        stop_flag = false;
        server_thread = std::jthread(&Server::server_loop, this, stop_src.get_token());
    }

    void stop() {
        stop_flag = true;
        cond_var.notify_one();
        if (server_thread.joinable()) {
            stop_src.request_stop();
            server_thread.join();
        }
    }

    size_t add_task(std::function<double()> task) {
        size_t task_id = task_counter++;
        std::packaged_task<T()> packaged_task(std::move(task));
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace(task_id, std::move(packaged_task)); // Добавляем задачу в очередь
        }
        cond_var.notify_one(); // Уведомляем поток сервера о новой задаче
        // std::cout << "Add\n";
        return task_id;
    }

    bool is_task_completed(size_t task_id) {
        std::lock_guard<std::mutex> lock(result_mutex);
        auto it = results.find(task_id);
        return it != results.end() && it->second.has_value();
    }

    double request_result(size_t task_id) {
        std::unique_lock<std::mutex> lock(result_mutex);
        auto it = results.find(task_id);
        if (it == results.end()) {
            throw std::runtime_error("Task ID not found!");
        }
        if (!it->second.has_value()) {
            throw std::runtime_error("Task result is not ready!");
        }
        return it->second.value();
    }

private:
    std::jthread server_thread;
    std::queue<std::pair<size_t, std::packaged_task<T()>>> tasks; // Изменен тип задач
    std::unordered_map<size_t, std::optional<T>> results; // Изменен тип результатов

    std::mutex queue_mutex, result_mutex;
    std::stop_source stop_src;
    std::condition_variable cond_var;
    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> task_counter{0};

    void server_loop(std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            std::pair<size_t, std::packaged_task<T()>> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cond_var.wait(lock, [this] { return !tasks.empty() || stop_flag; });

                if (stop_flag) break;

                if (!tasks.empty()) {
                    task = std::move(tasks.front());
                    tasks.pop();
                }
            }
            // std::cout << "Pop\n";

            // Проверяем валидность packaged_task
            if (!task.second.valid()) {
                std::cerr << "Invalid packaged_task!\n";
                continue;
            }

            // Создаем future перед выполнением задачи
            std::future<T> result = task.second.get_future();
            // Выполнение задачи
            try {
                task.second(); // Выполняем задачу
            } catch (const std::exception& e) {
                std::cerr << "Task execution failed: " << e.what() << '\n';
                continue;
            }
            // std::cout << "Done\n";

            // Сохраняем результат в карте
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                results[task.first] = result.get(); // Сохраняем результат
            }
        }
        std::cout << "Server stopped.\n";
    }
};

// Поток, который добавляет задачи в очередь
void add_task1_thread(Server<myType>& server, std::ofstream& out) {

    for(int i = 0; i < 1000; ++i)
    {    
        double arg = std::experimental::randint(0, 100);
        size_t task_id = server.add_task(std::bind(fun_sin<myType>, arg));

        // Ожидаем завершения задачи
        while (!server.is_task_completed(task_id)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            out << "sin " << arg << " = " << server.request_result(task_id) << '\n';
        }
    }
}

void add_task2_thread(Server<myType>& server, std::ofstream& out) {

    for(int i = 0; i < 1000; ++i)
    {    
        double arg = std::experimental::randint(0, 100);
        size_t task_id = server.add_task(std::bind(fun_sqrt<myType>, arg));

        // Ожидаем завершения задачи
        while (!server.is_task_completed(task_id)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            out << "sqrt " << arg << " = " << server.request_result(task_id) << '\n';
        }
    }
}

void add_task3_thread(Server<myType>& server, std::ofstream& out) {

    for(int i = 0; i < 1000; ++i)
    {    
        double arg1 = std::experimental::randint(0, 100);
        double arg2 = std::experimental::randint(0, 20);
        size_t task_id = server.add_task(std::bind(fun_pow<myType>, arg1, arg2));

        // Ожидаем завершения задачи
        while (!server.is_task_completed(task_id)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            out << "pow " << arg1 << " " <<  arg2 << " = " << server.request_result(task_id) << '\n';
        }
    }
}

int main() {
    std::cout << "Start\n";

    Server<myType> server;
    server.start(); // Запуск потока сервера

    std::ofstream out;
    out.open("Results.txt");
    std::thread add_task1(add_task1_thread, std::ref(server), std::ref(out));
    std::thread add_task2(add_task2_thread, std::ref(server), std::ref(out));
    std::thread add_task3(add_task3_thread, std::ref(server), std::ref(out));

    add_task1.join();
    add_task2.join();
    add_task3.join();
    server.stop();
    out.close();
    std::cout << "End\n";
}