#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <stdexcept>

#define LOG_EN false

struct Result {
    std::string operation;
    double arg1;
    double arg2;
    double result;
};

Result parse_result(const std::string& line) {
    Result res;
    std::istringstream fun(line);
    std::string op;
    char c;

    fun >> op;

    if (op == "sin" || op == "sqrt") {
        fun >> res.arg1 >> c >> res.result;
        res.operation = op;
    } 
    else if (op == "pow") {
        fun >> res.arg1 >> res.arg2 >> c >> res.result;
        res.operation = op;
    } 
    else {
        throw std::runtime_error("Unknown operation: " + op);
    }
    return res;
}

bool check_result(const Result& res) {
    double expected_result = 0.0;

    if (res.operation == "sin") {
        expected_result = std::sin(res.arg1);
    } 
    else if (res.operation == "sqrt") {
        expected_result = std::sqrt(res.arg1);
    } 
    else if (res.operation == "pow") {
        expected_result = std::pow(res.arg1, res.arg2);
    }

    if(LOG_EN){
        std::cout << "Expected: " << expected_result << std::endl;
        std::cout << "Result: " << res.result << std::endl;
        std::cout << "Dif: " << std::fabs(expected_result - res.result) << std::endl;
    }

    const double epsilon = 1e-5;
    double max_value = std::max(std::fabs(expected_result), std::fabs(res.result));
    if (max_value == 0) {
        return true;
    }
    double error;
    if(max_value >= 1)
        error = std::fabs(expected_result - res.result)/max_value;
    else
        error = std::fabs(expected_result - res.result);

    return error <= epsilon;
}


void test_results(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    int line_number = 0;
    bool all_passed = true;

    while (std::getline(file, line)) {
        ++line_number;
        Result res = parse_result(line);
        if (!check_result(res)) {
            std::cerr << "Test failed at line " << line_number << ": " << line << '\n';
            all_passed = false;
        }
        if(LOG_EN)
            std::cout << "======\n";
    }

    if (all_passed) {
        std::cout << line_number << " tests passed.\n";
    } else {
        std::cerr << "Fail.\n";
    }
}

int main() {
    test_results("Results.txt");
    return 0;
}