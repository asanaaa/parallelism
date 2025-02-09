#include<iostream>
#include<vector>
#include<cmath>

#ifdef NUM
#define MyType double
#else
#define MyType float
#endif

int main(){
    MyType elem = (2 * M_PI) / MyType(1e7);

    std::vector<MyType> numbers(1e7);

    MyType s = 0;
    for(MyType i = 0; i < 1e7; i++){
        numbers[i] = std::sin(elem * i);
        s += numbers[i];
    }

    std::cout << s << "\n";
    //std::cout << typeid(s).name();
}