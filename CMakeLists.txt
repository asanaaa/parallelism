cmake_minimum_required(VERSION 3.22.1)
project(lab1)
add_executable(${PROJECT_NAME} laba1.cpp)
option(DOUBLE "type" OFF)

if(DOUBLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NUM="double")

# else()
#     target_compile_definitions(laba1 PUBLIC NUM="float")
endif()
