cmake_minimum_required(VERSION 3.22)

project(jacobi VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_COMPILER pgc++)
message("-!-Compile C: " ${CMAKE_CXX_COMPILER})

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

list(APPEND option_compile -acc=gpu -Minfo=all)
list(APPEND option_link -acc=gpu)

# Добавляем исходники
set(SRC   
    jacobi.cpp
)

# Создаём папку для бинарников
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

add_executable(${PROJECT_NAME} ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
# openACC
target_compile_options(${PROJECT_NAME} PRIVATE ${option_compile}) 
target_link_options(${PROJECT_NAME} PRIVATE ${option_link})
# cuda and nvtx
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvToolsExt)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::program_options)

