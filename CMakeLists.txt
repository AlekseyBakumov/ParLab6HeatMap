cmake_minimum_required(VERSION 3.22)

project(Example_class VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "pgc++")
set(NAME "laplace")

message(STATUS "Compile C++: " ${CMAKE_CXX_COMPILER})

option(CUBLAS "Using cuBLAS" OFF)
set(ACCTYPE "HOST" CACHE STRING "Type of accelerator: HOST, MULTICORE, GPU")

find_package(Boost  REQUIRED COMPONENTS program_options)

if(ACCTYPE STREQUAL "HOST")
    message(STATUS "Build ACCTYPE=HOST")
    list(APPEND option_compile -acc=host -Minfo=all)
    list(APPEND option_link -acc=host)
elseif(ACCTYPE STREQUAL "MULTICORE")
    message(STATUS "Build ACCTYPE=MULTICORE")
    list(APPEND option_compile -acc=multicore -Minfo=all)
    list(APPEND option_link -acc=multicore)
elseif(ACCTYPE STREQUAL "GPU")
    message(STATUS "Build ACCTYPE=GPU")
    list(APPEND option_compile -acc=gpu -Minfo=all)
    list(APPEND option_link -acc=gpu)
endif()

add_executable(${NAME} "jacobi.cpp") 
target_compile_features(${NAME} PRIVATE cxx_std_20)
# Boost
target_link_libraries(${NAME} PRIVATE Boost::program_options)
# openACC
target_compile_options(${NAME} PRIVATE ${option_compile}) 
target_link_options(${NAME} PRIVATE ${option_link})
# CUDA and cuBLAS
if(CUBLAS AND ACCTYPE STREQUAL "GPU")
    message(STATUS "Build cuBLAS version")
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(${NAME} PRIVATE CUBLAS)
    target_include_directories(${NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${NAME} PRIVATE CUDA::cublas)
endif()