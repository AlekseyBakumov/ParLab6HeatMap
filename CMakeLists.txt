cmake_minimum_required(VERSION 3.16)
project(jacobi_openacc LANGUAGES CXX)

# OpenACC backend selector
set(PAR_MODE "gpu" CACHE STRING "OpenACC target backend (host, multicore, gpu)")
set_property(CACHE PAR_MODE PROPERTY STRINGS host multicore gpu)

# Use NVIDIA HPC compiler
if (NOT CMAKE_CXX_COMPILER MATCHES "nvc\\+\\+")
    message(FATAL_ERROR "You must use NVIDIA HPC compiler (nvc++) for OpenACC")
endif()

# Enable OpenACC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minfo=accel -fast -lm")

# Select backend
if (PAR_MODE STREQUAL "gpu")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=gpu")
elseif (PAR_MODE STREQUAL "multicore")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=multicore")
elseif (PAR_MODE STREQUAL "host")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=host")
elseif (PAR_MODE STREQUAL "nopar")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=noautopar")
else()
    message(FATAL_ERROR "Unknown PAR_MODE value: ${PAR_MODE}")
endif()




add_executable(laplace
                jacobi.cpp
                #laplace2d.cpp
                )


#target_include_directories(laplace PRIVATE ${Boost_INCLUDE_DIRS})
#target_link_libraries(laplace PRIVATE ${Boost_LIBRARIES})
