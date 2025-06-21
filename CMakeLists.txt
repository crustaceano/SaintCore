cmake_minimum_required(VERSION 3.29)
project(SaintCore)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(include)

add_executable(SaintCore
        include/functions.h
        include/model.h
        include/tensor.h
        include/exceptions.h
        include/types.h
        src/functions.cpp
        src/model.cpp
        src/tensor.cpp
        main.cpp)
