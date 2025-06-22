cmake_minimum_required(VERSION 3.29)
project(SaintCore)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(include)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Основной исполняемый файл
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

add_executable(
        SaintCoreTests
        include/functions.h
        include/model.h
        include/tensor.h
        include/exceptions.h
        include/types.h
        src/functions.cpp
        src/model.cpp
        src/tensor.cpp
)

target_include_directories(SaintCoreTests PRIVATE include)

# 🔥 ВАЖНО: линковка с Google Test
target_link_libraries(SaintCoreTests gtest_main)

# Настройка запуска тестов
enable_testing()
include(GoogleTest)
gtest_discover_tests(SaintCoreTests)
