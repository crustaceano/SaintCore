cmake_minimum_required(VERSION 3.17)
project(SaintCore LANGUAGES CXX)

# Используем C++17, как требует LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Включаем ваши директории с заголовками
include_directories(.)
include_directories(include)
include_directories(include/perfect_visualization)

# 🔧 Основной исполняемый файл
add_executable(SaintCore
        src/functions.cpp
        src/model.cpp
        src/tensor.cpp
        src/container.cpp
        main_container.cpp
        visualization.cpp
        src/Image.cpp
        src/bmp_writer.cpp
        mnist_reader.cpp
)

# 🎯 GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(SaintCoreTests
        src/functions.cpp
        src/model.cpp
        src/tensor.cpp
        src/container.cpp
        tests/test_functions.cpp
        tests/test_tensor.cpp
        tests/test_model.cpp
        tests/test_arab_conbtainer.cpp
)
target_include_directories(SaintCoreTests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SaintCoreTests PRIVATE gtest_main)

enable_testing()
include(GoogleTest)
gtest_discover_tests(SaintCoreTests)
