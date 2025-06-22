#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct MNIST_Example {
    std::vector<double> pixels;  // 784 пикселя (0-255)
    int label;                   // метка класса (0-9)
};

// Функция для реверса байтов
uint32_t reverse_bytes(uint32_t v);

// Загрузка данных из распакованных файлов
std::vector<MNIST_Example> load_mnist(const std::string& images_path, const std::string& labels_path);
