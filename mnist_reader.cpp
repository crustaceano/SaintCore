#include "mnist_reader.h"
#include <fstream>
#include <iostream>

// Функция для реверса байтов
uint32_t reverse_bytes(uint32_t v) {
    return ((v >> 24) & 0xff) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | ((v << 24) & 0xff000000);
}

// Загрузка данных из распакованных файлов
std::vector<MNIST_Example> load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::vector<MNIST_Example> dataset;

    // Открываем файлы изображений и меток
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file) {
        std::cerr << "Error opening images file: " << images_path << std::endl;
        return dataset;
    }

    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file) {
        std::cerr << "Error opening labels file: " << labels_path << std::endl;
        return dataset;
    }

    // Читаем заголовки изображений
    uint32_t magic, num_images, rows, cols;
    images_file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    images_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    images_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    images_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Конвертируем big-endian в little-endian
    magic = reverse_bytes(magic);
    num_images = reverse_bytes(num_images);
    rows = reverse_bytes(rows);
    cols = reverse_bytes(cols);

    // Читаем заголовки меток
    uint32_t labels_magic, num_labels;
    labels_file.read(reinterpret_cast<char*>(&labels_magic), sizeof(labels_magic));
    labels_file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    labels_magic = reverse_bytes(labels_magic);
    num_labels = reverse_bytes(num_labels);

    // Проверка согласованности данных
    if (magic != 2051 || labels_magic != 2049 || num_images != num_labels) {
        std::cerr << "Invalid MNIST files format!" << std::endl;
        return dataset;
    }

    dataset.resize(num_images);
    const size_t image_size = rows * cols;
    std::vector<unsigned char> buffer(image_size);

    // Чтение данных
    for (uint32_t i = 0; i < num_images; ++i) {
        // Читаем изображение
        images_file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        dataset[i].pixels.resize(image_size);
        for (size_t j = 0; j < image_size; ++j) {
            dataset[i].pixels[j] = static_cast<double>(buffer[j]);
        }

        // Читаем метку
        unsigned char label;
        labels_file.read(reinterpret_cast<char*>(&label), sizeof(label));
        dataset[i].label = static_cast<int>(label);
    }

    return dataset;
}