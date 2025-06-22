#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <io.h>

using namespace std;

// Структура для хранения данных MNIST
struct MNIST_Example {
    vector<double> pixels;  // 784 пикселя (0-255)
    int label;              // метка класса (0-9)
};

// Функция для реверса байтов
uint32_t reverse_bytes(uint32_t v) {
    return ((v >> 24) & 0xff) | ((v >> 8) & 0xff00) |
        ((v << 8) & 0xff0000) | ((v << 24) & 0xff000000);
}

// Загрузка данных из распакованных файлов
vector<MNIST_Example> load_mnist(const string& images_path, const string& labels_path) {
    vector<MNIST_Example> dataset;

    // Открываем файлы изображений и меток
    ifstream images_file(images_path, ios::binary);
    if (!images_file) {
        cerr << "Error opening images file: " << images_path << endl;
        return dataset;
    }

    ifstream labels_file(labels_path, ios::binary);
    if (!labels_file) {
        cerr << "Error opening labels file: " << labels_path << endl;
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
        cerr << "Invalid MNIST files format!" << endl;
        return dataset;
    }

    dataset.resize(num_images);
    const size_t image_size = rows * cols;
    vector<unsigned char> buffer(image_size);

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


int main() {
    try {
        std::string file = "../mnist/train-images.idx3-ubyte";

        std::cout << "One" << std::endl;
        // Пути к распакованным файлам в папке mnist
        const string base_path = "../mnist/";

        // Загрузка тренировочного набора
        auto train_set = load_mnist(
            base_path + "train-images.idx3-ubyte",
            base_path + "train-labels.idx1-ubyte"
        );
        cout << "Loaded " << train_set.size() << " training examples" << endl;

        // Загрузка тестового набора
        auto test_set = load_mnist(
            base_path + "t10k-images.idx3-ubyte",
            base_path + "t10k-labels.idx1-ubyte"
        );
        cout << "Loaded " << test_set.size() << " test examples" << endl;

        // Пример доступа к данным
        if (!train_set.empty()) {
            cout << "\nFirst training example:" << endl;
            cout << "Label: " << train_set[0].label << endl;
            cout << "Pixel values (first 1000): ";
            for (int i = 0; i < 1000; ++i) {
                cout << static_cast<int>(train_set[0].pixels[i]) << " ";
            }
            cout << endl;
        }
    }
    catch (exception& err) {
        std::cout << err.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unknown error" << std::endl;
    }
    return 0;
}