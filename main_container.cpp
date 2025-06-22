#include "mnist_reader.h"
#include "visualization.h"

// Путь к распакованным файлам в папке mnist
const std::string BASE_PATH = "../mnist/";

std::vector<MNIST_Example> load_dataset(const std::string& images_path, const std::string& labels_path) {
    std::vector<MNIST_Example> dataset = load_mnist(BASE_PATH + images_path, BASE_PATH + labels_path);
    std::cout << "Loaded " << dataset.size() << " samples" << std::endl;
    return dataset;
}

int main() {
    try {
        // Загрузка тренировочного и тестового наборов
        std::vector<MNIST_Example> train_set = load_dataset("train-images.idx3-ubyte",
                                                        "train-labels.idx1-ubyte");
        std::vector<MNIST_Example> test_set = load_dataset("t10k-images.idx3-ubyte",
                                                        "t10k-labels.idx1-ubyte");

        // Пример доступа к данным
        if (!train_set.empty()) {
            std::cout << "\n2000`s training example:" << std::endl;
            std::cout << "Label: " << train_set[2000].label << std::endl;

            visualize(train_set[2000].pixels, "../visualized_data/output0.bmp");
            std::cout << std::endl;
        }
    }
    catch (std::exception& err) {
        std::cout << err.what() << std::endl;
    }
    return 0;
}