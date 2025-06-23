#include <include/container.h>
#include <include/tensor.h>
#include "mnist_reader.h"
#include "visualization.h"
#include <random>

// Путь к распакованным файлам в папке mnist
const std::string BASE_PATH = "../mnist/";

std::vector<MNIST_Example> load_dataset(const std::string& images_path, const std::string& labels_path) {
    std::string img_full = BASE_PATH + images_path;
    std::string lbl_full = BASE_PATH + labels_path;
    std::cout << "Loading MNIST images from: " << img_full << "\n";
    std::cout << "Loading MNIST labels from: " << lbl_full << "\n";
    auto data = load_mnist(img_full, lbl_full);
    std::cout << "Loaded " << data.size() << " samples\n";
    return data;
}

SaintCore::Tensor transform_data(std::vector<MNIST_Example> &data) {
    using namespace SaintCore;
    std::vector<std::vector<floatT>> tensor_data;
    for(int i = 0; i < data.size(); i++) {
        std::vector<floatT> sample(28 * 28);
        for (int j = 0; j < 28 * 28; j++) {
            sample[j] = data[i].pixels[j] / 255.0f;
        }
        tensor_data.push_back(sample);
    }
    return tensor_data;
}


SaintCore::Containers::SequenceContainer get_model() {
    using namespace SaintCore;
    using namespace SaintCore::Containers;
    using namespace SaintCore::Models;
    SequenceContainer sequence_container;
    LinearModel linear_model1(784, 128);
    ReLU relu1;
    LinearModel linear_model2(128, 10);
    CrossEntropyLoss cross_entropy_loss;
    sequence_container.add(std::make_shared<LinearModel>(linear_model1));
    sequence_container.add(std::make_shared<ReLU>(relu1));
    sequence_container.add(std::make_shared<LinearModel>(linear_model2));
    sequence_container.add(std::make_shared<CrossEntropyLoss>(cross_entropy_loss));
    return sequence_container;
}

int main() {
    using namespace SaintCore;
    using namespace Containers;
    using namespace Models;
    std::cout << "Sosali\n";


    Tensor input(std::vector<std::vector<floatT>>{{1, 1}, {1, 2}, {1, 3}, {1, 3}, {1, 3}});
    Tensor output(std::vector<std::vector<floatT>>{{0, 1, 0, 1, 1}});
    floatT alpha = 0.5;
    SequenceContainer model = get_model();
    for (int i = 0; i < 5; i++) {
        // std::cout << *sequence_container.get(0).get()->get_parameters()[0] << std::endl;
        // std::cout << *sequence_container.get(1).get()->get_parameters()[1] << std::endl;
        model.forward(input, output);
        model.backward(output);
        model.optimize(alpha);
    }

    // std::vector<MNIST_Example> dataset = load_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    // Tensor data = transform_data(dataset);
    // std::cout << "Transformed data: \n" << data << "\n";
}
