#include <include/container.h>
#include <include/tensor.h>
#include "mnist_reader.h"
#include "visualization.h"
#include <random>
#include <map>


// Путь к распакованным файлам в папке mnist
const std::string BASE_PATH = "../mnist/";

int gen_rand(int a, int b) {
    int r = rand() * (1 << 16) + rand();
    return a + r % (b - a + 1);
}

void shuffle(std::vector<MNIST_Example> &data) {
    for (int i = 0; i < data.size(); i++) {
        int j = gen_rand(0, data.size() - 1);
        if (i != j) {
            std::swap(data[i], data[j]);
        }
    }
}

std::vector<MNIST_Example> load_dataset(const std::string &images_path, const std::string &labels_path) {
    std::string img_full = BASE_PATH + images_path;
    std::string lbl_full = BASE_PATH + labels_path;
    std::cout << "Loading MNIST images from: " << img_full << "\n";
    std::cout << "Loading MNIST labels from: " << lbl_full << "\n";
    auto data = load_mnist(img_full, lbl_full);
    std::cout << "Loaded " << data.size() << " samples\n";
    return data;
}

std::pair<SaintCore::Tensor, std::vector<SaintCore::floatT> > transform_data(std::vector<MNIST_Example> &data) {
    using namespace SaintCore;
    std::vector<std::vector<floatT> > tensor_data;
    std::vector<floatT> labels;
    for (int i = 0; i < data.size(); i++) {
        std::vector<floatT> sample(28 * 28);
        for (int j = 0; j < 28 * 28; j++) {
            sample[j] = data[i].pixels[j] / 255.0f;
        }
        tensor_data.push_back(sample);
        labels.push_back(data[i].label);
    }
    std::cout << "Successfully transformed data to tensor format\n";
    return {tensor_data, labels};
}


SaintCore::Containers::SequenceContainer get_model() {
    using namespace SaintCore;
    using namespace SaintCore::Containers;
    using namespace SaintCore::Models;
    SequenceContainer sequence_container;
    LinearModel linear_model1(784, 128);
    ReLU relu1;
    LinearModel linear_model2(128, 10);
    // ReLU relu2;
    CrossEntropyLoss cross_entropy_loss;
    sequence_container.add(std::make_shared<LinearModel>(linear_model1));
    sequence_container.add(std::make_shared<ReLU>(relu1));
    sequence_container.add(std::make_shared<LinearModel>(linear_model2));
    // sequence_container.add(std::make_shared<ReLU>(relu2));
    sequence_container.add(std::make_shared<CrossEntropyLoss>(cross_entropy_loss));
    return sequence_container;
}

std::map<std::string, float> evaluate_metrics(SaintCore::Containers::SequenceContainer &model, SaintCore::Tensor &data,
                                              SaintCore::Tensor &labels) {
    using namespace SaintCore;
    model.forward(data, labels);
    Tensor logits = model.get_logits(data, labels);
    Tensor soft_out = SaintCore::Functions::softmax(logits);

    Tensor predictions = Functions::argmax(soft_out);

    float total = labels.get_cols();
    int correct = 0;
    for (int i = 0; i < total; i++) {
        if (predictions.at(0, i) == labels.at(0, i)) {
            correct++;
        }
    }
    return {
        {"accuracy", correct / total},
    };
}

std::map<std::string, float> partial_fit(SaintCore::Containers::SequenceContainer &model,
                                         SaintCore::Tensor &train_data,
                                         SaintCore::Tensor &train_labels,
                                         SaintCore::floatT alpha) {
    using namespace SaintCore;
    using namespace SaintCore::Containers;
    using namespace SaintCore::Models;

    floatT loss = model.forward(train_data, train_labels);
    model.backward(train_labels);
    model.optimize(alpha);
    std::cout << "Loss on batch: " << loss << "\n";
    auto metrics = evaluate_metrics(model, train_data, train_labels);
    std::cout << "Accuracy on batch: " << metrics["accuracy"] << "\n";
    metrics["loss"] = loss;
    return metrics;
}

void train_epoch(SaintCore::Containers::SequenceContainer &model,
                 std::vector<SaintCore::Tensor> &train_data,
                 std::vector<SaintCore::Tensor> &train_labels,
                 SaintCore::floatT alpha) {
    using namespace SaintCore;
    using namespace SaintCore::Containers;
    using namespace SaintCore::Models;

    std::map<std::string, float> total_metrics;

    for (int i = 0; i < train_data.size(); i++) {
        auto metrics = partial_fit(model, train_data[i], train_labels[i], alpha);
        for (const auto &metric: metrics) {
            total_metrics[metric.first] += metric.second;
        }
        total_metrics["batch_am"] += 1;
    }

    std::cout << "Train Epoch Metrics: ";
    for (const auto &metric: total_metrics) {
        if (metric.first == "batch_am") continue;
        std::cout << metric.first << ": " << (metric.second / total_metrics["batch_am"]) << ", ";
    }
    std::cout << "\n";
}

void val_epoch(SaintCore::Containers::SequenceContainer &model,
               std::vector<SaintCore::Tensor> &val_data,
               std::vector<SaintCore::Tensor> &val_labels) {
    using namespace SaintCore;
    using namespace SaintCore::Containers;
    using namespace SaintCore::Models;

    std::map<std::string, float> total_metrics;
    for (int i = 0; i < val_data.size(); i++) {
        auto metrics = evaluate_metrics(model, val_data[i], val_labels[i]);
        for (const auto &metric: metrics) {
            total_metrics[metric.first] += metric.second;
        }
        total_metrics["batch_am"] += 1;
    }

    for (const auto &metric: total_metrics) {
        if (metric.first == "batch_am") continue;
        total_metrics[metric.first] /= total_metrics["batch_am"];
    }
    std::cout << "Validation Epoch Metrics: ";
    for (const auto &metric: total_metrics) {
        std::cout << metric.first << ": " << metric.second << ", ";
    }
}


int main() {
    using namespace SaintCore;
    using namespace Containers;
    using namespace Models;


    floatT alpha = 0.01f;
    SequenceContainer model = get_model();

    std::vector<MNIST_Example> dataset = load_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    shuffle(dataset);
    auto p = transform_data(dataset);
    Tensor data(p.first);
    std::vector labels(p.second);
    int train_size = data.get_rows() * 0.8;
    int val_size = data.get_rows() - train_size;
    int batch_size = 64;

    int train_bs = (train_size - 1) / batch_size + 1;
    int val_bs = (val_size - 1) / batch_size + 1;
    std::vector<Tensor> train_data(train_bs);
    std::vector<Tensor> train_labels(train_bs);
    std::vector<Tensor> val_data(val_bs);
    std::vector<Tensor> val_labels(val_bs);

    for (int i = 0; i < train_bs; i++) {
        std::vector<std::vector<floatT> > batch_data;
        std::vector<floatT> batch_labels;
        for (int j = i * batch_size; j < std::min(train_size, (i + 1) * batch_size); j++) {
            batch_data.push_back(data[j]);
            batch_labels.push_back(labels[j]);
        }
        train_data[i] = Tensor(batch_data);
        train_labels[i] = Tensor(batch_labels);
    }

    for (int i = 0; i < val_bs; i++) {
        std::vector<std::vector<floatT> > batch_data;
        std::vector<floatT> batch_labels;
        for (int j = train_size + i * batch_size; j < std::min(data.get_rows(), train_size + (i + 1) * batch_size); j
             ++) {
            batch_data.push_back(data[j]);
            batch_labels.push_back(labels[j]);
        }
        val_data[i] = Tensor(batch_data);
        val_labels[i] = Tensor(batch_labels);
    }

    // train_epoch(model, train_data, train_labels, alpha);
    // val_epoch(model, val_data, val_labels);
    // train_epoch(model, train_data, train_labels, alpha);

    model.load("ZOV.zov");
    val_epoch(model, val_data, val_labels);

}
