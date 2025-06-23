#include "mnist_reader.h"
#include "visualization.h"
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <chrono>

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

struct MLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    MLPImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(28*28, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), 28*28});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};
TORCH_MODULE(MLP);

std::pair<torch::Tensor, torch::Tensor> vec2tens(std::vector<MNIST_Example> dataset) {
    const size_t size = dataset.size();

    // Опции для данных
    torch::TensorOptions data_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::TensorOptions label_options = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor data = torch::empty({(int64_t)size, 1, 28, 28}, data_options);
    torch::Tensor labels = torch::empty({(int64_t)size}, label_options);
    // Заполняем
    for (size_t i = 0; i < size; ++i) {
        // Преобразуем vector<double> в Tensor float и нормализуем
        torch::Tensor sample = torch::from_blob(
            dataset[i].pixels.data(),
            {28*28},
            torch::TensorOptions().dtype(torch::kFloat64)
        ).to(torch::kFloat32).div_(255.0f);
        // reshape к [1,28,28]
        sample = sample.view({1, 28, 28});
        // Копируем в train_data[i]
        data[i].copy_(sample);
        // Метка
        labels[i] = dataset[i].label;
    }
    return {data, labels};
}

int main() {
    // 1. Загрузка наборов данных
    auto train_set = load_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    auto test_set  = load_dataset("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte");

    if (train_set.empty() || test_set.empty()) {
        std::cerr << "Failed to load MNIST data. Exiting.\n";
        return 1;
    }

    // 2. Конвертация std::vector<MNIST_Example> в torch::Tensor
    auto [train_data, train_labels] = vec2tens(train_set);
    auto [test_data, test_labels] = vec2tens(test_set);

    const size_t train_size = train_data.size(0);
    const size_t test_size = test_data.size(0);

    torch::TensorOptions label_options = torch::TensorOptions().dtype(torch::kInt64);

    // 3. Гиперпараметры
    const size_t num_epochs = 3;
    const size_t batch_size = 64;
    const size_t num_batches = (train_size + batch_size - 1) / batch_size;

    // 4. Создать модель и оптимизатор
    MLP model;
    torch::optim::SGD optimizer(model->parameters(),0.01); // lr = 0.01

    // 5. Обучение модели
    std::vector<size_t> indices(train_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937_64 rnd_engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t correct = 0;
        size_t total = 0;

        // Перемешиваем индексы для каждой эпохи
        std::shuffle(indices.begin(), indices.end(), rnd_engine);

        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Вычисляем начало и конец батча в индексе
            size_t start = batch_idx * batch_size;
            size_t end = std::min(start + batch_size, train_size);
            size_t current_batch_size = end - start;
            if (current_batch_size == 0) break;

            // Собираем индексы батча
            // Создаём tensor из индексов
            std::vector<int64_t> batch_indices;
            batch_indices.reserve(current_batch_size);
            for (size_t j = start; j < end; ++j) {
                batch_indices.push_back((int64_t)indices[j]);
            }
            torch::Tensor idx_tensor = torch::tensor(batch_indices, label_options); // int64

            // Получаем батч через index_select по нулевому измерению
            torch::Tensor batch_data = train_data.index_select(0, idx_tensor);
            torch::Tensor batch_labels = train_labels.index_select(0, idx_tensor);

            optimizer.zero_grad();
            torch::Tensor outputs = model->forward(batch_data); // [batch,10]
            torch::Tensor loss = torch::nn::functional::cross_entropy(outputs, batch_labels);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>() * current_batch_size;

            // Подсчёт точности
            auto preds = outputs.argmax(1);
            auto correct_batch = preds.eq(batch_labels).sum().item<int64_t>();
            correct += correct_batch;
            total += current_batch_size;
        }

        double avg_loss = epoch_loss / train_size;
        double accuracy = static_cast<double>(correct) / total * 100.0;
        std::cout << "Epoch [" << epoch << "/" << num_epochs << "], "
                  << "Train Loss: " << avg_loss << ", Train Acc: " << accuracy << "%\n";

        // 5.5. Оценка на тесте
        model->eval();
        torch::NoGradGuard no_grad;
        double test_loss = 0.0;
        size_t test_correct = 0;
        size_t test_total = 0;

        const size_t test_batch_size = 1000; // или другой
        const size_t num_test_batches = (test_size + test_batch_size - 1) / test_batch_size;
        for (size_t batch_idx = 0; batch_idx < num_test_batches; ++batch_idx) {
            size_t start = batch_idx * test_batch_size;
            size_t end = std::min(start + test_batch_size, test_size);
            size_t current_batch_size = end - start;
            if (current_batch_size == 0) break;

            // Собираем прямой диапазон: здесь не перемешиваем test
            // Можно собрать индексы  start..end-1
            std::vector<int64_t> batch_indices;
            batch_indices.reserve(current_batch_size);
            for (size_t j = start; j < end; ++j) {
                batch_indices.push_back((int64_t)j);
            }
            torch::Tensor idx_tensor = torch::tensor(batch_indices, label_options);
            torch::Tensor batch_data = test_data.index_select(0, idx_tensor);
            torch::Tensor batch_labels = test_labels.index_select(0, idx_tensor);

            torch::Tensor outputs = model->forward(batch_data);
            torch::Tensor loss = torch::nn::functional::cross_entropy(outputs, batch_labels);
            test_loss += loss.item<double>() * current_batch_size;

            auto preds = outputs.argmax(1);
            test_correct += preds.eq(batch_labels).sum().item<int64_t>();
            test_total += current_batch_size;
        }
        double avg_test_loss = test_loss / test_size;
        double test_acc = static_cast<double>(test_correct) / test_total * 100.0;
        std::cout << "Epoch [" << epoch << "] Test Loss: " << avg_test_loss
                  << ", Test Acc: " << test_acc << "%\n";
    }

    // 6. Сохранение модели
    std::string model_path = "mnist_mlp.pt";
    torch::save(model, model_path);
    std::cout << "Model saved to " << model_path << "\n";

    // Пример доступа к данным
    if (!train_set.empty()) {
        std::cout << "\nSave example in bmp with label: " << train_set[0].label << std::endl;

        visualize(train_set[0].pixels, "../visualized_data/output0.bmp");
        std::cout << std::endl;
    }



    return 0;
}