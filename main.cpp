#include <torch/torch.h>
#include <iostream>

int main() {
    // Создаём тензор-скаляр со значением 2.0 и включённым автоградом
    torch::Tensor x = torch::tensor(2.0, torch::requires_grad());

    // Выполняем операцию y = x * x
    torch::Tensor y = x * x;

    // Вычисляем градиент dy/dx
    y.backward();

    // Выводим результат: x, y и градиент
    std::cout << "x: " << x.item<double>()
              << ", y: " << y.item<double>()
              << ", gradient dy/dx: " << x.grad().item<double>()
              << std::endl;

    return 0;
}
