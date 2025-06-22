#include <include/container.h>
#include <include/tensor.h>
#include <include/exceptions.h>


int main() {
    using namespace SaintCore;
    using namespace Containers;
    using namespace Models;
    std::cout << "Sosali\n";
    SequenceContainer sequence_container;
    LinearModel linear_model1(2, 3);
    ReLU relu1;
    LinearModel linear_model2(3, 2);
    ReLU relu2;
    CrossEntropyLoss cross_entropy_loss;
    sequence_container.add(std::make_shared<LinearModel>(linear_model1));
    // sequence_container.add(std::make_shared<ReLU>(relu1));
    sequence_container.add(std::make_shared<LinearModel>(linear_model2));
    // sequence_container.add(std::make_shared<ReLU>(relu2));
    sequence_container.add(std::make_shared<CrossEntropyLoss>(cross_entropy_loss));


    Tensor input(std::vector<std::vector<floatT>>{{1, 1}});
    Tensor output(std::vector<std::vector<floatT>>{{0}});
    floatT alpha = 0.5;

    for (int i = 0; i < 5; i++) {
        // std::cout << *sequence_container.get(0).get()->get_parameters()[0] << std::endl;
        // std::cout << *sequence_container.get(1).get()->get_parameters()[1] << std::endl;
        sequence_container.forward(input, output);
        sequence_container.backward(output);
        sequence_container.optimize(alpha);
    }
    std::cout << *sequence_container.get(0).get()->get_parameters()[0] << std::endl;
    std::cout << *sequence_container.get(1).get()->get_parameters()[1] << std::endl;
}
