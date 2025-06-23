// #include <gtest/gtest.h>
// #include <cmath>
// #include <include/functions.h>
// #include <include/exceptions.h>
// #include <include/tensor.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <cstdint>
// #include "visualization.h"
// #include <include/container.h>
// #include <include/model.h>
//
// #include "exceptions.h"
//
// TEST(test_arab_container, bomb_test) {
//         using namespace SaintCore;
//         using namespace Containers;
//         using namespace Models;
//
//         SequenceContainer sequence_container;
//         LinearModel linear_model1(2, 3);
//         LinearModel linear_model2(3, 9);
//         LinearModel linear_model3(9, 13);
//         LinearModel linear_model4(13, 21);
//         LinearModel linear_model5(21, 34);
//         LinearModel linear_model6(34, 1);
//         sequence_container.add(std::make_shared<LinearModel>(linear_model1));
//         sequence_container.add(std::make_shared<LinearModel>(linear_model2));
//         sequence_container.add(std::make_shared<LinearModel>(linear_model3));
//         sequence_container.add(std::make_shared<LinearModel>(linear_model4));
//         sequence_container.add(std::make_shared<LinearModel>(linear_model5));
//         sequence_container.add(std::make_shared<LinearModel>(linear_model6));
//
//         Tensor input({{1, 1}});
//         floatT alpha = 0.5;
//
//         for (int i = 0; i < 5; i++) {
//             std::cout << *sequence_container.get(0).get()->get_parameters()[0] << std::endl;
//             std::cout << *sequence_container.get(1).get()->get_parameters()[1] << std::endl;
//             sequence_container.forward(input);
//             sequence_container.backward();
//             sequence_container.optimize(alpha);
//         }
//         std::cout << *sequence_container.get(0).get()->get_parameters()[0] << std::endl;
//         std::cout << *sequence_container.get(1).get()->get_parameters()[1] << std::endl;
// }