#include <ATen/core/interned_strings.h>
x//
// Created by axmed on 21.06.2025.
//

#ifndef MODEL_H
#define MODEL_H
#include <functions.h>

#include "include/types.h"
#include "include/tensor.h"
#include <string>
#include <cmath>

namespace SaintCore {
    namespace Models {
        class BaseModel {
        public:
            virtual ~BaseModel() = default;
            virtual Tensor forward(const Tensor &input) = 0;

            // Переключение режима обучения/инференса
            void train() { this->training_ = true; }
            void eval() { this->training_ = false; }

            virtual std::vector<Tensor *> get_parameters() const = 0;
            virtual void update_parameters(std::vector<Tensor> &new_params) = 0;
            virtual Tensor getGrad(const Tensor &input) const = 0;
            virtual std::vector<Tensor> getTrainParams_grad(const Tensor& input) const = 0;
        protected:
            bool training_ = true;
        };

        class LinearModel : public BaseModel {
        public:
            explicit LinearModel(int in_channels, int out_channels)
                : in_channels(in_channels),
                  out_channels(out_channels),
                  weights(in_channels, out_channels),
                  bias(1, out_channels) {
            }

            ~LinearModel() override;

            Tensor forward(const Tensor &input) override;

            std::vector<Tensor *> get_parameters() const override;
            void update_parameters(std::vector<Tensor> &new_params) override;
            Tensor getGrad(const Tensor &input) const override;
            std::vector<Tensor> getTrainParams_grad(const Tensor& input) const override;

            Tensor get_weights() const {
                return weights;
            }

            Tensor get_bias() const {
                return bias;
            }

        private:
            int in_channels;
            int out_channels;
            Tensor weights;
            Tensor bias;
        };

        class ReLU : public BaseModel {
        public:
            explicit ReLU() = default;

            Tensor forward(const Tensor &input) override {
                Tensor output(input.get_rows(), input.get_cols());
                for (int i = 0; i < input.get_rows(); ++i) {
                    for (int j = 0; j < input.get_cols(); ++j) {
                        output.at(i, j) = std::max(0.0f, input.at(i, j));
                    }
                }
                return output;
            }

            [[nodiscard]] std::vector<Tensor *> get_parameters() const override {
                return {};
            }

            void update_parameters(std::vector<Tensor> &new_params) override {}

            [[nodiscard]] Tensor getGrad(const Tensor &input) const override {
                Tensor grad(input.get_rows(), input.get_cols());
                for (int i = 0; i < input.get_rows(); ++i) {
                    for (int j = 0; j < input.get_cols(); ++j) {
                        grad.at(i, j) = input.at(i, j) > 0 ? 1.0f : 0.0f;
                    }
                }
                return grad;
            }

            [[nodiscard]] std::vector<Tensor> getTrainParams_grad(const Tensor& input) const override {
                return {};
            }
        };

        class CrossEntropyLoss : public BaseModel {
        public:
            CrossEntropyLoss() = default;

            Tensor forward(const Tensor &input, const Tensor& targets) {
                Tensor softmax_out = Functions::softmax(input);

                int batch_size = softmax_out.get_rows();
                float loss = 0.0f;
                for(int i = 0; i < batch_size; ++i) {
                    int target_class = static_cast<int>(targets.at(i, 0));
                    float log_prob =  -std::log(softmax_out.at(i, target_class));
                    loss += log_prob;
                }
                return Tensor({std::vector<floatT>{loss / batch_size}});
            }

            // ∂L/∂z = softmax(z) - one_hot(y)
            Tensor getGrad(const Tensor &input, const Tensor& targets) const {
                Tensor softmax_out = Functions::softmax(input);
                return softmax_out - Functions::one_hot(targets, input.get_cols());
            }

            std::vector<Tensor *> get_parameters() const override {
                return {};
            }

            void update_parameters(std::vector<Tensor> &new_params) override {}

            std::vector<Tensor> getTrainParams_grad(const Tensor& input) const override {
                return {};
            }
        };
    }
}

#endif //MODEL_H
