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

            virtual Tensor forward(const std::vector<Tensor> &input) = 0;

            // Переключение режима обучения/инференса
            void train() { this->training_ = true; }
            void eval() { this->training_ = false; }

            virtual std::vector<Tensor *> get_parameters() const = 0;

            virtual void update_parameters(std::vector<Tensor> &new_params) = 0;

            virtual Tensor getGrad(const std::vector<Tensor> &input) const = 0;

            virtual std::vector<Tensor> getTrainParams_grad(const Tensor &input) const = 0;

            virtual Tensor propagateGrad(const std::vector<Tensor> &input, Tensor &grad) = 0;

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

            Tensor forward(const std::vector<Tensor> &input) override;

            Tensor propagateGrad(const std::vector<Tensor> &input, Tensor &grad) override;

            std::vector<Tensor *> get_parameters() const override;

            void update_parameters(std::vector<Tensor> &new_params) override;

            Tensor getGrad(const std::vector<Tensor> &input) const override;

            std::vector<Tensor> getTrainParams_grad(const Tensor &input) const override;

            // virtual Tensor propagateGrad(Tensor &grad) override;

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

            Tensor forward(const std::vector<Tensor> &inputs) override {
                const Tensor &input = inputs[0];
                Tensor output(input.get_rows(), input.get_cols());
                for (int i = 0; i < input.get_rows(); ++i) {
                    for (int j = 0; j < input.get_cols(); ++j) {
                        output.at(i, j) = std::max(0.0f, input.at(i, j));
                    }
                }
                return output;
            }

            std::vector<Tensor *> get_parameters() const override {
                return {};
            }

            void update_parameters(std::vector<Tensor> &new_params) override {
            }

            Tensor getGrad(const std::vector<Tensor> &inputs) const override {
                Tensor input = inputs[0];
                Tensor grad(input.get_rows(), input.get_cols());
                for (int i = 0; i < input.get_rows(); ++i) {
                    for (int j = 0; j < input.get_cols(); ++j) {
                        grad.at(i, j) = input.at(i, j) > 0 ? 1.0f : 0.0f;
                    }
                }
                return grad;
            }

            Tensor propagateGrad(const std::vector<Tensor> &input, Tensor &grad) override {
                Tensor cur_grads = getGrad(input);
                
            }


            [[nodiscard]] std::vector<Tensor> getTrainParams_grad(const Tensor &input) const override {
                return {};
            }
        };

        class CrossEntropyLoss : public BaseModel {
        public:
            CrossEntropyLoss() = default;

            Tensor forward(const std::vector<Tensor> &inputs) override {
                Tensor input = inputs[0], targets = inputs[1];
                Tensor softmax_out = Functions::softmax(input);

                int batch_size = softmax_out.get_rows();
                float loss = 0.0f;
                for (int i = 0; i < batch_size; ++i) {
                    int target_class = static_cast<int>(targets.at(0, i));
                    float log_prob = -std::log(softmax_out.at(i, target_class));
                    loss += log_prob;
                }
                return Tensor({std::vector<floatT>{loss / batch_size}});
            }

            // ∂L/∂z = softmax(z) - one_hot(y)
            Tensor getGrad(const std::vector<Tensor> &inputs) const override {
                Tensor input = inputs[0], targets = inputs[1];
                Tensor softmax_out = Functions::softmax(input);
                return softmax_out - Functions::one_hot(targets, input.get_rows());
            }

            std::vector<Tensor *> get_parameters() const override {
                return {};
            }

            void update_parameters(std::vector<Tensor> &new_params) override {
            }

            std::vector<Tensor> getTrainParams_grad(const Tensor &input) const override {
                return {};
            }
        };
    }
}

#endif //MODEL_H
