//
// Created by axmed on 21.06.2025.
//

#ifndef MODEL_H
#define MODEL_H
#include "include/types.h"
#include "include/tensor.h"
#include <string>

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
    }
}

#endif //MODEL_H
