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

            // Методы сериализации модели (по желанию)
            virtual void save(const std::string &path) const;

            virtual std::vector<Tensor *> get_parameters() const;

            virtual Tensor getGrad(const Tensor &input) const;

            virtual std::vector<Tensor> getTrainParams_grad() const;

            virtual void load(const std::string &path) {
            }

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
