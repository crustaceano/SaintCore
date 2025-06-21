//
// Created by axmed on 21.06.2025.
//

#ifndef MODEL_H
#define MODEL_H
#include "include/types.h"
#include "include/tensor.h"
#include <string>

namespace SaintCore {
    class BaseModel {
    public:
        virtual ~BaseModel() = default;

        virtual Tensor forward(const Tensor& input) = 0;

        // Переключение режима обучения/инференса
        bool train() const { return training_;}
        bool eval() const { return !training_;}

        // Методы сериализации модели (по желанию)
        virtual void save(const std::string& path) const {}
        virtual void load(const std::string& path) {}

    protected:
        bool training_ = true;
    };
}

#endif //MODEL_H
