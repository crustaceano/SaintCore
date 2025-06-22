#ifndef CONTAINER_H
#define CONTAINER_H

#include <cstddef>
#include <memory>
#include <include/model.h>
#include <include/types.h>
#include <tensor.h>

namespace SaintCore {
    namespace Containers {
        class Container {
        public:
            virtual ~Container() = default;
            virtual void add(std::shared_ptr<Models::BaseModel> item) = 0;
            virtual void remove(size_t index) = 0;
            virtual size_t size() const = 0;
            virtual bool is_empty() const = 0;
            virtual void clear() = 0;
            virtual void optimize(floatT alpha) = 0;
        };

        class SequenceContainer : public Container {
        public:
            virtual ~SequenceContainer();

            void checkIndex(size_t index) const;

            std::shared_ptr<Models::BaseModel> get(size_t index);

            void add(std::shared_ptr<Models::BaseModel> item) override;

            void remove(size_t index) override;

            virtual size_t size() const override;
            virtual bool is_empty() const override;
            virtual void clear() override;

            void forward(const Tensor& input);
            void backward();
            void optimize(floatT alpha) override;
        private:
            std::vector<std::shared_ptr<Models::BaseModel>> items_;
            std::vector<Tensor> inputs;
            std::vector<std::vector<Tensor>> paramsGrads;
        };
    }
}

#endif //CONTAINER_H
