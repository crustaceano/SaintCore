#ifndef CONTAINER_H
#define CONTAINER_H

#include <cstddef>
#include <include/model.h>
#include <include/types.h>
#include <tensor.h>

namespace SaintCore {
    namespace Containers {
        class Container {
        public:
            virtual ~Container() = default;

            virtual void add(const Models::BaseModel &item) = 0;

            virtual void remove(const Models::BaseModel &item) = 0;

            virtual size_t size() const = 0;

            virtual bool is_empty() const = 0;

            virtual void clear() = 0;
        };

        class SequenceContainer : public Container {
        public:
            ~SequenceContainer() = default;

            Models::BaseModel &get(size_t index);

            const Models::BaseModel &get(size_t index) const;

            void insert(size_t index, const Models::BaseModel &item);

            void erase(size_t index);

            virtual size_t find(const Models::BaseModel &item);
        private:
            std::vector<Models::BaseModel> items_;
        };


    }


}

#endif //CONTAINER_H
