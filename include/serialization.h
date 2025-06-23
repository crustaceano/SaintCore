#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <string>
#include "container.h"

namespace SaintCore {
    namespace Containers {
        // Вспомогательные free-функции
        void save_container(const SequenceContainer &model, const std::string &filename);
        void load_container(SequenceContainer &model, const std::string &filename);
    }
}

#endif // SERIALIZATION_H
