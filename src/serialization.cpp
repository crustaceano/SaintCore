#include "serialization.h"
#include <fstream>
#include <stdexcept>

using namespace SaintCore::Containers;

void Container::save(const std::string &filename) const {
    save_container(static_cast<const SequenceContainer&>(*this), filename);
}

void Container::load(const std::string &filename) {
    load_container(static_cast<SequenceContainer&>(*this), filename);
}

void SequenceContainer::save(const std::string &filename) const {
    save_container(*this, filename);
}

void SequenceContainer::load(const std::string &filename) {
    load_container(*this, filename);
}

void SaintCore::Containers::save_container(const SequenceContainer &cont, const std::string &filename) {
    using namespace SaintCore;
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for saving: " + filename);

    // Сохраняем число слоёв
    size_t n = cont.size();
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // Для каждого слоя сохраняем параметры
    for (size_t i = 0; i < n; ++i) {
        auto layer = cont.get(i);
        auto params = layer->get_parameters();
        size_t pcount = params.size();
        ofs.write(reinterpret_cast<const char*>(&pcount), sizeof(pcount));
        for (auto p : params) {
            int rows = p->get_rows();
            int cols = p->get_cols();
            ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    float v = p->at(r, c);
                    ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
        }
    }
}

void SaintCore::Containers::load_container(SequenceContainer &cont, const std::string &filename) {
    using namespace SaintCore;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file for loading: " + filename);

    size_t n;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != cont.size())
        throw std::runtime_error("Layer count mismatch: file has " + std::to_string(n) +
                                 ", container has " + std::to_string(cont.size()));

    for (size_t i = 0; i < n; ++i) {
        auto layer = cont.get(i);
        size_t pcount_file;
        ifs.read(reinterpret_cast<char*>(&pcount_file), sizeof(pcount_file));
        auto params = layer->get_parameters();
        if (pcount_file != params.size())
            throw std::runtime_error("Parameter count mismatch at layer " + std::to_string(i));

        std::vector<Tensor> new_params;
        new_params.reserve(pcount_file);
        for (size_t j = 0; j < pcount_file; ++j) {
            int rows, cols;
            ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            Tensor T(rows, cols);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    float v;
                    ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
                    T.at(r, c) = v;
                }
            new_params.push_back(std::move(T));
        }
        layer->update_parameters(new_params);
    }
}

