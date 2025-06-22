#include <algorithm>
#include <include/container.h>

#include "exceptions.h"

SaintCore::Containers::SequenceContainer::~SequenceContainer() = default;


void SaintCore::Containers::SequenceContainer::checkIndex(size_t index) const {
    if (index >= items_.size()) throw BaseException("Incorrect index");
}


std::shared_ptr<SaintCore::Models::BaseModel> SaintCore::Containers::SequenceContainer::get(size_t index) {
    checkIndex(index);
    return items_[index];
}


void SaintCore::Containers::SequenceContainer::add(std::shared_ptr<Models::BaseModel> item) {
    items_.push_back(item);
}


void SaintCore::Containers::SequenceContainer::remove(size_t index) {
    checkIndex(index);
    items_.erase(items_.begin() + index);
}


size_t SaintCore::Containers::SequenceContainer::size() const  {
    return items_.size();
}


bool SaintCore::Containers::SequenceContainer::is_empty() const  {
    return items_.empty();
}


void SaintCore::Containers::SequenceContainer::clear()  {
    items_.clear();
}


void SaintCore::Containers::SequenceContainer::forward(const SaintCore::Tensor &input) {
    inputs.clear();
    inputs.push_back(input);
    for (int i = 0; i + 1 < items_.size(); i++) {
        inputs.push_back(items_[i].get()->forward(inputs.back()));
    }
}


void SaintCore::Containers::SequenceContainer::backward() {
    paramsGrads.clear();
    Tensor step = get_E(items_.back().get()->getGrad(inputs.back()).get_rows());
    for (int i = items_.size() - 1; i >= 0; i--) {
        std::vector<Tensor> params = items_[i].get()->getTrainParams_grad(inputs[i]);
        paramsGrads.push_back({});
        for (int j = 0; j < params.size(); j++) {
            paramsGrads.back().push_back(params[j] * step);
        }
        step = step * items_[i].get()->getGrad(inputs[i]);
    }
    std::reverse(paramsGrads.begin(), paramsGrads.end());
}


void SaintCore::Containers::SequenceContainer::optimize(floatT alpha) {
    for (int i = 0; i < items_.size(); i++) {
        std::vector<Tensor*> params = items_[i].get()->get_parameters();
        std::vector<Tensor> new_params;
        // std :: cout << paramsGrads.size() << std::endl;
        // std :: cout << paramsGrads[0].size() << std::endl;
        // std :: cout << params.size() << std::endl;
        for (int j = 0; j < params.size(); j++) {
            paramsGrads[i][j] * alpha;
            *(params[j]);
            new_params.push_back(*(params[j]) - paramsGrads[i][j] * alpha);
        }
        items_[i].get()->update_parameters(new_params);
    }
}

