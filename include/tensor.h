//
// Created by axmed on 21.06.2025.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <include/types.h>

namespace SaintCore {
	class Tensor {
		int cols, rows;
		std::vector<std::vector<floatT>> data;
	public:
		friend Tensor operator+(Tensor const& a, Tensor const& b);
		friend Tensor operator*(Tensor const& a, Tensor const& b);
		friend Tensor operator-(Tensor const& a, Tensor const& b);
		friend Tensor operator*(Tensor const& a, int b);
		friend Tensor operator%(Tensor const& a, Tensor const& b);
	};
}

#endif //TENSOR_H
