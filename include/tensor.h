#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <include/types.h>

namespace SaintCore {
	class Tensor {
		int cols, rows;
		std::vector<std::vector<floatT>> data;
	public:
		Tensor(int rows, int cols);
		Tensor(std::vector<std::vector<floatT>> const & vec);
		
		// sum
		friend Tensor operator+(Tensor const& a, Tensor const& b);
		// substract
		friend Tensor operator-(Tensor const& a, Tensor const& b);
		// multiply
		friend Tensor operator*(Tensor const& a, Tensor const& b);
		// mul by float
		friend Tensor operator*(Tensor const& a, float b);
		// elements multiply
		friend Tensor operator%(Tensor const& a, Tensor const& b);

		// get by index
		std::vector<floatT> const& operator[](int ind) const;
		std::vector<floatT>& operator[](int ind);

		// get transposed
		Tensor transposed() const;

		int get_cols() const;
		int get_rows() const;
		friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

	};
}

#endif //TENSOR_H
