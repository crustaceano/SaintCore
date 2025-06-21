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
		Tensor(int rows, int cols);
		
<<<<<<< HEAD
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
=======
		// ????????
		friend Tensor operator+(Tensor const& a, Tensor const& b);
		// ?????????
		friend Tensor operator-(Tensor const& a, Tensor const& b);
		// ?????????
		friend Tensor operator*(Tensor const& a, Tensor const& b);
		// ????????? ?? ?????
		friend Tensor operator*(Tensor const& a, float b);
		// ???????????? ?????????
		friend Tensor operator%(Tensor const& a, Tensor const& b);

		// ?????? ?? ???????
		std::vector<floatT> const& operator[](int ind) const;
		std::vector<floatT>& operator[](int ind);

		// ???????? ?????????????????
>>>>>>> 576ca1ba1d447d0b37be6d5183f0c21506915748
		Tensor transposed() const;

		int get_cols() const;
		int get_rows() const;
	};
}

#endif //TENSOR_H
