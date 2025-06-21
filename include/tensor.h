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
		
		// сложение
		friend Tensor operator+(Tensor const& a, Tensor const& b);
		// вычитание
		friend Tensor operator-(Tensor const& a, Tensor const& b);
		// умножение
		friend Tensor operator*(Tensor const& a, Tensor const& b);
		// умножение на число
		friend Tensor operator*(Tensor const& a, float b);
		// поэлементное умножение
		friend Tensor operator%(Tensor const& a, Tensor const& b);

		// доступ по индексу
		std::vector<floatT> const& operator[](int ind) const;

		// получить транспонированную
		Tensor transposed() const;

		int get_cols() const;
		int get_rows() const;
	};
}

#endif //TENSOR_H
