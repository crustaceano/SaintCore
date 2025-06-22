#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <include/types.h>

namespace SaintCore {
	class Tensor {
		int cols, rows;
		std::vector<std::vector<floatT>> data;
		static const floatT eps;
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


		void checkIndex(int ind1, int ind2) const;
		// get by index
		floatT const& at(int ind1, int ind2) const;
		floatT & at(int ind1, int ind2);
		// std::vector<floatT> const& operator[](int ind) const;
		// std::vector<floatT>& operator[](int ind);

		friend bool operator==(Tensor const& a, Tensor const& b);

		// get transposed
		Tensor transposed() const;

		int get_cols() const;
		int get_rows() const;
		friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

	};
}

#endif //TENSOR_H
