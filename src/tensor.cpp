#include <include/tensor.h>
#include <include/exceptions.h>
#include <ctime>
#include <vector>
#include <iostream>

SaintCore::floatT randomFloat() {
	return static_cast<SaintCore::floatT>(rand()) / static_cast<SaintCore::floatT>(RAND_MAX);
}


SaintCore::floatT my_abs(SaintCore::floatT a) {
	if (a >= 0) return a;
	return -a;
}


const SaintCore::floatT SaintCore::Tensor::eps = 1e-6;


SaintCore::Tensor::Tensor(int rows, int cols) : cols(cols), rows(rows) {
	data.assign(rows, std::vector<floatT>(cols, 0));
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			data[i][j] = randomFloat();
}


SaintCore::Tensor::Tensor(std::vector<std::vector<floatT>> const& vec) {
	data = vec;
	rows = vec.size();
	if (rows == 0) {
		cols = 0;
		return;
	}
	cols = vec[0].size();
	for (int i = 0; i < vec.size(); i++) {
		if (vec[0].size() != vec[i].size())
			throw BaseException("Different row size in matrix");
	}
}


// OK
SaintCore::Tensor SaintCore::operator+(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_cols() && a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	if (a.get_cols() == b.get_cols() && a.get_rows() == b.get_rows()) {
		Tensor c(a.get_rows(), a.get_cols());
		for (int i = 0; i < a.get_rows(); i++)
			for (int j = 0; j < a.get_cols(); j++)
				c.at(i, j) = a.at(i, j) + b.at(i, j);
		return c;
	}
	if (a.get_cols() == b.get_cols() && std::min(a.get_rows(), b.get_rows()) == 1) {
		Tensor c(std::max(a.get_rows(), b.get_rows()), a.get_cols());
		for (int i = 0; i < c.get_rows(); i++)
			for (int j = 0; j < c.get_cols(); j++) {
				if (a.get_rows() == 1)
					c.at(i, j) = a.at(0, j) + b.at(i, j);
				else
					c.at(i, j) = a.at(i, j) + b.at(0, j);
			}
		return c;
	}
	if (a.get_rows() == b.get_rows() && std::min(a.get_cols(), b.get_cols()) == 1) {
		Tensor c(a.get_rows(), std::max(a.get_cols(), b.get_cols()));
		for (int i = 0; i < c.get_rows(); i++)
			for (int j = 0; j < c.get_cols(); j++) {
				if (a.get_cols() == 1)
					c.at(i, j) = a.at(i, 0) + b.at(i, j);
				else
					c.at(i, j) = a.at(i, j) + b.at(i, 0);
			}
		return c;
	}
	throw BaseException("Strange size, bitch");;
}


// OK
SaintCore::Tensor SaintCore::operator*(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_rows()) throw BaseException("Wrong mul size");
	Tensor c(a.get_rows(), b.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < b.get_cols(); j++) {
			c.at(i, j) = 0;
			for (int k = 0; k < a.get_cols(); k++)
				c.at(i, j) += a.at(i, k) * b.at(k, j);
		}
	return c;
}


// OK
SaintCore::Tensor SaintCore::operator-(Tensor const& a, Tensor const& b) {
	return a + (b * (-1));
}


// OK
SaintCore::Tensor SaintCore::operator*(Tensor const& a, float b) {
	//if (b == 1) throw BaseException("Hahaha))");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < a.get_cols(); j++)
			c.at(i, j) = a.at(i, j) * b;
	return c;
}


// OK
SaintCore::Tensor SaintCore::operator%(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_cols() && a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	if (a.get_cols() == b.get_cols() && a.get_rows() == b.get_rows()) {
		Tensor c(a.get_rows(), a.get_cols());
		for (int i = 0; i < a.get_rows(); i++)
			for (int j = 0; j < a.get_cols(); j++)
				c.at(i, j) = a.at(i, j) * b.at(i, j);
		return c;
	}
	if (a.get_cols() == b.get_cols() && std::min(a.get_rows(), b.get_rows()) == 1) {
		Tensor c(std::max(a.get_rows(), b.get_rows()), a.get_cols());
		for (int i = 0; i < c.get_rows(); i++)
			for (int j = 0; j < c.get_cols(); j++) {
				if (a.get_rows() == 1)
					c.at(i, j) = a.at(0, j) * b.at(i, j);
				else
					c.at(i, j) = a.at(i, j) * b.at(0, j);
			}
		return c;
	}
	if (a.get_rows() == b.get_rows() && std::min(a.get_cols(), b.get_cols()) == 1) {
		Tensor c(a.get_rows(), std::max(a.get_cols(), b.get_cols()));
		for (int i = 0; i < c.get_rows(); i++)
			for (int j = 0; j < c.get_cols(); j++) {
				if (a.get_cols() == 1)
					c.at(i, j) = a.at(i, 0) * b.at(i, j);
				else
					c.at(i, j) = a.at(i, j) * b.at(i, 0);
			}
		return c;
	}
	throw BaseException("Strange size, bitch");;
}


void SaintCore::Tensor::checkIndex(int ind1, int ind2) const {
	if (ind1 < 0 || ind1 >= rows || ind2 < 0 || ind2 >= cols) throw BaseException("Index out of range");
}


SaintCore::floatT const& SaintCore::Tensor::at(int ind1, int ind2) const {
	checkIndex(ind1, ind2);
	return data[ind1][ind2];
}


SaintCore::floatT & SaintCore::Tensor::at(int ind1, int ind2) {
	checkIndex(ind1, ind2);
	return data[ind1][ind2];
}


std::vector<SaintCore::floatT> const& SaintCore::Tensor::operator[](int ind) const {
	return data[ind];
}


std::vector<SaintCore::floatT>& SaintCore::Tensor::operator[](int ind) {
	if (ind < 0 || ind >= rows) throw BaseException("Index out of range");
	return data[ind];
}


bool SaintCore::operator==(Tensor const& a, Tensor const& b) {
	if (a.rows != b.rows || a.cols != b.cols) return false;
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++) {
			if (my_abs(a.at(i, j) - b.at(i, j)) > SaintCore::Tensor::eps)
				return false;
		}
	return true;
}


bool SaintCore::operator!=(Tensor const& a, Tensor const& b) {
	return !(a == b);
}


// OK
SaintCore::Tensor SaintCore::Tensor::transposed() const {
	Tensor c(cols, rows);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			c.at(j, i) = data[i][j];
	return c;
}


int SaintCore::Tensor::get_cols() const {
	return cols;
}


int SaintCore::Tensor::get_rows() const {
	return rows;
}


std::ostream& SaintCore::operator<<(std::ostream& os, const Tensor& tensor) {
	for (int i = 0; i < tensor.get_rows(); i++) {
		for (int j = 0; j < tensor.get_cols(); j++) {
			os << tensor.at(i, j) << " ";
		}
		os << "\n";
	}
	return os;
}


SaintCore::Tensor SaintCore::get_E(int size) {
	SaintCore::Tensor e(size, size);
	for (int i = 0; i < e.get_rows(); i++) {
		for (int j = 0; j < e.get_cols(); j++) {
			if (i == j) e.at(i, j) = 1;
			else e.at(i, j) = 0;
		}
	}
	return e;
}


