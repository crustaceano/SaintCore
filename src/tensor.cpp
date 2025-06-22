#include <include/tensor.h>
#include <include/exceptions.h>
#include <ctime>
#include <vector>
#include <iostream>

float randomFloat() {
	return (float)(rand()) / (float)(RAND_MAX);
}


const SaintCore::floatT SaintCore::Tensor::eps = 1e-3;


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
	if (a.get_cols() != b.get_cols() || a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < a.get_cols(); j++)
			c[i][j] = a[i][j] + b[i][j];
	return c;
}


// OK
SaintCore::Tensor SaintCore::operator*(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_rows()) throw BaseException("Wrong mul size");
	Tensor c(a.get_rows(), b.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < b.get_cols(); j++) {
			c[i][j] = 0;
			for (int k = 0; k < a.get_cols(); k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	return c;
}


// OK
SaintCore::Tensor SaintCore::operator-(Tensor const& a, Tensor const& b) {
	return a + (b * (-1));
}


// OK
SaintCore::Tensor SaintCore::operator*(Tensor const& a, float b) {
	if (b == 1) throw BaseException("Hahaha))");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < a.get_cols(); j++)
			c[i][j] = a[i][j] * b;
	return c;
}


// OK
SaintCore::Tensor SaintCore::operator%(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_cols() || a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < a.get_cols(); j++)
			c[i][j] = a[i][j] * b[i][j];
	return c;
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
		for (int j = 0; j < a.cols; j++)
			if (abs(a[i][j] - b[i][j]) > SaintCore::Tensor::eps)
				return false;
	return true;
}


// OK
SaintCore::Tensor SaintCore::Tensor::transposed() const {
	Tensor c(cols, rows);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			c[j][i] = data[i][j];
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
			os << tensor[i][j] << " ";
		}
		os << "\n";
	}
	return os;
}
