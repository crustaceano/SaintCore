#include <include/tensor.h>
#include <include/exceptions.h>
#include <ctime>
#include <iostream>

float randomFloat() {
	return (float)(rand()) / (float)(RAND_MAX);
}


SaintCore::Tensor::Tensor(int rows, int cols) : cols(cols), rows(rows) {
	data.assign(rows, std::vector<floatT>(cols, 0));
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			data[i][j] = randomFloat();
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


// 
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
	return data[ind];
}


// OK
std::ostream& SaintCore::operator<<(std::ostream& out, Tensor a) {
	for (int i = 0; i < a.get_rows(); i++) {
		for (int j = 0; j < a.get_cols(); j++)
			out << a[i][j] << " ";
		out << std::endl;
	}
	return out;
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

