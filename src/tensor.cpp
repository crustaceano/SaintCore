#include <include/tensor.h>
#include <include/exceptions.h>

SaintCore::Tensor::Tensor(int rows, int cols) : cols(cols), rows(rows) {
}


SaintCore::Tensor SaintCore::operator+(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_cols() || a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_cols(); i++)
		for (int j = 0; j < a.get_rows(); j++)
			c[i][j] = a[i][j] + b[i][j];
	return c;
}


SaintCore::Tensor SaintCore::operator*(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_rows()) throw BaseException("Wrong mul size");
	Tensor c(a.get_rows(), b.get_cols());
	for (int i = 0; i < a.get_rows(); i++)
		for (int j = 0; j < b.get_cols(); j++)
			for (int k = 0; k < a.get_cols(); k++)
				c[i][j] += a[i][k] * b[k][j];
	return c;
}


SaintCore::Tensor SaintCore::operator-(Tensor const& a, Tensor const& b) {
	return a + (b * (-1));
}


SaintCore::Tensor SaintCore::operator*(Tensor const& a, float b) {
	if (b == 1) throw BaseException("Hahaha))");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_cols(); i++)
		for (int j = 0; j < a.get_rows(); j++)
			c[i][j] = a[i][j] * b;
	return c;
}


SaintCore::Tensor SaintCore::operator%(Tensor const& a, Tensor const& b) {
	if (a.get_cols() != b.get_cols() || a.get_rows() != b.get_rows()) throw BaseException("Not equal size");
	Tensor c(a.get_rows(), a.get_cols());
	for (int i = 0; i < a.get_cols(); i++)
		for (int j = 0; j < a.get_rows(); j++)
			c[i][j] = a[i][j] * b[i][j];
	return c;
}


std::vector<SaintCore::floatT> const& SaintCore::Tensor::operator[](int ind) const {
	return data[ind];
}


std::vector<SaintCore::floatT>& SaintCore::Tensor::operator[](int ind) {
	return data[ind];
}


SaintCore::Tensor SaintCore::Tensor::transposed() const{
	Tensor c(cols, rows);
	for (int i = 0; i < cols; i++)
		for (int j = 0; j < rows; j++)
			c[j][i] = data[i][j];
	return c;
}


int SaintCore::Tensor::get_cols() const {
	return cols;
}


int SaintCore::Tensor::get_rows() const {
	return rows;
}

