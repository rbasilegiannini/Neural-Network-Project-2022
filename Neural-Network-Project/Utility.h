#pragma once

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using boost::numeric::ublas::prod;
using boost::numeric::ublas::matrix;
using std::vector;
using std::max_element;
using std::min_element;
using std::transform;
using std::back_inserter;
using std::plus;

/**
 * 
 * Utilities used to support modules.
 * 
 */

/**
	Real has the float precision. 
 */
typedef float Real;

/**
	Matrix of real type.
 */
typedef matrix<Real> mat_r;

/**
	Vector of real type.
 */
typedef vector<Real> vec_r;

/**
 * Range (of integers) generator.
 * 
 * \param	first is the first element of the range.
 * \param	last is the last element of the range (not included).
 * \return	a vector with the range.
 */
inline vector<long long int> RangeGen(const long long int first, const long long int last) {
	long long int size = abs(last - first);
	vector<long long int> range(size);
	long long int idx{ first };

	if (last > first) {
		for (auto& value : range)
			value = idx++;
	}
	else if (last < first) {
		for (auto& value : range)
			value = idx--;
	}
	else
		range.push_back(idx);

	return range;
}

/**
 * This function computes the SoftMax.
 * 
 * \param outputs is the column vector with all the outputs.
 * \param idxOutput is the neuron's output index.
 * \return the result.
 */
inline Real SoftMax(const mat_r& outputs, const size_t idxOutput) {

	//	Handling potential numeric instability
	Real maxValue = *max_element(outputs.data().begin(), outputs.data().end());
	mat_r outputsNorm{ outputs };
	for (auto& o : outputsNorm.data())
		o -= maxValue;

	Real summation{ 0 };
	for (const auto& h : RangeGen(0, outputsNorm.size1()))
		summation += exp(outputsNorm(h, 0));
	Real SM = exp(outputsNorm(idxOutput, 0)) / summation;

 	return SM;
}


/**
 * This function performs the row by column product.
 * 
 * \param row
 * \param column
 * \return the result.
 */
inline Real row_by_column (const mat_r& row, const mat_r& column) {

	auto result = prod(row, column);
	return result(0, 0);
}

/**
 * This function extracts a column from a matrix of real.
 * 
 * \param mat
 * \param col is the index of the column.
 * \return the column as a column vector.
 */
inline mat_r extract_column(const mat_r& mat, const size_t col) {

	mat_r extractor(mat.size1(), 1);
	for (const auto& i : RangeGen(0, mat.size1()))
		extractor(i, 0) = mat(i, col);

	return extractor;
}

/**
 * This function converts a matrix in a vector.
 * 
 * \param mat
 * \return a vector with the same type element of the matrix.
 */
template <typename T>
inline vector<T> ConvertMatToArray(const matrix<T>& mat) {
	vector<T> arr;
	for (const auto& row : RangeGen(0, mat.size1()))
		for (const auto& col : RangeGen(0, mat.size2()))
			arr.push_back(mat(row, col));
	return arr;
}

/**
 * This function is used to normalize a vector in [l_ext, r_ext].
 * 
 * \param vec is the vector to normalize.
 * \param max value for normalization.
 * \param min value for normalization.
 * \return A vector of the same type but normalized.
 */
template <typename T>
inline vector<T> NormalizeVector(const vector<T> vec, T max, T min, T l_ext, T r_ext) {
	vector<T> arrNorm;

	for (const auto& e : vec)
		arrNorm.push_back(l_ext + (((e - min) * (r_ext - l_ext)) / (max - min)));
	return arrNorm;
}

//	Overload operator += of vector
template <typename T>
vector<T>& operator+=(vector<T>& vec1, const vector<T>& vec2) {
	assert(vec1.size() == vec2.size());

	for (const auto& i : RangeGen(0, vec1.size()))
		vec1[i] = vec1[i] + vec2[i];
	return vec1;
}

//	Overload operator + of vector
template <typename T>
vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
	assert(vec1.size() == vec2.size());

	vector<T> result;
	result.reserve(vec1.size());

	transform(vec1.begin(), vec1.end(), vec2.begin(),
		back_inserter(result), plus<T>());
	return result;
}

/**
 * This function returns the sign of an input.
 * 
 * \param val
 * \return +1 or -1.
 */
template <typename T> 
inline int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}