#pragma once

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using boost::numeric::ublas::prod;
using boost::numeric::ublas::matrix;

using std::vector;

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
	Row vector of real type.
 */
typedef vector<Real> vec_r;

/**
 *	Range generator.
 * 
 * \param	first is the first element of the range.
 * \param	last is the last element of the range (not included).
 * \return	a vector with the range.
 */
inline vector<int> RangeGen(const int first, const int last) {
	int size = abs(last - first);
	vector<int> range(size);
	int idx{ first };

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

inline Real SoftMax(const mat_r& outputs, const size_t idxOutput) {

	Real summation{ 0 };
	for (const auto& h : RangeGen(0, outputs.size1())) 
		summation += exp(outputs(h, 0));
	Real SM = exp(outputs(idxOutput, 0)) / summation;

 	return SM;
}

inline Real row_by_column (const mat_r& row, const mat_r& column) {

	auto result = prod(row, column);
	return result(0, 0);
}

inline mat_r extract_column(const mat_r& mat, const size_t col) {

	mat_r extractor(mat.size1(), 1);
	for (const auto& i : RangeGen(0, mat.size1()))
		extractor(i, 0) = mat(i, col);

	return extractor;
}
