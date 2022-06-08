#pragma once

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

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
