#pragma once

#include <vector>

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
 *	Range generator.
 * 
 * \param	first is the first element of the range.
 * \param	last is the last element of the range (not included).
 * \return	a vector with the range.
 */
inline vector<int> RangeGen(int first, int last) {
	int size = abs(last - first);
	vector<int> range(size);
	int idx{ first };

	if (last > first) {
		for (auto& value : range)
			value = idx++;
	}
	else {
		for (auto& value : range)
			value = idx--;
	}
	return range;
}
