#pragma once

#include <vector>
#include <map>
#include <functional>
#include <string>
#include "Utility.h"

using std::vector;
using std::map;
using std::function;
using std::string;

/**
 * @brief All error functions provided are reachable by these enums.
 */
enum class ErrorFuncType { SUMOFSQUARES };

/**
 * This function returns the name of the error function.
 *
 * \param type is the error function's type.
 * \return A string with the error function's name.
 */
string NameOfErrorFuncType(const ErrorFuncType type);

/**
 *
 * @brief The class provides a map used to call an error function by ErrorFuncType enum.
 *
 */
class ErrorFunction {
public:
	static map <ErrorFuncType, function<Real(const vector<Real>& , const vector<Real>&)>> EFunction;

private:
	ErrorFunction();
	static Real _sumOfSquares(const vector<Real>& output, const vector<Real>& target);

};

