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
	ErrorFunction() = delete;

	static map <ErrorFuncType, function<Real(const vector<Real>& , const vector<Real>&)>> EFunction;
	static map <ErrorFuncType, function<Real(const Real, const Real)>> EFunctionDer_RespectOutput;

private:
	static Real _sumOfSquares(const vector<Real>& NNOutput, const vector<Real>& targets);
	static Real _sumOfSquaresDer_RespectOutput(const Real neuronOutput, const Real target);

};

