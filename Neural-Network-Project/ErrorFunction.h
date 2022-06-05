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
enum class ErrorFuncType { SUMOFSQUARES, CROSSENTROPY, CROSSENTROPY_SOFTMAX};

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

	static map <ErrorFuncType, function<Real(const matrix<Real>& , const matrix<Real>&)>> EFunction;
	static map <ErrorFuncType, function<Real(const Real, const Real)>> EFunctionDer_RespectOutput;

private:
	static Real _sumOfSquares(const matrix<Real>& NNOutput, const matrix<Real>& targets);
	static Real _sumOfSquaresDer(const Real neuronOutput, const Real target);

	static Real _crossEntropy(const matrix<Real>& NNOutput, const matrix<Real>& targets);
	static Real _crossEntropy_softMax(const matrix<Real>& NNOutput, const matrix<Real>& targets);
	static Real _crossEntropyDer(const Real neuronOutput, const Real target);
	static Real _crossEntropyDer_softMax(const Real neuronOutput, const Real target);

};

