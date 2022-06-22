#pragma once

#include <map>
#include <functional>
#include <string>
#include "Utility.h"

using std::map;
using std::function;
using std::string;

/**
	All error functions provided are reachable by these enums.
 */
enum class EFuncType { SUMOFSQUARES, CROSSENTROPY, CROSSENTROPY_SOFTMAX};

/**
 * This function returns the name of the error function.
 *
 * \param type is the error function's type.
 * \return A string with the error function's name.
 */
string NameOfErrorFuncType(const EFuncType type);

/**
 *
 * @brief The class provides a map used to call an error function by ErrorFuncType enum.
 *
 */
class ErrorFunction {
public:
	ErrorFunction() = delete;

	static map <EFuncType, function<Real(const mat_r& , const mat_r&)>> EFunction;
	static map <EFuncType, function<Real(const Real, const Real)>> EFunctionDer_RespectOutput;

private:
	static Real _sumOfSquares(const mat_r& NNOutput, const mat_r& target);
	static Real _sumOfSquaresDer(const Real neuronOutput, const Real target);

	static Real _crossEntropy(const mat_r& NNOutput, const mat_r& target);
	static Real _crossEntropy_softMax(const mat_r& NNOutput, const mat_r& target);
	static Real _crossEntropyDer(const Real neuronOutput, const Real target);
	static Real _crossEntropyDer_softMax(const Real neuronOutput, const Real target);

};

