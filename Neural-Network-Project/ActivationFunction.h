#pragma once

/**
 *
 * This module contains the activation functions used to compute the neurons' output. 
 *
 */

#include <map>
#include <functional>
#include <string>
#include "Utility.h"

using std::map;
using std::function;
using std::string;

/**
 * @brief All activation functions provided are reachable by these enums.
 */
enum class AFuncType {SIGMOID, IDENTITY};

/**
 * This function returns the name of the activation function.
 * 
 * \param type is the activation function's type.
 * \return A string with the activation function's name.
 */
string NameOfAFuncType(const AFuncType type);

/**
 *
 * @brief The class provides a map used to call an activation function by AFuncType enum.
 *
 */
class ActivationFunction {
public:
	static map <AFuncType, function<Real(const Real)>> AFunction;

private:
	ActivationFunction();
	static Real _sigmoid(const Real input);
	static Real _identity(const Real input);
};
