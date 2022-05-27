#pragma once

/**
 *
 * This class contains the activation functions used to compute the neurons' output. 
 *
 */

#include <map>
#include <functional>
#include "Utility.h"

using std::map;
using std::function;

/**
 * @brief All activation functions provided are reachable by these enums.
 */
enum class AFuncType {SIGMOID, IDENTITY};

/**
 *
 * @brief The class provides a map used to call an activation function by AFuncType enum.
 *
 */
class ActivationFunction {
public:
	static map <AFuncType, function<Real(Real)>> AFunction;

private:
	ActivationFunction();
	static Real _sigmoid(const Real input);
	static Real _identity(const Real input);
};
