#include "ActivationFunction.h"
#include <cmath>

Real Sigmoid(Real input) {
	return (Real)(1 / (1 + exp(-input)));
}

Real Identity(Real input) {
	return input;
}
