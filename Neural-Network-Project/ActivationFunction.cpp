#include "ActivationFunction.h"
#include <cmath>
#include "Utility.h"


Real ActivationFunction::_sigmoid(const Real input) {
return (Real)(1 / (1 + exp(-input)));
}

Real ActivationFunction::_identity(const Real input) {
	return input;
}

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid(input); }},
	{AFuncType::IDENTITY, [](Real input) {return _identity(input); }}
};

