#include "ActivationFunction.h"
#include <cmath>
#include "Utility.h"

string NameOfAFuncType(const AFuncType type) {

	string name;
	switch (type)
	{
	case AFuncType::IDENTITY:
		name = "IDENTITY";
		break;

	case AFuncType::SIGMOID:
		name = "SIGMOID";
		break;

	default:
		break;
	}

	return name;
}

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid(input); }},
	{AFuncType::IDENTITY, [](Real input) {return _identity(input); }}
};

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunctionDerivative = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoidDerivative(input); }},
	{AFuncType::IDENTITY, [](Real input) {return 1; }}
};


Real ActivationFunction::_sigmoid(const Real input) {
	return (Real)(1 / (1 + exp(-input)));
}

Real ActivationFunction::_identity(const Real input) {
	return input;
}

Real ActivationFunction::_sigmoidDerivative(const Real input) {
	auto derivative = _sigmoid(input) * (1 - _sigmoid(input));
	return (derivative);
}