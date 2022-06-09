#include "ActivationFunction.h"
#include <cmath>

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

	case AFuncType::RELU:
		name = "ReLu";
		break;

	default:
		break;
	}

	return name;
}

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid(input); }},
	{AFuncType::IDENTITY, [](Real input) {return _identity(input); }},
	{AFuncType::RELU, [](Real input) {return _relu(input); }}
};

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction_Der = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid_Der(input); }},
	{AFuncType::IDENTITY, [](Real input) {return 1; }},
	{AFuncType::RELU, [](Real input) {return _relu_Der(input); }}
};


Real ActivationFunction::_sigmoid(const Real input) {
	return (Real)(1 / (1 + exp(-input)));
}

Real ActivationFunction::_identity(const Real input) {
	return input;
}

Real ActivationFunction::_relu(const Real input) {
	if (input < 0)
		return 0;
	else
		return input;
}

Real ActivationFunction::_sigmoid_Der(const Real input) {
	auto derivative = _sigmoid(input) * (1 - _sigmoid(input));
	return (derivative);
}

Real ActivationFunction::_relu_Der(const Real input) {
	if (input < 0)
		return 0;
	else
		return 1;
}
