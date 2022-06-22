#include "ActivationFunction.h"
#include <cmath>

//	Leaky ReLU hyperparam
constexpr Real alpha{ 0.02 };

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
		name = "ReLU";
		break;
	case AFuncType::LEAKYRELU:

		name = "LeakyReLU";
		break;
	default:
		break;
	}

	return name;
}

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid(input); }},
	{AFuncType::IDENTITY, [](Real input) {return _identity(input); }},
	{AFuncType::RELU, [](Real input) {return _relu(input); }},
	{AFuncType::LEAKYRELU, [](Real input) {return _LeakyRelu(input); }}
};

map <AFuncType, function<Real(Real)>> ActivationFunction::AFunction_Der = {
	{AFuncType::SIGMOID, [](Real input) {return _sigmoid_Der(input); }},
	{AFuncType::IDENTITY, [](Real input) {return 1; }},
	{AFuncType::RELU, [](Real input) {return _relu_Der(input); }},
	{AFuncType::LEAKYRELU, [](Real input) {return _LeakyRelu_Der(input); }}

};


Real ActivationFunction::_sigmoid(const Real input) {
	return (Real)(1 / (1 + exp(-input)));
}

Real ActivationFunction::_identity(const Real input) {
	return input;
}

Real ActivationFunction::_relu(const Real input) {
	return std::max((Real)0, input);
}

Real ActivationFunction::_sigmoid_Der(const Real input) {
	auto derivative = _sigmoid(input) * (1 - _sigmoid(input));
	return (derivative);
}

Real ActivationFunction::_relu_Der(const Real input) {

	if (input <= (Real)0)
		return (Real)0;
	else
		return (Real)1;
}

Real ActivationFunction::_LeakyRelu(const Real input) {

	if (input > 0)
		return input;
	else
		return alpha * input;

}

Real ActivationFunction::_LeakyRelu_Der(const Real input) {

	if (input > 0)
		return 1;
	else
		return alpha;

}
