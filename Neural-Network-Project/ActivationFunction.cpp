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

Real ActivationFunction::_sigmoid(const Real input) {
	return (Real)(1 / (1 + exp(-input)));
}

Real ActivationFunction::_identity(const Real input) {
	return input;
}
