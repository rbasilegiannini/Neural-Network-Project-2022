#include "ErrorFunction.h"
#include <iostream>

map<ErrorFuncType, function<Real(const vector<Real>&, const vector<Real>&)>> ErrorFunction::EFunction = {

    { ErrorFuncType::SUMOFSQUARES, [](const vector<Real>& output, const vector<Real>& target) { 
        return _sumOfSquares(output, target); }}

};

string NameOfErrorFuncType(const ErrorFuncType type) {

	string name;
	switch (type)
	{
	case ErrorFuncType::SUMOFSQUARES:
		name = "SUMOFSQUARES";
		break;

	default:
		break;
	}

	return name;
}

Real ErrorFunction::_sumOfSquares(const vector<Real>& output, const vector<Real>& target) {

	// Check to avoid out of range. TODO: use exception 
	if (target.size() != output.size()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return 0;
	}
		
	Real error{ 0 };
	for (size_t idxOutput{ 0 }; idxOutput < output.size(); idxOutput++) 
		error += pow(output[idxOutput] - target[idxOutput], 2);

	return error;
}
