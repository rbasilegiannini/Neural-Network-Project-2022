#include "ErrorFunction.h"
#include <iostream>

map<ErrorFuncType, function<Real(const vector<Real>&, const vector<Real>&)>> ErrorFunction::EFunction = {

    { ErrorFuncType::SUMOFSQUARES, [](const vector<Real>& output, const vector<Real>& target) { 
        return _sumOfSquares(output, target); }}

};

map<ErrorFuncType, function<Real(const Real, const Real)>> ErrorFunction::EFunctionDer_RespectOutput = {

	{ ErrorFuncType::SUMOFSQUARES, [](const Real output, const Real target) {
		return _sumOfSquaresDer_RespectOutput(output, target); }}

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

Real ErrorFunction::_sumOfSquares(const vector<Real>& NNOutput, const vector<Real>& targets) {

	// Check to avoid out of range. TODO: use exception 
	if (targets.size() != NNOutput.size()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return 0;
	}
		
	Real error{ 0 };
	for (size_t idxOutput{ 0 }; idxOutput < NNOutput.size(); idxOutput++) 
		error += pow(NNOutput[idxOutput] - targets[idxOutput], 2);

	return error/2;
}

Real ErrorFunction::_sumOfSquaresDer_RespectOutput(const Real neuronOutput, const Real target) {
	return (neuronOutput - target);
}

