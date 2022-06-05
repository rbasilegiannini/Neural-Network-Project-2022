#include "ErrorFunction.h"
#include "Utility.h" 
#include <iostream>

map<ErrorFuncType, function<Real(const matrix<Real>&, const matrix<Real>&)>> ErrorFunction::EFunction = {

    { ErrorFuncType::SUMOFSQUARES, [](const matrix<Real>& output, const matrix<Real>& target) {
        return _sumOfSquares(output, target); }},

	{ ErrorFuncType::CROSSENTROPY, [](const matrix<Real>& output, const matrix<Real>& target) {
		return _crossEntropy(output, target); }},

	{ ErrorFuncType::CROSSENTROPY_SOFTMAX, [](const matrix<Real>& output, const matrix<Real>& target) {
		return _crossEntropy_softMax(output, target); }}
};

map<ErrorFuncType, function<Real(const Real, const Real)>> ErrorFunction::EFunctionDer_RespectOutput = {

	{ ErrorFuncType::SUMOFSQUARES, [](const Real output, const Real target) {
		return _sumOfSquaresDer(output, target); }},

	{ ErrorFuncType::CROSSENTROPY, [](const Real output, const Real target) {
		return _crossEntropyDer(output, target); }},

	{ ErrorFuncType::CROSSENTROPY_SOFTMAX, [](const Real output, const Real target) {
		return _crossEntropyDer_softMax(output, target); }}

};

string NameOfErrorFuncType(const ErrorFuncType type) {

	string name;
	switch (type)
	{
	case ErrorFuncType::SUMOFSQUARES:
		name = "SUMOFSQUARES";
		break;

	case ErrorFuncType::CROSSENTROPY:
		name = "CROSSENTROPY";
		break;

	case ErrorFuncType::CROSSENTROPY_SOFTMAX:
		name = "CROSSENTROPY_SOFTMAX";
		break;

	default:
		break;
	}

	return name;
}

Real ErrorFunction::_sumOfSquares(const matrix<Real>& NNOutput, const matrix<Real>& targets) {

	// Check to avoid out of range. TODO: use exception 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return 0;
	}
		
	Real error{ 0 };
	for (const auto& k : RangeGen (0, NNOutput.size1()))
		error += pow(NNOutput(k,0) - targets(k,0), 2);

	return error/2;
}

Real ErrorFunction::_sumOfSquaresDer(const Real neuronOutput, const Real target) {
	return (neuronOutput - target);
}

Real ErrorFunction::_crossEntropy(const matrix<Real>& NNOutput, const matrix<Real>& targets) {

	//	Check to avoid out of range. TODO: use exception 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return -1;
	}

	//	Summation
	Real summation{ 0 };
	for (const auto& k : RangeGen(0, NNOutput.size1())) {
		//	Check if the log domain is respected. TODO: use exception
		if (NNOutput(k,0) >= 1) {
			std::cout << "[ERROR] output is not compatible with Cross Entropy." << std::endl;
			return -1;
		}

		summation += targets(k,0) * log(1 - NNOutput(k,0));
	}
	Real loss{ -summation };

	return loss;
}

Real ErrorFunction::_crossEntropy_softMax(const matrix<Real>& NNOutput, const matrix<Real>& targets) {
	//	Check to avoid out of range. TODO: use exception 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return -1;
	}

	//	Summation
	Real summation{ 0 };
	for (const auto& k : RangeGen(0, NNOutput.size1())) {
		summation += targets(k,0) * log(SoftMax(NNOutput, k));
	}

	Real loss{ -summation };

	return loss;
}

Real ErrorFunction::_crossEntropyDer(const Real neuronOutput, const Real target) {
	return (target / (neuronOutput - target));
}

Real ErrorFunction::_crossEntropyDer_softMax(const Real softMax_output, const Real target) {
	return (softMax_output - target);
}
