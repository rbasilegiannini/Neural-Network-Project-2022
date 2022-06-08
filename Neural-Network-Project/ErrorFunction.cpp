#include "ErrorFunction.h"
#include "Utility.h" 
#include <iostream>

map<ErrorFuncType, function<Real(const mat_r&, const mat_r&)>> ErrorFunction::EFunction = {

    { ErrorFuncType::SUMOFSQUARES, [](const mat_r& output, const mat_r& target) {
        return _sumOfSquares(output, target); }},

	{ ErrorFuncType::CROSSENTROPY, [](const mat_r& output, const mat_r& target) {
		return _crossEntropy(output, target); }},

	{ ErrorFuncType::CROSSENTROPY_SOFTMAX, [](const mat_r& output, const mat_r& target) {
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

Real ErrorFunction::_sumOfSquares(const mat_r& NNOutput, const mat_r& targets) {

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

Real ErrorFunction::_crossEntropy(const mat_r& NNOutput, const mat_r& targets) {

	//	Check to avoid out of range. TODO: use exception 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return -1;
	}

	//	Summation
	Real summation{ 0 };
	for (const auto& k : RangeGen(0, NNOutput.size1())) {
		//	Check if the log domain is respected. TODO: use exception
		if (NNOutput(k,0) <= 0) {
			std::cout << "[ERROR] output is not compatible with Cross Entropy." << std::endl;
			return -1;
		}

		summation += targets(k,0) * log(NNOutput(k,0));
	}
	Real loss{ -summation };

	return loss;
}

Real ErrorFunction::_crossEntropy_softMax(const mat_r& NNOutput, const mat_r& targets) {
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
	return (-target / neuronOutput);
}

Real ErrorFunction::_crossEntropyDer_softMax(const Real softMax_output, const Real target) {
	if (target != 0 && target != 1)
		std::cout << "[ERROR] The target is not in one-hot encoding." << std::endl;

	return (softMax_output - target);
}
