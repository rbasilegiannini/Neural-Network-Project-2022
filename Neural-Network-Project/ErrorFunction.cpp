#include "ErrorFunction.h"

using std::max;

map<EFuncType, function<Real(const mat_r&, const mat_r&)>> ErrorFunction::EFunction = {

    { EFuncType::SUMOFSQUARES, [](const mat_r& output, const mat_r& target) {
        return _sumOfSquares(output, target); }},

	{ EFuncType::CROSSENTROPY, [](const mat_r& output, const mat_r& target) {
		return _crossEntropy(output, target); }},

	{ EFuncType::CROSSENTROPY_SOFTMAX, [](const mat_r& output, const mat_r& target) {
		return _crossEntropy_softMax(output, target); }}
};

map<EFuncType, function<Real(const Real, const Real)>> ErrorFunction::EFunctionDer_RespectOutput = {

	{ EFuncType::SUMOFSQUARES, [](const Real output, const Real target) {
		return _sumOfSquaresDer(output, target); }},

	{ EFuncType::CROSSENTROPY, [](const Real output, const Real target) {
		return _crossEntropyDer(output, target); }},

	{ EFuncType::CROSSENTROPY_SOFTMAX, [](const Real output, const Real target) {
		return _crossEntropyDer_softMax(output, target); }}

};

string NameOfErrorFuncType(const EFuncType type) {

	string name;
	switch (type)
	{
	case EFuncType::SUMOFSQUARES:
		name = "SUMOFSQUARES";
		break;

	case EFuncType::CROSSENTROPY:
		name = "CROSSENTROPY";
		break;

	case EFuncType::CROSSENTROPY_SOFTMAX:
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
		error += (Real)pow(NNOutput(k,0) - targets(k,0), 2);

	return error/2;
}

Real ErrorFunction::_sumOfSquaresDer(const Real neuronOutput, const Real target) {
	return (neuronOutput - target);
}

Real ErrorFunction::_crossEntropy(const mat_r& NNOutput, const mat_r& targets) {

	//	Check to avoid out of range. 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return -1;
	}

	//	Summation
	Real summation{ 0 };
	for (const auto& k : RangeGen(0, NNOutput.size1())) {
		//	Check if the log domain is respected. 
		if (NNOutput(k,0) <= 0) {
			std::cout << "[ERROR] output is not compatible with Cross Entropy." << std::endl;
			return -1;
		}
		auto out = max(NNOutput(k, 0), (Real)1e-07);	// To avoid 0
		summation += targets(k,0) * log(out);
	}
	Real loss{ -summation };

	return loss;
}

Real ErrorFunction::_crossEntropy_softMax(const mat_r& NNOutput, const mat_r& targets) {
	//	Check to avoid out of range. 
	if (targets.size1() != NNOutput.size1()) {
		std::cout << "[ERROR] output is not compatible with targets." << std::endl;
		return -1;
	}

	//	Summation
	Real summation{ 0 };
	for (const auto& k : RangeGen(0, NNOutput.size1())) {
		// DEBUG
		auto sm = max(SoftMax(NNOutput, k), (Real)1e-07);	// To avoid 0
		summation += targets(k,0) * log(sm);
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
