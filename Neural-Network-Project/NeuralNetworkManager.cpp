#include "NeuralNetworkManager.h"
#include "BackPropagation.h"

NeuralNetworkManager& NeuralNetworkManager::GetNNManager(const Hyperparameters& hyp) {
	static NeuralNetworkManager nnManager (hyp);
	return nnManager;
}

void NeuralNetworkManager::Run(const vector<Real>& input) {

	try {
		_input = input;
		_netResult = _neuralNetwork.ComputeNetwork(_input);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
}

vector<Real> NeuralNetworkManager::ComputeGradE_PerSample(const ErrorFuncType EFuncType, const vector<Real>& target) 
throw (InvalidParametersException) {
	if (_netResult.activationsPerLayer.empty())
		throw InvalidParametersException("[MANAGER] it's mandatory to perform a forward propagation step first.");

	vector<Real> gradE;

	//	Fill DataFromNetwork structure
	vector<size_t> allNeuronsNumber;
	vector<AFuncType> allAFunc;
	vector<mat_r> weightsPerLayer;

	for (const auto& layer : RangeGen(0, _neuralNetwork.GetNumLayers())) {
		allNeuronsNumber.push_back(_neuralNetwork.GetNumNeurons_PerLayer(layer));
		allAFunc.push_back(_neuralNetwork.GetAFunc_PerLayer(layer));
		weightsPerLayer.push_back(_neuralNetwork.GetWeights_PerLayer(layer));
	}

	DataFromNetwork dataNN{
		_neuralNetwork.GetNumLayers(),
		allNeuronsNumber,
		_netResult.activationsPerLayer,
		weightsPerLayer,
		allAFunc,
		_netResult.neuronsOutputPerLayer,
		_input
	};

	//	Convert targets' vector to matrix
	mat_r matTarget (target.size(), 1);
	for (const auto& k : RangeGen(0, target.size()))
		matTarget(k, 0) = target[k];
	
	try {
		gradE = BackPropagation(dataNN, EFuncType, matTarget);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
		
	return gradE;
}

void NeuralNetworkManager::SetAllParam_PerLayer(const size_t layer, const mat_r& newMat) {
	try {
		_neuralNetwork.SetAllParam_PerLayer(layer, newMat);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
}

void NeuralNetworkManager::SetAFunc_PerLayer(const size_t layer, const AFuncType AFunctionType) {

	try {
		_neuralNetwork.SetAFunc_PerLayer(layer, AFunctionType);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
}

void NeuralNetworkManager::ResetHyperparameters(const Hyperparameters& hyp) {
	_neuralNetwork = NeuralNetworkFF(hyp.inputDimension, hyp.numNeuronsPerLayer, hyp.AFuncPerLayer);
}

matrix<Real> NeuralNetworkManager::GetAllParam_PerLayer(const size_t layer) {
	try {
		return _neuralNetwork.GetAllParam_PerLayer(layer);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
}

void NeuralNetworkManager::RandomInitialization(const int l_ext, const int r_ext) {
	_neuralNetwork.RandomInitialization(l_ext, r_ext);
}

NeuralNetworkManager::NeuralNetworkManager(const Hyperparameters& hyp) 
	: _neuralNetwork { NeuralNetworkFF(hyp.inputDimension, hyp.numNeuronsPerLayer, hyp.AFuncPerLayer) }
{
}

