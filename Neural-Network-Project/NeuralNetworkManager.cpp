#include "NeuralNetworkManager.h"
#include "BackPropagation.h"

NeuralNetworkManager& NeuralNetworkManager::GetNNManager(const Hyperparameters& hyp) {
	static NeuralNetworkManager nnManager (hyp);
	return nnManager;
}

void NeuralNetworkManager::Run(const vec_r& input) {

	try {
		_input = input;
		_netResult = _neuralNetwork.ComputeNetwork(_input);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}
}

vec_r NeuralNetworkManager::ComputeGradE_PerSample(const EFuncType EFuncType, const vec_r& target)
noexcept(false) {
	if (_netResult.activationsPerLayer.empty())
		throw InvalidParametersException("[MANAGER] it's mandatory to perform a forward propagation step first.");

	vec_r gradE;

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

mat_r NeuralNetworkManager::GetAllParam_PerLayer(const size_t layer) {
	try {
		return _neuralNetwork.GetAllParam_PerLayer(layer);
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
		return mat_r();
	}
}

vector<size_t> NeuralNetworkManager::GetAllNumNeurons() {
	vector<size_t> AllNumNeurons;
	for (const auto& layer : RangeGen(0, GetNumLayers()))
		AllNumNeurons.push_back(_neuralNetwork.GetNumNeurons_PerLayer(layer));
	return AllNumNeurons;
}

vector<AFuncType> NeuralNetworkManager::GetAllAFuncType() {
	vector<AFuncType> AllFuncType;
	for (const auto& layer : RangeGen(0, GetNumLayers()))
		AllFuncType.push_back(_neuralNetwork.GetAFunc_PerLayer(layer));
	return AllFuncType;
}

void NeuralNetworkManager::RandomInitialization(const int l_ext, const int r_ext) {
	_neuralNetwork.RandomInitialization(l_ext, r_ext);
}

NeuralNetworkManager::NeuralNetworkManager(const Hyperparameters& hyp) 
	: _neuralNetwork { NeuralNetworkFF(hyp.inputDimension, hyp.numNeuronsPerLayer, hyp.AFuncPerLayer) }
{
}

