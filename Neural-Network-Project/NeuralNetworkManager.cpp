#include "NeuralNetworkManager.h"
#include "BackPropagation.h"

NeuralNetworkManager& NeuralNetworkManager::GetNNManager(const Hyperparameters& hyp) {
	static NeuralNetworkManager nnManager (hyp);
	return nnManager;
}

void NeuralNetworkManager::Run(const vector<Real>& input) throw (InvalidParametersException) {
	if (input.size() != _neuralNetwork.GetInputDimension())
		throw InvalidParametersException("[MANAGER] input's dimension must be compatible with the network.");

	_input = input;
	_netResult = _neuralNetwork.ComputeNetwork(_input);
}

vector<Real> NeuralNetworkManager::ComputeGradE_PerSample(const ErrorFuncType EFuncType, const vector<Real>& targets) {
	
	vector<Real> gradE;

	//	Fill DataFromNetwork structure
	vector<size_t> allNeuronsNumber;
	vector<AFuncType> allAFunc;
	vector<matrix<Real>> weightsPerLayer;

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
	matrix<Real> matTarget (targets.size(), 1);
	for (const auto& k : RangeGen(0, targets.size()))
		matTarget(k, 0) = targets[k];
	
	gradE = BackPropagation(dataNN, EFuncType, matTarget);

	return gradE;
}

void NeuralNetworkManager::SetAFunc_PerLayer(const size_t layer, const AFuncType AFunctionType) throw (InvalidParametersException) {
	if (layer >= _neuralNetwork.GetNumLayers())
		throw InvalidParametersException("[MANAGER] layer must be in [0, ..., NetworkLayer-1].");

	_neuralNetwork.SetAFunc_PerLayer(layer, AFunctionType);
}

NeuralNetworkManager::NeuralNetworkManager(const Hyperparameters& hyp) 
	: _neuralNetwork { NeuralNetworkFF(hyp.inputDimension, hyp.numNeuronsPerLayer, hyp.AFuncPerLayer) }
{
}

