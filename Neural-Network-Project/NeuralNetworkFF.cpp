#include "NeuralNetworkFF.h"
#include <random>

using std::cout;
using std::endl;
using boost::numeric::ublas::subrange;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::random_device;

NeuralNetworkFF :: NeuralNetworkFF(const size_t inputDim, const vector<size_t>& nNeuronsPerLayer, const vector<AFuncType>& AFuncPerLayer ) {
	_numNeurons_PerLayer = nNeuronsPerLayer;
	_numLayers = nNeuronsPerLayer.size();
	_inputDimension = inputDim;
	_activationFunction_PerLayer = AFuncPerLayer;
	
	// Resize vectors (each vector's element concerns a specific layer) 
	_weights_PerLayer.resize(_numLayers);
	_bias_PerLayer.resize(_numLayers);
	_activationFunction_PerLayer.resize(_numLayers);

	for (const auto& layer : RangeGen(0, _numLayers)) {
		// Resize matrix and bias
		if (layer == 0)
			_weights_PerLayer[layer].resize(_numNeurons_PerLayer[layer], _inputDimension);
		else
			_weights_PerLayer[layer].resize(_numNeurons_PerLayer[layer], _numNeurons_PerLayer[layer - 1]);

		_bias_PerLayer[layer].resize(_numNeurons_PerLayer[layer], 1); // It's a column vector

		_randomInit(layer, -1, 1);
	}
}

size_t NeuralNetworkFF::GetNumNeurons_PerLayer(const size_t layer) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	return _numNeurons_PerLayer[layer];
}

mat_r NeuralNetworkFF::GetWeights_PerLayer(const size_t layer) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	return _weights_PerLayer[layer];
}

mat_r NeuralNetworkFF::GetBias_PerLayer(const size_t layer) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	return _bias_PerLayer[layer];
}

AFuncType NeuralNetworkFF::GetAFunc_PerLayer(const size_t layer) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	return _activationFunction_PerLayer[layer];
}

mat_r NeuralNetworkFF::GetAllParam_PerLayer(const size_t layer) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	// Set allParams matrices
	size_t numNeurons = _numNeurons_PerLayer[layer];		// All neurons
	size_t numParams = _weights_PerLayer[layer].size2() + 1;	// All weights + bias
	mat_r params(numNeurons, numParams);

	//	First columns ~> biases
	for (const auto& i : RangeGen(0, params.size1()))
		params(i, 0) = _bias_PerLayer[layer](i, 0);

	//	Remaining columns ~> weights
	for (const auto& i : RangeGen(0, params.size1())) 
		for (const auto& j : RangeGen(1, params.size2()))
			params(i, j) = _weights_PerLayer[layer](i, j - 1);
	
	return params;
}

void NeuralNetworkFF::SetAFunc_PerLayer(const size_t layer, const AFuncType AFunctionType) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	_activationFunction_PerLayer[layer] = AFunctionType;
}

void NeuralNetworkFF::SetAllWeights(const size_t layer, const mat_r& newWeights) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	// Tool to partially or totally copy a matrix
	auto copyNewMatrix = [&](const size_t rows, const size_t cols){
		for (const auto& i : RangeGen(0, rows)) 
			for (const auto& j : RangeGen(0, cols)) 
				_weights_PerLayer[layer](i, j) = newWeights(i, j);
	};

	/* 
	 *	Possible scenarios of out-of-range:
	 * 
		new.size1 > old.size1;  iterate over old.size1
		new.size2 > old.size2;  iterate over old.size2
	
		new.size1 > old.size1;  iterate over old.size1
		new.size2 < old.size2;  iterate over new.size2

		new.size1 < old.size1;  iterate over new.size1
		new.size2 > old.size2;  iterate over old.size2

		new.size1 < old.size1;  iterate over new.size1
		new.size2 < old.size2;  iterate over new.size2
	*
	*/

	// Checks to avoid out-of-range. Case: new.size1 > old.size1
	if (newWeights.size1() > _weights_PerLayer[layer].size1()) {

		// Case: new.size2 > old.size2
		if (newWeights.size2() > _weights_PerLayer[layer].size2()) 
			copyNewMatrix(_weights_PerLayer[layer].size1(), _weights_PerLayer[layer].size2());
		
		// Case: new.size2 <= old.size2
		else 
			copyNewMatrix(_weights_PerLayer[layer].size1(), newWeights.size2());

		return;
	}

	// Checks to avoid out-of-range. Case: new.size1 <= old.size1
	if (newWeights.size1() <= _weights_PerLayer[layer].size1()) {

		// Case: new.size2 > old.size2
		if (newWeights.size2() > _weights_PerLayer[layer].size2()) 
			copyNewMatrix(newWeights.size1(), _weights_PerLayer[layer].size2());

		// Case: new.size2 <= old.size2
		else 
			copyNewMatrix(newWeights.size1(), newWeights.size2());

		return;
	}
}

void NeuralNetworkFF::SetAllBiases(const size_t layer, const vec_r& newBias) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	// Checks to avoid out-of-range
	if (newBias.size() > _numNeurons_PerLayer[layer]) {
		// Only first _numNeuronsPerLayer[layer] elements are set
		for (const auto& i : RangeGen (0, _numNeurons_PerLayer[layer]))
			_bias_PerLayer[layer](i, 0) = newBias[i];
	}
	else {
		// Only first newBias.size() elements are set
		for (const auto& i : RangeGen(0, newBias.size())) 
			_bias_PerLayer[layer](i, 0) = newBias[i];
	}
}

void NeuralNetworkFF::SetParam_PerNeuron(const size_t layer, const size_t neuron, const size_t conn, const Real newParam) 
noexcept(false) {

	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");
	if (neuron >= _numNeurons_PerLayer[layer])
		throw InvalidParametersException("[NNFF] neuron must be in [0, ..., NumNeuron(layer)-1].");

	//	In case of bias
	if (conn == 0)
		_bias_PerLayer[layer](neuron, conn) = newParam;
	else
		if (conn < _weights_PerLayer[layer].size2() + 1)
			_weights_PerLayer[layer](neuron, conn - 1) = newParam;
		else
			throw InvalidParametersException("[NNFF] param doesn't exist.");

}

void NeuralNetworkFF::SetAllParam_PerLayer(const size_t layer, const mat_r& newMat) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");
	if (_numNeurons_PerLayer[layer] != newMat.size1())
		throw InvalidParametersException("[NNFF] the new MatParam is not compatible with the layer (number of neurons).");
	if (_weights_PerLayer[layer].size2() + 1 != newMat.size2())
		throw InvalidParametersException("[NNFF] the new MatParam is not compatible with the layer (number of connections).");

	for (const auto& neuron : RangeGen(0, _numNeurons_PerLayer[layer])) {
		_bias_PerLayer[layer](neuron, 0) = newMat(neuron, 0);

		for (const auto& conn : RangeGen(0, _weights_PerLayer[layer].size2()))
			_weights_PerLayer[layer](neuron, conn) = newMat(neuron, conn + 1);
	}
}

NetworkResult NeuralNetworkFF::ComputeNetwork(const vec_r& input) noexcept(false) {

	NetworkResult netResult;

	// Check compatibility between the input's dimension and the network
	if (input.size() != _inputDimension) 
		throw InvalidParametersException("[NNFF] input's dimension must be compatible with the network.");

	mat_r activation;
	vec_r input_with_bias;
	input_with_bias.push_back(1); // First element is 1, because we have to consider also bias dimension

	for (const auto& i : input)
		input_with_bias.push_back(i);

	mat_r outputLayer(input_with_bias.size(), 1);	// It's a column vector

	//	OutputLayer initialization
	for (const auto& i : RangeGen(0, input_with_bias.size()))
		outputLayer(i, 0) = input_with_bias[i];

	//	From the input layer to the last hidden layer
	for (const auto& layer : RangeGen(0, _numLayers - 1)) {
		activation = prod(GetAllParam_PerLayer(layer), outputLayer);

		// Compute the output layer
		outputLayer.resize(_numNeurons_PerLayer[layer] + 1, 1);
		outputLayer(0, 0) = 1;	//	For the bias
		
		for (const auto& idxNeuron : RangeGen(0, activation.size1()))
			outputLayer(idxNeuron+1, 0) = ActivationFunction::AFunction[_activationFunction_PerLayer[layer]](activation(idxNeuron, 0));

		// Fill the NetworkResult structure
		netResult.activationsPerLayer.push_back(activation);
		netResult.neuronsOutputPerLayer.push_back(
			subrange(outputLayer, 1, outputLayer.size1(), 0, outputLayer.size2())
		);
	}

	//	For the output layer: compute the output network
	activation = prod(GetAllParam_PerLayer(_numLayers - 1), outputLayer);

	outputLayer.resize(_numNeurons_PerLayer[_numLayers - 1], 1);

	for (const auto& idxNeuron : RangeGen(0, activation.size1()))
		outputLayer(idxNeuron, 0) = ActivationFunction::AFunction[_activationFunction_PerLayer[_numLayers - 1]](activation(idxNeuron, 0));

	netResult.activationsPerLayer.push_back(activation);
	netResult.neuronsOutputPerLayer.push_back(outputLayer);

	return netResult;
}

void NeuralNetworkFF::RandomInitialization(const int l_ext, const int r_ext) {

	try {
		for (const auto& layer : RangeGen(0, _numLayers))
			_randomInit(layer, l_ext, r_ext);
	}
	catch (InvalidParametersException e) {
		cout << e.getErrorMessage() << endl;
	}
}

void NeuralNetworkFF::PrintNetwork() {

	// Print all index layer, weights and bias
	size_t idxLayer{ 0 };
	for (const auto& weightMatrix : _weights_PerLayer) {

		string layer;

		if (idxLayer == _numLayers - 1)
			layer = "Output";
		else
			layer = std::to_string(idxLayer + 1);

		cout << "***************";
		cout << "Layer: " << layer;
		cout << "***************" << endl;

		cout << "W dimensions: " << weightMatrix.size1() << "x" << weightMatrix.size2() << ", ";
		cout << "weights: " << endl;
		for (const auto& i : RangeGen(0, weightMatrix.size1())) {
			for (const auto& j : RangeGen(0, weightMatrix.size2()))
				cout << weightMatrix(i, j) << ' ';
			cout << endl;
		}
		cout << endl;

		cout << "Bias:" << endl;
		for (const auto& i : RangeGen(0, _bias_PerLayer[idxLayer].size1()))
			cout << _bias_PerLayer[idxLayer](i, 0) << endl;
		cout << endl;

		cout << "The activation function: " << 
			NameOfAFuncType(_activationFunction_PerLayer[idxLayer]) << endl;
		cout << endl;

		idxLayer++;
	}
}

void NeuralNetworkFF::_randomInit(const size_t layer, const int l_ext, const int r_ext) noexcept(false) {
	if (layer >= _numLayers)
		throw InvalidParametersException("[NNFF] layer must be in [0, ..., NetworkLayer-1].");

	if (l_ext >= r_ext)
		throw InvalidParametersException("[NNFF] l_ext must be less than r_ext.");

	random_device rd;
	default_random_engine rgn{ rd() };
	uniform_real_distribution<Real> uniformDist((Real)l_ext, (Real)r_ext);

//	srand(time(0));
	//	Init biases
	for (auto& b : _bias_PerLayer[layer].data())
		//b = (Real)(((rand() % ((r_ext - l_ext) * 1000)) + ((l_ext * 1000) + 1)) * 0.001);	// Random value in [l_ext, l_ext]
		b = uniformDist(rgn);

	//	Init weights
	for (auto& w : _weights_PerLayer[layer].data())
		//w = (Real)(((rand() % ((r_ext - l_ext) * 1000)) + ((l_ext * 1000) + 1)) * 0.001);	// Random value in [l_ext, l_ext]
		w = uniformDist(rgn);

}
