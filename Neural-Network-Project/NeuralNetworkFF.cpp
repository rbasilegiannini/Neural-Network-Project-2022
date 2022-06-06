#include "NeuralNetworkFF.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <boost/numeric/ublas/matrix_proxy.hpp>


using std::cout;
using std::endl;
using boost::numeric::ublas::subrange;

NeuralNetworkFF :: NeuralNetworkFF(
	const size_t inputDimension, 
	const vector<size_t>& nNeuronsPerLayer,
	const vector<AFuncType>& AFuncPerLayer ) 
	:
	_numNeuronsPerLayer{ nNeuronsPerLayer },
	_numLayers{ nNeuronsPerLayer.size() },
	_inputDimension{ inputDimension },
	_activationFunctionPerLayer{ AFuncPerLayer }
{
	// Resize vectors (each vector's element concerns a specific layer) 
	_weightsPerLayer.resize(_numLayers);
	_biasPerLayer.resize(_numLayers);
	_activationFunctionPerLayer.resize(_numLayers);

	size_t layer{ 0 };
	for (auto& weightMatrix : _weightsPerLayer) {
		srand(time(0));

		// Resize matrix and bias
		if (layer == 0)
			weightMatrix.resize(_numNeuronsPerLayer[layer], _inputDimension);
		else
			weightMatrix.resize(_numNeuronsPerLayer[layer], _numNeuronsPerLayer[layer - 1]);

		_biasPerLayer[layer].resize(_numNeuronsPerLayer[layer], 1); // It's a column vector

		// Random initialization of weights
		for (auto& w : weightMatrix.data()) 
			w = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1]

		// Random initialization of bias 
		for (auto& b : _biasPerLayer[layer].data())
			b = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1]

		// Set allParams matrices
		size_t numNeurons = _numNeuronsPerLayer[layer];		// All neurons
		size_t numParams = _weightsPerLayer[layer].size2() + 1;	// All weights + bias
		matrix<Real> params(numNeurons, numParams);

		//	First columns ~> biases
		for (const auto& i : RangeGen(0, params.size1()))
			params(i, 0) = _biasPerLayer[layer](i, 0);

		//	Remaining columns ~> weights
		for (const auto& i : RangeGen(0, params.size1())) {
			for (const auto& j : RangeGen(1, params.size2()))
				params(i, j) = _weightsPerLayer[layer](i, j - 1);
		}

		_allParamsPerLayer.push_back(params);

		layer++;
	}
}

void NeuralNetworkFF::SetActivationFunction(const size_t idxLayer, const AFuncType AFunctionType) {
	if (idxLayer < _activationFunctionPerLayer.size())
		_activationFunctionPerLayer[idxLayer] = AFunctionType;
}

void NeuralNetworkFF::SetWeights(const size_t idxLayer, const matrix<Real>& newWeights) {

	// Tool to partially or totally copy a matrix
	auto copyNewMatrix = [&](const size_t rows, const size_t cols){

		for (const auto& i : RangeGen(0, rows)) {
			for (const auto& j : RangeGen(0, cols)) {

				_weightsPerLayer[idxLayer](i, j) = newWeights(i, j);

				// j+1 because first column are reserved for biases
				_allParamsPerLayer[idxLayer](i, j + 1) = newWeights(i, j);	
			}
		}
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
	if (newWeights.size1() > _weightsPerLayer[idxLayer].size1()) {

		// Case: new.size2 > old.size2
		if (newWeights.size2() > _weightsPerLayer[idxLayer].size2()) 
			copyNewMatrix(_weightsPerLayer[idxLayer].size1(), _weightsPerLayer[idxLayer].size2());
		
		// Case: new.size2 <= old.size2
		else 
			copyNewMatrix(_weightsPerLayer[idxLayer].size1(), newWeights.size2());

		return;
	}

	// Checks to avoid out-of-range. Case: new.size1 <= old.size1
	if (newWeights.size1() <= _weightsPerLayer[idxLayer].size1()) {

		// Case: new.size2 > old.size2
		if (newWeights.size2() > _weightsPerLayer[idxLayer].size2()) 
			copyNewMatrix(newWeights.size1(), _weightsPerLayer[idxLayer].size2());

		// Case: new.size2 <= old.size2
		else 
			copyNewMatrix(newWeights.size1(), newWeights.size2());

		return;
	}
}

void NeuralNetworkFF::SetBias(const size_t idxLayer, vector<Real>& newBias) {
	
	// Checks to avoid out-of-range
	if (newBias.size() > _numNeuronsPerLayer[idxLayer]) {

		// Only first _numNeuronsPerLayer[idxLayer] elements are set
		for (const auto& i : RangeGen (0, _numNeuronsPerLayer[idxLayer])) {
			_biasPerLayer[idxLayer](i, 0) = newBias[i];
			_allParamsPerLayer[idxLayer](i, 0) = newBias[i];
		}
	}
	else {

		// Only first newBias.size() elements are set
		for (const auto& i : RangeGen(0, newBias.size())) { 
			_biasPerLayer[idxLayer](i, 0) = newBias[i];
			_allParamsPerLayer[idxLayer](i, 0) = newBias[i];
		}
	}
}

void NeuralNetworkFF::SetParamPerNeuron(const size_t idxLayer, const size_t idxNeuron, const size_t idxConnection, const Real newWeight) {
	_allParamsPerLayer[idxLayer](idxNeuron, idxConnection) = newWeight;

	if (idxConnection > 0 && idxConnection < _weightsPerLayer[idxLayer].size2())
		_weightsPerLayer[idxLayer](idxNeuron, idxConnection) = newWeight;
}

NetworkResult NeuralNetworkFF::ComputeNetwork(const vector<Real>& input) {

	NetworkResult netResult;

	// Check compatibility between the input's dimension and the network
	if (input.size() != _inputDimension) {
		cout << "[ERROR] The network is not compatible with the input's dimension." << endl;
	//	return vector<Real>(_numNeuronsPerLayer.back(), 0);
	}

	matrix<Real> activation;
	vector<Real> input_with_bias;
	input_with_bias.push_back(1); // First element is 1, because we have to consider also bias dimension

	for (const auto& i : input)
		input_with_bias.push_back(i);

	matrix<Real> outputLayer(input_with_bias.size(), 1);	// It's a column vector

	//	OutputLayer initialization
	for (const auto& i : RangeGen(0, input_with_bias.size()))
		outputLayer(i, 0) = input_with_bias[i];

	//	From the input layer to the last hidden layer
	for (const auto& layer : RangeGen(0, _numLayers - 1)) {
		activation = prod(_allParamsPerLayer[layer], outputLayer);

		// Compute the output layer
		outputLayer.resize(_numNeuronsPerLayer[layer] + 1, 1);
		outputLayer(0, 0) = 1;	//	For the bias
		
		for (const auto& idxNeuron : RangeGen(0, activation.size1()))
			outputLayer(idxNeuron+1, 0) = ActivationFunction::AFunction[_activationFunctionPerLayer[layer]](activation(idxNeuron, 0));

		// Fill the NetworkResult structure
		netResult.activationsPerLayer.push_back(activation);
		netResult.neuronsOutputPerLayer.push_back(
			subrange(outputLayer, 1, outputLayer.size1(), 0, outputLayer.size2())
		);
	}

	//	For the output layer: compute the output network
	activation = prod(_allParamsPerLayer[_numLayers - 1], outputLayer);
	outputLayer.resize(_numNeuronsPerLayer[_numLayers - 1], 1);

	for (const auto& idxNeuron : RangeGen(0, activation.size1()))
		outputLayer(idxNeuron, 0) = ActivationFunction::AFunction[_activationFunctionPerLayer[_numLayers - 1]](activation(idxNeuron, 0));

	netResult.activationsPerLayer.push_back(activation);
	netResult.neuronsOutputPerLayer.push_back(outputLayer);

	return netResult;
}

void NeuralNetworkFF::PrintNetwork() {

	// Print all index layer, weights and bias
	size_t idxLayer{ 0 };
	for (const auto& weightMatrix : _weightsPerLayer) {

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
		for (const auto& i : RangeGen(0, _biasPerLayer[idxLayer].size1()))
			cout << _biasPerLayer[idxLayer](i, 0) << endl;
		cout << endl;

		cout << "The activation function: " << 
			NameOfAFuncType(_activationFunctionPerLayer[idxLayer]) << endl;
		cout << endl;

		idxLayer++;
	}
}
