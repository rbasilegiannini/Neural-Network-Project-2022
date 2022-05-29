#include "NeuralNetworkFF.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

NeuralNetworkFF :: NeuralNetworkFF(const size_t inputDimension, const vector<size_t>& _nNeuronsPerLayer) :
	_numNeuronsPerLayer{ _nNeuronsPerLayer },
	_numLayers{ _nNeuronsPerLayer.size() },
	_inputDimension{ inputDimension }
{
	// Resize vectors (each vector's element concerns a specific layer) 
	_weightsPerLayer.resize(_numLayers);
	_biasPerLayer.resize(_numLayers);
	_activationFunctionPerLayer.resize(_numLayers);

	size_t idxLayer{ 0 };
	for (auto& weightMatrix : _weightsPerLayer) {

		// Resize matrix and bias
		if (idxLayer == 0)
			weightMatrix.resize(_numNeuronsPerLayer[idxLayer], _inputDimension);
		else
			weightMatrix.resize(_numNeuronsPerLayer[idxLayer], _numNeuronsPerLayer[idxLayer - 1]);

		_biasPerLayer[idxLayer].resize(_numNeuronsPerLayer[idxLayer], 1); // It's a column vector

		// Random initialization of weights
		srand(time(0));
		for (auto i = 0; i < weightMatrix.size1(); i++) {
			for (auto j = 0; j < weightMatrix.size2(); j++)
				weightMatrix(i, j) = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1] 
		}

		// Random initialization of bias 
		for (size_t i{ 0 }; i < _biasPerLayer[idxLayer].size1(); i++)
			_biasPerLayer[idxLayer](i, 0) = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1]

		idxLayer++;
	}

	// Default activation functions: SIGMOID
	for (auto& AFunc : _activationFunctionPerLayer)
		AFunc = AFuncType::SIGMOID;
}

void NeuralNetworkFF::SetActivationFunction(const size_t idxLayer, const AFuncType AFunctionType) {
	if (idxLayer < _activationFunctionPerLayer.size())
		_activationFunctionPerLayer[idxLayer] = AFunctionType;
}

void NeuralNetworkFF::SetWeights(const size_t idxLayer, const matrix<Real>& newWeights) {

	// Tool to partially or totally copy a matrix
	auto copyNewMatrix = [&](const size_t rows, const size_t cols){

			for (size_t i{ 0 }; i < rows; i++) {
				for (size_t j{ 0 }; j < cols; j++)
					_weightsPerLayer[idxLayer](i, j) = newWeights(i, j);
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
		for (size_t i{ 0 }; i < _numNeuronsPerLayer[idxLayer]; i++)
			_biasPerLayer[idxLayer](i, 0) = newBias[i];
	}
	else {

		// Only first newBias.size() elements are set
		for (size_t i{ 0 }; i < newBias.size(); i++)
			_biasPerLayer[idxLayer](i, 0) = newBias[i];
	}
}

vector<Real> NeuralNetworkFF::ComputeNetwork(const vector<Real>& input) {

	// Check compatibility between the input's dimension and the network
	if (input.size() != _inputDimension) {
		cout << "[ERROR] The network is not compatible with the input's dimension." << endl;
		return vector<Real>(_numNeuronsPerLayer.back(), 0);
	}

	vector<Real> result;
	matrix<Real> activation;
	matrix<Real> outputLayer (input.size(), 1);	// It's a column vector

	// OutputLayer initialization
	for (size_t i{ 0 }; i < outputLayer.size1(); i++)	
		outputLayer(i, 0) = input[i];

	for (size_t idxLayer{ 0 }; idxLayer < _numLayers; idxLayer++) {
		activation = prod(_weightsPerLayer[idxLayer], outputLayer) + _biasPerLayer[idxLayer];

		// Compute the output layer
		outputLayer.resize(_numNeuronsPerLayer[idxLayer], 1);
		for (size_t i{ 0 }; i < outputLayer.size1(); i++)
			outputLayer(i, 0) = ActivationFunction::AFunction[_activationFunctionPerLayer[idxLayer]](activation(i,0));
	}

	// Convert output matrix to std::vector
	for (size_t i{ 0 }; i < outputLayer.size1(); i++)
		result.push_back(outputLayer(i, 0));

	return result;
}

void NeuralNetworkFF::PrintNetwork() {

	// Print all weights and bias
	size_t idxLayer{ 0 };
	for (const auto& weightMatrix : _weightsPerLayer) {

		cout << "Dim: " << weightMatrix.size1() << "x" << weightMatrix.size2() << endl;
		cout << "Weights:" << endl;
		for (auto i = 0; i < weightMatrix.size1(); i++) {
			for (auto j = 0; j < weightMatrix.size2(); j++)
				cout << weightMatrix(i, j) << ' ';
			cout << endl;
		}
		cout << endl;

		cout << "Bias:" << endl;
		for (size_t i{ 0 }; i < _biasPerLayer[idxLayer].size1(); i++)
			cout << _biasPerLayer[idxLayer](i, 0) << endl;

		cout << endl;

		idxLayer++;
	}
}
