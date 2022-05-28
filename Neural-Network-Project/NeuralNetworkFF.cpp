#include "NeuralNetworkFF.h"
#include <cstdlib>
#include <ctime>

NeuralNetworkFF :: NeuralNetworkFF(const size_t inputDimension, const vector<size_t>& _nNeuronsPerLayer) :
	_numNeuronsPerLayer{ _nNeuronsPerLayer },
	_numLayers{ _nNeuronsPerLayer.size() } 
{
	// Resize vectors (each vector's element concerns a specific layer) 
	_weightPerLayer.resize(_numLayers);
	_biasPerLayer.resize(_numLayers);
	_activationFunctionPerLayer.resize(_numLayers);

	size_t idxLayer{ 0 };
	for (auto& weightMatrix : _weightPerLayer) {

		// Resize matrix and bias
		if (idxLayer == 0)
			weightMatrix.resize(_numNeuronsPerLayer[idxLayer], inputDimension);
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

vector<Real> NeuralNetworkFF::ComputeNetwork(const vector<Real>& input) {

	vector<Real> result;
	matrix<Real> activation;
	matrix<Real> outputLayer (input.size(), 1);	// It's a column vector

	// OutputLayer initialization
	for (size_t i{ 0 }; i < outputLayer.size1(); i++)	
		outputLayer(i, 0) = input[i];

	for (size_t idxLayer{ 0 }; idxLayer < _numLayers; idxLayer++) {
		activation = prod(_weightPerLayer[idxLayer], outputLayer) + _biasPerLayer[idxLayer];

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
