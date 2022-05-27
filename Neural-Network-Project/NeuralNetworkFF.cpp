#include "NeuralNetworkFF.h"
#include <cstdlib>
#include <ctime>

NeuralNetworkFF::NeuralNetworkFF(const size_t inputDimension, const vector<size_t> _nNeuronsPerLayer) :
	_numNeuronsPerLayer{ _nNeuronsPerLayer },
	_numLayers{ _numNeuronsPerLayer.size() }
{
	_weightPerLayer.resize(_numLayers);

	// Resize matrices and set weights for each hidden layer
	size_t idxLayer{ 0 };
	for (auto& weightMatrix : _weightPerLayer) {

		if (idxLayer == 0)
			weightMatrix.resize(_numNeuronsPerLayer[idxLayer], inputDimension);
		else
			weightMatrix.resize(_numNeuronsPerLayer[idxLayer], _numNeuronsPerLayer[idxLayer - 1]);

		// Random initialization of weights
		srand(time(0));
		for (auto i = 0; i < weightMatrix.size1(); i++) {
			for (auto j = 0; j < weightMatrix.size2(); j++) 
				weightMatrix(i, j) = (Real) (((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1] 
		}

		idxLayer++;
	}

	// Resize bias
	// Random initialization of bias 

	// Default activation functions: IDENTITY
	_activationFunctionPerLayer.resize(_numNeuronsPerLayer.size());
	for (auto& AFunc : _activationFunctionPerLayer)
		AFunc = AFuncType::IDENTITY;

}
