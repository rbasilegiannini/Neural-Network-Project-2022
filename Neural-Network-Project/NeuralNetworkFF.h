#pragma once
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "ActivationFunction.h"


using std::vector;
using boost::numeric::ublas::matrix;

/**
 *
 * @brief Data structure of a multilayer Neural Network Feed Forward.
 *
 */
class NeuralNetworkFF {
public:
	NeuralNetworkFF(const size_t inputDimension, const vector<size_t>& _nNeuronsPerHiddenLayer);

	size_t GetNumLayers() { return _numLayers; }
	size_t GetNumNeuronsPerLayer(const size_t idxLayer) { return _numNeuronsPerLayer[idxLayer]; }
	matrix<Real> GetWeightsPerLayer(const size_t idxLayer) { return _weightsPerLayer[idxLayer]; }
	matrix<Real> GetBiasPerLayer(const size_t idxLayer) { return _biasPerLayer[idxLayer]; }

	void SetActivationFunction(const size_t idxLayer, const AFuncType AFunctionType);
	void SetWeights(const size_t idxLayer, const matrix<Real>& newWeights);
	void SetBias(const size_t idxLayer, vector<Real>& newBias);

	vector<Real> ComputeNetwork(const vector<Real>& input);
	void PrintNetwork();

private:
	const size_t _numLayers;
	const size_t _inputDimension;
	const vector<size_t> _numNeuronsPerLayer;   
	vector<matrix<Real>> _weightsPerLayer;
	vector<matrix<Real>> _biasPerLayer;
	vector<AFuncType> _activationFunctionPerLayer;
};
