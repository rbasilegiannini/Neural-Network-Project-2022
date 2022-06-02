#pragma once
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "ActivationFunction.h"

using std::vector;
using boost::numeric::ublas::matrix;


/**
 	This data structure contains the result of the network computation:
 	- the neurons' output (a column vector for each layer)
	- the neurons' activation (a column vector for each layer)
 */
struct NetworkResult {
	vector<matrix<Real>> neuronsOutputPerLayer;
	vector<matrix<Real>> activationsPerLayer;
};

/**
 *
 *	\brief Data structure of a multilayer Neural Network Feed Forward.
 *
 */
class NeuralNetworkFF {
public:

	/**
	 *	Create a NN FF with random weights and bias. 
	 *	For all layers the default activation function is the sigmoid.
	 * 
	 * \param	inputDimension is the input's dimension accepted by the network.
	 * \param	nNeuronsPerLayer is the vector that contains the number of neurons for each layer.
	 */
	NeuralNetworkFF(const size_t inputDimension, const vector<size_t>& nNeuronsPerLayer);

	size_t GetNumLayers() { return _numLayers; }
	size_t GetNumNeuronsPerLayer(const size_t idxLayer) { return _numNeuronsPerLayer[idxLayer]; }
	matrix<Real> GetWeightsPerLayer(const size_t idxLayer) { return _weightsPerLayer[idxLayer]; }
	matrix<Real> GetBiasPerLayer(const size_t idxLayer) { return _biasPerLayer[idxLayer]; }
	AFuncType GetAFuncPerLayer(const size_t idxLayer) { return _activationFunctionPerLayer[idxLayer]; }

	void SetActivationFunction(const size_t idxLayer, const AFuncType AFunctionType);
	void SetWeights(const size_t idxLayer, const matrix<Real>& newWeights);
	void SetBias(const size_t idxLayer, vector<Real>& newBias);
	void SetWeightPerNeuron(const size_t idxLayer, const size_t idxNeuron, const size_t idxConnection, const Real newWeight);

	/**
	 *	This function computes the output of the network based on the current weights and bias.
	 * 
	 * \param	input is the vector with real numbers of the input. This vector must have the same dimension of 
	 *			the input of the network.
	 * \return	A vector of reals that contains the result of the computation.
	 */

	//vector<Real> ComputeNetwork(const vector<Real>& input);
	NetworkResult ComputeNetwork(const vector<Real>& input);

	/**
		This function prints, for each layer:
		- The index layer (from the first hidden layer to the output layer);
		- The dimension and the values of the weights matrix;
		- The bias column;
		- The activation function.
	 * 
	 */
	void PrintNetwork();

private:
	const size_t _numLayers;
	const size_t _inputDimension;
	const vector<size_t> _numNeuronsPerLayer;   
	vector<matrix<Real>> _weightsPerLayer;
	vector<matrix<Real>> _biasPerLayer;
	vector<AFuncType> _activationFunctionPerLayer;
};

