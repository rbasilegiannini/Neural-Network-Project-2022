#pragma once

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"


/**
	This structure contains informations derived from the neural network:
	- the number of layers;
	- the number of neurons for each layer;
	- the activation's values for each neurons (a column vector for each layer);
	- the parameters, not only weigths (a matrix of real for each layer);
	- the activation function's derivative for each layer.
	- the network outputs (a column vector)
 */
struct DataFromNetwork {
	const size_t numLayers;
	const vector<size_t> numNeuronsPerLayer;
	const vector<matrix<Real>> activationsPerLayer;
	const vector<matrix<Real>>  parameters;
	const vector<AFuncType> AFunctionDerivativePerLayer;
	const matrix<Real> NNOutputs;
};

class BackPropagation {
public:

	/**
	 * This function computes a vector of delta_i, where i = 1, ..., OutputLayer.
	 * Delta_i is a vector of delta values, one per neuron.
	 * 
	 * \param	dataNN is the data structure that contains informations derived from the neural network.
	 * \param	EFuncType is the type of the error function.
	 * \return	A vector of delta values.
	 */
	static vector<vector<Real>> BProp(const DataFromNetwork& dataNN, const ErrorFuncType EFuncType, const vector<Real>& targets);

};

