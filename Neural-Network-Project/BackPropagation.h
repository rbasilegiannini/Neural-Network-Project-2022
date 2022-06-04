#pragma once

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"


/**
	This structure contains informations derived from the neural network:
	- the number of layers;
	- the number of neurons for each layer;
	- the activation's values for each neurons (a column vector for each layer);
	- the weights, without other parameteres (a matrix of real for each layer);
	- the activation function's derivative for each layer;
	- the neurons' output per layer (a column vector); 
	- the network's input.
 */
struct DataFromNetwork {
	const size_t numLayers;
	const vector<size_t> numNeuronsPerLayer;
	const vector<matrix<Real>> activationsPerLayer;
	const vector<matrix<Real>>  weightsPerLayer;
	const vector<AFuncType> AFunctionDerivativePerLayer;
	const vector<matrix<Real>>  neuronsOutputPerLayer;
	const vector<Real>	NNinput;
};

class BackPropagation {
public:
	BackPropagation() = delete;

	/**
	 * This function computes a vector of delta_i, where i = 1, ..., OutputLayer.
	 * Delta_i is a vector of delta values, one per neuron.
	 * 
	 * \param	dataNN is the data structure that contains informations derived from the neural network.
	 * \param	EFuncType is the type of the error function.
	 * \return	A vector of delta values.
	 */
	static vector<Real> BProp(const DataFromNetwork& dataNN, const ErrorFuncType EFuncType, const vector<Real>& targets);

};

