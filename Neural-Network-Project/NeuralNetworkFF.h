#pragma once
#include <boost/numeric/ublas/io.hpp>
#include "ActivationFunction.h"
#include "InvalidParametersException.h"

/**
 	This data structure contains the result of the network computation:
 	- the neurons' output (a column vector for each layer)
	- the neurons' activation (a column vector for each layer)
 */
struct NetworkResult {
	vector<mat_r> neuronsOutputPerLayer;
	vector<mat_r> activationsPerLayer;
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
	 * \param	AFuncPerLayer is the vector that contains the activation function for each layer.
	 */
	NeuralNetworkFF(
		const size_t inputDimension, 
		const vector<size_t>& nNeuronsPerLayer, 
		const vector<AFuncType>& AFuncPerLayer
		);

	size_t GetNumLayers()	{ return _numLayers; }
	size_t GetInputDimension()	{ return _inputDimension; }
	size_t GetNumNeurons_PerLayer(const size_t layer)	noexcept(false);
	mat_r GetWeights_PerLayer(const size_t layer)	noexcept(false);
	mat_r GetBias_PerLayer(const size_t layer)	noexcept(false);
	mat_r GetAllParam_PerLayer(const size_t layer)	noexcept(false);
	AFuncType GetAFunc_PerLayer(const size_t layer)	noexcept(false);

	void SetParam_PerNeuron(const size_t layer, const size_t neuron,const size_t conn, const Real newParam)	
		noexcept(false);
	void SetAllParam_PerLayer(const size_t layer, const mat_r& newMat) noexcept(false);
	void SetAFunc_PerLayer(const size_t layer, const AFuncType AFuncType) noexcept(false);
	void SetAllWeights(const size_t layer, const mat_r& newWeights)	noexcept(false);
	void SetAllBiases(const size_t layer, const vec_r& newBias)	noexcept(false);

	/**
	 *	This function computes the output of the network based on the current weights and bias.
	 * 
	 * \param	input is the vector with real numbers of the input. This vector must have the same dimension of 
	 *			the input of the network.
	 * \return	A vector of reals that contains the result of the computation.
	 */
	NetworkResult ComputeNetwork(const vec_r& input) noexcept(false);

	/**
	 *	This function initializes the neural network's parameteres with random value.
	 * 
	 * \param	l_ext is the left extreme (included)
	 * \param	r_ext ir the right extreme (included)
	 */
	void RandomInitialization(const int l_ext, const int r_ext);

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
	size_t _numLayers;
	size_t _inputDimension;
	vector<size_t> _numNeurons_PerLayer;   
	vector<mat_r> _weights_PerLayer;
	vector<mat_r> _bias_PerLayer;
	vector<AFuncType> _activationFunction_PerLayer;

	void _randomInit(const size_t layer, const int l_ext, const int r_ext) noexcept(false);
};

