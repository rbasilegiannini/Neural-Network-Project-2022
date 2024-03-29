#pragma once

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"

/**
	This data structure contains the hyperparameters of the network:
	- the input's dimension;
	- the number of neurons for each layer;
	- the activation function for each layer.
 */
struct Hyperparameters {
	size_t inputDimension { 1 };
	vector<size_t> numNeuronsPerLayer{ 1 };
	vector<AFuncType> AFuncPerLayer {AFuncType::SIGMOID};
};

/**
 *
 *	\brief	This module represents the neural network manager through which user can use it.
 *			It's possibile to instance only one manager.
 *
 */
class NeuralNetworkManager {
public:
	/**
	 *	This function creates the NNManager and the managed neural network.
	 * 
	 * \param	hyp represents the hyperparameters of the network.
	 * \return	The NeuralNetworkManager object (singleton).
	 */
	static NeuralNetworkManager& GetNNManager(const Hyperparameters& hyp);
	
	/**
	 *	This method performs the Forward Propagation step.
	 * 
	 * \param input
	 * \return 
	 */
	void Run(const vec_r& input);

	/**
	 *	This method computes the gradient of the error function for a sample.
	 * 
	 * \param	EFuncType is the error function's type of which we want to compute the gradient.
	 * \param	target
	 * \return	A vector cointaining the partial derivatives of the error function.
	 */
	vec_r ComputeGradE_PerSample(const EFuncType EType, const vec_r& target) noexcept(false);

	/**
	 *	This function initializes the neural network's parameteres with random value.
	 *
	 * \param	l_ext is the left extreme (included)
	 * \param	r_ext ir the right extreme (included)
	 */
	void RandomInitialization(const float l_ext, const float r_ext);

	/**
	 *	This function changes the NN's hyperparameters. The parameteres will be reinitialized.
	 * 
	 * \param	hyp represents the new hyperparameters.
	 */
	void ResetHyperparameters(const Hyperparameters& hyp);

	/**
	 *	This getter returns the matrix of parameters for a given layer.
	 * 
	 * \param layer
	 * \return	A matrix with all paramateres for a given layer.
	 */
	mat_r GetAllParam_PerLayer(const size_t layer);

	size_t GetNumLayers() { return _neuralNetwork.GetNumLayers(); }
	mat_r GetNetworkOutput() { return _netResult.neuronsOutputPerLayer.back(); }
	vector<size_t> GetAllNumNeurons();
	vector<AFuncType> GetAllAFuncType();

	/**
	 *	This setter sets the matrix of parameteres for a given layer.
	 * 
	 * \param layer
	 * \param newMat
	 */
	void SetAllParam_PerLayer(const size_t layer, const mat_r& newMat);

	/**
	 *	This setter sets the activation functions for a given layer.
	 * 
	 * \param layer
	 * \param AFunctionType
	 */
	void SetAFunc_PerLayer(const size_t layer, const AFuncType AFunctionType);
	
	void PrintNetwork() { _neuralNetwork.PrintNetwork(); }

	//	Test function
	NeuralNetworkFF GetNet() { return _neuralNetwork; }
private:
	//	Singleton
	NeuralNetworkManager() = default;
	NeuralNetworkManager(const Hyperparameters& hyp);
	NeuralNetworkManager(const NeuralNetworkManager& oth) = delete;
	NeuralNetworkManager& operator = (const NeuralNetworkManager& oth) = delete;

	NeuralNetworkFF _neuralNetwork;
	NetworkResult _netResult;
	vec_r _input;
};

