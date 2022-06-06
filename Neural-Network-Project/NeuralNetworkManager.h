#pragma once

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"
#include "InvalidParametersException.h"

/**
	This data structure contains the hyperparameters of th network:
	- the input's dimension;
	- the number of neurons for each layer;
	- the activation function for each layer.
 */
struct Hyperparameters {
	const size_t inputDimension;
	const vector<size_t> numNeuronsPerLayer;
	const vector<AFuncType> AFuncPerLayer;
};

/**
 *
 *	\brief	This module represents the neural network manager through which user can use it.
 *			It's possibile to instance only one manager.
 *
 */
class NeuralNetworkManager {
public:
	static NeuralNetworkManager& GetNNManager(const Hyperparameters& hyp);
	
	/**
	 *	This method performs the Forward Propagation step.
	 * 
	 * \param input
	 * \return 
	 */
	void Run(const vector<Real>& input) throw (InvalidParametersException);

	/**
	 *	This method compute the gradient of the error function for a sample.
	 * 
	 * \param	EFuncType is the error function's type of which we want to compute the gradient.
	 * \param	target
	 * \return	A vector cointaining the partial derivatives of the error function.
	 */
	vector<Real> ComputeGradE_PerSample(const ErrorFuncType EFuncType, const vector<Real>& target);

	void SetAFunc_PerLayer(const size_t layer, const AFuncType AFunctionType) throw (InvalidParametersException);


private:
	//	Singleton
	NeuralNetworkManager() = default;
	NeuralNetworkManager(const Hyperparameters& hyp);
	NeuralNetworkManager(const NeuralNetworkManager& oth) = delete;
	NeuralNetworkManager& operator = (const NeuralNetworkManager& oth) = delete;

	NeuralNetworkFF _neuralNetwork;
	NetworkResult _netResult;
	vector<Real> _input;
};

