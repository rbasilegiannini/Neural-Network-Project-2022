#include "BackPropagation.h"
#include <deque>

using std::deque;

vector<vector<Real>> BackPropagation::BProp(const DataFromNetwork& dataNN, const ErrorFuncType EFuncType, const vector<Real>& targets) {
	
	deque<deque<Real>> allDelta;

	//	Compute the output delta
	deque<Real> deltaOutput;

	AFuncType AFuncType = dataNN.AFunctionDerivativePerLayer.back();
	for (size_t k{ 0 }; k < dataNN.numNeuronsPerLayer.back() - 1; k++) {

		auto a_k = dataNN.activationsPerLayer.back()[k];
		auto y_k = dataNN.NNOutputs[k];
		auto t_k = targets[k];
		auto AFuncDer_k = ActivationFunction::AFunctionDerivative[AFuncType](a_k);
		auto EFuncDer_k = ErrorFunction::EFunctionDer_RespectOutput[EFuncType](y_k, t_k);

		deltaOutput.push_front(AFuncDer_k * EFuncDer_k);
	}

	allDelta.push_front(deltaOutput);

	//	Compute the internal delta

	// From the LAST-1 layer to the first layer.
	for (size_t i{ (dataNN.numLayers - 1) - 1 }; i > -1; i--) { 

		AFuncType = dataNN.AFunctionDerivativePerLayer[i];

		// For all neurons of the layer i
		for (size_t j{ 0 }; j < dataNN.numNeuronsPerLayer[i]; j++) {	

			auto a_i = dataNN.activationsPerLayer[i][j];
			auto AFuncDer_i = ActivationFunction::AFunctionDerivative[AFuncType](a_i);

		//	for (size_t h{ 0 }; h < dataNN.numNeuronsPerLayer[i+1];.)

		}
	}

}
