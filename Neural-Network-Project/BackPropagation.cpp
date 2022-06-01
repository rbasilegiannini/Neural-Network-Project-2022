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
		auto AFuncDer_k = ActivationFunction::AFunctionDerivative[AFuncType];
		auto EFuncDer_k = ErrorFunction::EFunctionDer_RespectOutput[EFuncType];

		deltaOutput.push_front(AFuncDer_k(a_k) * EFuncDer_k(y_k, t_k));
	}

	allDelta.push_front(deltaOutput);

	//	Compute the internal delta

	//	From the LAST-1 layer to the first layer.
	for (int i = (int)(dataNN.numLayers - 1) - 1; i > -1; i--) {

		deque<Real> delta_i;
		AFuncType = dataNN.AFunctionDerivativePerLayer[i];

		//	For all neurons of the layer i
		for (size_t j{ 0 }; j < dataNN.numNeuronsPerLayer[i]; j++) {	

			auto a_i = dataNN.activationsPerLayer[i][j];
			auto AFuncDer_i = ActivationFunction::AFunctionDerivative[AFuncType];

			//	Compute the summation, for all the neurons of the next layer (connected to the node i because of the NN is FF)
			Real summation{ 0 };
			for (size_t h{ 0 }; h < dataNN.numNeuronsPerLayer[i + 1]; h++) {
				summation += dataNN.parameters[i + 1](h, i) * allDelta[i + 1][h];
			}

			delta_i.push_front(AFuncDer_i(a_i) * summation);
		}

		allDelta.push_front(delta_i);
	}
	
	//	Convert result in a vector of vectors of reals
	vector<vector<Real>> allDelta_vector;
	for (const auto& d_i : allDelta) {
		vector<Real> temp;
		for (const auto& d_ij : d_i) temp.push_back(d_ij);
		allDelta_vector.push_back(temp);
	}
	
	return allDelta_vector;
}
