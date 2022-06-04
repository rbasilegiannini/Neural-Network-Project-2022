#include "BackPropagation.h"
#include <deque>

using std::deque;

vector<Real> BackPropagation::BProp(const DataFromNetwork& dataNN, const ErrorFuncType EFuncType, const vector<Real>& targets) {

#pragma region Compute delta	

	deque<vector<Real>> allDelta(dataNN.numLayers);

	//	Output delta
	vector<Real> deltaOutput;

	AFuncType AFuncType = dataNN.AFunctionDerivativePerLayer.back();
	for (size_t k{ 0 }; k < dataNN.numNeuronsPerLayer.back(); k++) {

		auto a_k = dataNN.activationsPerLayer.back()(k,0);
		auto y_k = dataNN.neuronsOutputPerLayer.back()(k,0);
		auto t_k = targets[k];
		auto AFuncDer_k = ActivationFunction::AFunctionDerivative[AFuncType];
		auto EFuncDer_k = ErrorFunction::EFunctionDer_RespectOutput[EFuncType];

		deltaOutput.push_back(AFuncDer_k(a_k) * EFuncDer_k(y_k, t_k));
	}

	*(allDelta.end()-1) = deltaOutput;

	//	Internal delta

	//	From the LAST-1 layer to the first layer.
	for (int layer = (int)(dataNN.numLayers - 1) - 1; layer > -1; layer--) {

		vector<Real> delta_i;
		*allDelta.begin() = delta_i;	// delta_i is empty. 

		AFuncType = dataNN.AFunctionDerivativePerLayer[layer];

		//	For all neurons of the layer i
		for (size_t idxNeuron{ 0 }; idxNeuron < dataNN.numNeuronsPerLayer[layer]; idxNeuron++) {	

			auto a_i = dataNN.activationsPerLayer[layer](idxNeuron, 0); 
			auto AFuncDer_i = ActivationFunction::AFunctionDerivative[AFuncType];

			//	Compute the summation, for all the neurons of the next layer
			Real summation{ 0 };
			for (size_t forwardConnection{ 0 }; forwardConnection < dataNN.numNeuronsPerLayer[layer + 1]; forwardConnection++) 
				summation += dataNN.weightsPerLayer[layer + 1](forwardConnection, idxNeuron) * allDelta[layer + 1][forwardConnection];
			
			delta_i.push_back(AFuncDer_i(a_i) * summation);
		}

		allDelta[layer] = delta_i;	//	Now delta_i is filled.
	}

#pragma endregion

#pragma region Compute Grad E
	vector<Real> gradE;

	for (size_t layer{ 0 }; layer < dataNN.numLayers; layer++) {
		Real dE_dparm;
		
		for (size_t idxNeuron{ 0 }; idxNeuron < dataNN.numNeuronsPerLayer[layer]; idxNeuron++) {

			//	Compute dE_dparm: biases
			dE_dparm = allDelta[layer][idxNeuron] * 1;
			gradE.push_back(dE_dparm);

			//	Compute dE_dparm: weights
			for (size_t idxConnection = 0; idxConnection < dataNN.weightsPerLayer[layer].size2(); idxConnection++) {

				if (layer == 0)
					dE_dparm = allDelta[layer][idxNeuron] * dataNN.NNinput[idxConnection];
				else
					dE_dparm = allDelta[layer][idxNeuron] * dataNN.neuronsOutputPerLayer[layer - 1](idxConnection, 0);

				gradE.push_back(dE_dparm);
			}
		}
	}

#pragma endregion

	return gradE;
}
