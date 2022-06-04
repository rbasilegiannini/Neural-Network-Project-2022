#include "BackPropagation.h"
#include <deque>

using std::deque;

vector<Real> BackPropagation::BProp(const DataFromNetwork& dataNN, const ErrorFuncType EFuncType, const vector<Real>& targets) {

	auto nLayers = dataNN.numLayers;
	auto AFuncLayer = dataNN.AFunctionDerivativePerLayer;
	auto AValLayer = dataNN.activationsPerLayer;
	auto weightsLayer = dataNN.weightsPerLayer;
	auto input = dataNN.NNinput;
	auto outputsLayer = dataNN.neuronsOutputPerLayer;
	auto nNeuronsLayer = dataNN.numNeuronsPerLayer;

#pragma region Compute delta	

	deque<vector<Real>> allDelta(nLayers);

	//	Output delta
	vector<Real> deltaOutput;

	AFuncType AFuncType = AFuncLayer.back();
	for (const auto& k : RangeGen(0, nNeuronsLayer.back())) {

		auto a_k = AValLayer.back()(k, 0);
		auto y_k = outputsLayer.back()(k, 0);
		auto t_k = targets[k];
		auto AFuncDer_k = ActivationFunction::AFunctionDerivative[AFuncType];
		auto EFuncDer_k = ErrorFunction::EFunctionDer_RespectOutput[EFuncType];

		deltaOutput.push_back(AFuncDer_k(a_k) * EFuncDer_k(y_k, t_k));
	}

	*(allDelta.end() - 1) = deltaOutput;

	//	Internal delta

	//	From the LAST-1 layer to the first layer.
	for (const auto& layer : RangeGen((nLayers - 1) - 1, -1)) {

		vector<Real> delta_i;
		*allDelta.begin() = delta_i;	// delta_i is empty. 

		AFuncType = AFuncLayer[layer];

		//	For all neurons of the layer i
		for (const auto& neuron : RangeGen(0, nNeuronsLayer[layer])) {

			auto a_i = AValLayer[layer](neuron, 0);
			auto AFuncDer_i = ActivationFunction::AFunctionDerivative[AFuncType];

			//	Compute the summation, for all the neurons of the next layer
			Real summation{ 0 };
			for (const auto& forwardConn : RangeGen(0, nNeuronsLayer[layer + 1]))
				summation += weightsLayer[layer + 1](forwardConn, neuron) * allDelta[layer + 1][forwardConn];

			delta_i.push_back(AFuncDer_i(a_i) * summation);
		}

		allDelta[layer] = delta_i;	//	Now delta_i is filled.
	}

#pragma endregion

#pragma region Compute Grad E
	vector<Real> gradE;

	for (const auto& layer : RangeGen(0, nLayers)) {
		Real dE_dparm;
		
		for (const auto& neuron : RangeGen(0, nNeuronsLayer[layer])) {

			//	Compute dE_dparm: biases
			dE_dparm = allDelta[layer][neuron] * 1;
			gradE.push_back(dE_dparm);

			//	Compute dE_dparm: weights
			for (const auto& conn : RangeGen(0, weightsLayer[layer].size2())) {

				if (layer == 0)
					dE_dparm = allDelta[layer][neuron] * input[conn];
				else
					dE_dparm = allDelta[layer][neuron] * outputsLayer[layer - 1](conn, 0);

				gradE.push_back(dE_dparm);
			}
		}
	}

#pragma endregion

	return gradE;
}
