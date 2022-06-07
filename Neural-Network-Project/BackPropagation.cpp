#include "BackPropagation.h"
#include <deque>

using std::deque;

vector<Real> BackPropagation(const DataFromNetwork& dataNN, const ErrorFuncType EType, const matrix<Real>& target) 
throw (InvalidParametersException) {

	auto nLayers = dataNN.numLayers;
	auto AFuncLayer = dataNN.AFunctionDerivativePerLayer;
	auto AValLayer = dataNN.activationsPerLayer;
	auto weightsLayer = dataNN.weightsPerLayer;
	auto input = dataNN.NNinput;
	auto outputsLayer = dataNN.neuronsOutputPerLayer;
	auto nNeuronsLayer = dataNN.numNeuronsPerLayer;

#pragma region Tools

	auto PostProcessing = [&]() {
		matrix<Real> outputs(nNeuronsLayer.back(), 1);

		for (const auto& k : RangeGen(0, nNeuronsLayer.back())) {
			outputs(k, 0) = SoftMax(outputsLayer.back(), k);
		}

		outputsLayer.back() = outputs;
	};

	AFuncType AFuncType = AFuncLayer.back();
	auto ComputeDeltaOutput = [&](const size_t neuron) -> Real {

		auto a_k = AValLayer.back()(neuron, 0);
		auto y_k = outputsLayer.back()(neuron, 0);
		auto t_k = target(neuron,0);
		auto AFuncDer_k = ActivationFunction::AFunction_Der[AFuncType];
		auto EFuncDer_k = ErrorFunction::EFunctionDer_RespectOutput[EType];

		return (AFuncDer_k(a_k) * EFuncDer_k(y_k, t_k));
	};

#pragma endregion

#pragma region Checks
	if (target.size1() != nNeuronsLayer.back())
		throw InvalidParametersException("[BACKPROP] the target is not compatbile with network's output.");

	//	Check on loss function
	ErrorFuncType EFuncType{ EType };
	if (EType == ErrorFuncType::CROSSENTROPY_SOFTMAX) {
		if (dataNN.AFunctionDerivativePerLayer.back() == AFuncType::IDENTITY) 
			PostProcessing();

		else {
			std::cout << "[ERROR] It's not possibile to run the Post Processing step. " << std::endl;
			std::cout << "New loss function: CROSS ENTROPY. " << std::endl;

			EFuncType = ErrorFuncType::CROSSENTROPY;
		}
	}

	if (EType == ErrorFuncType::CROSSENTROPY) {
		if (dataNN.AFunctionDerivativePerLayer.back() != AFuncType::SIGMOID) {
			std::cout << "[ERROR] It's not possibile to use CROSS ENTROPY loss. " << std::endl;
			std::cout << "New loss function: SUM OF SQUARES. " << std::endl;

			EFuncType = ErrorFuncType::SUMOFSQUARES;
		}
	}

#pragma endregion

#pragma region Compute delta	

	deque<vector<Real>> allDelta(nLayers);

	vector<Real> deltaOutput;
	for (const auto& k : RangeGen(0, nNeuronsLayer.back()))
		deltaOutput.push_back(ComputeDeltaOutput(k));
	allDelta.back() = deltaOutput;

	//	Internal delta

	//	From the LAST-1 layer to the first layer.
	auto lastHiddenL = (nLayers - 1) - 1;
	for (const auto& layer : RangeGen(lastHiddenL, -1)) {

		vector<Real> delta_i;
		allDelta.front() = delta_i; // delta_i is empty. 

		AFuncType = AFuncLayer[layer];

		//	For all neurons of the layer i
		for (const auto& neuron : RangeGen(0, nNeuronsLayer[layer])) {

			auto a_i = AValLayer[layer](neuron, 0);
			auto AFuncDer_i = ActivationFunction::AFunction_Der[AFuncType];

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
