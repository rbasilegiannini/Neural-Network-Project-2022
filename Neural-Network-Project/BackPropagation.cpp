#include "BackPropagation.h"

vec_r BackPropagation(const DataFromNetwork& dataNN, const EFuncType EType, const mat_r& target)
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
		mat_r outputs(nNeuronsLayer.back(), 1);

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
	EFuncType EFuncType{ EType };
	if (EType == EFuncType::CROSSENTROPY_SOFTMAX) {
		if (dataNN.AFunctionDerivativePerLayer.back() == AFuncType::IDENTITY) 
			PostProcessing();

		else {
			std::cout << "[ERROR] It's not possibile to run the Post Processing step. " << std::endl;
			std::cout << "New loss function: CROSS ENTROPY. " << std::endl;

			EFuncType = EFuncType::CROSSENTROPY;
		}
	}

	if (EType == EFuncType::CROSSENTROPY) {
		if (dataNN.AFunctionDerivativePerLayer.back() != AFuncType::SIGMOID) {
			std::cout << "[ERROR] It's not possibile to use CROSS ENTROPY loss. " << std::endl;
			std::cout << "New loss function: SUM OF SQUARES. " << std::endl;

			EFuncType = EFuncType::SUMOFSQUARES;
		}
	}

#pragma endregion

#pragma region Compute delta	

	vector<mat_r> allDelta_row(nLayers);

	mat_r deltaOut_row(1, nNeuronsLayer.back());
	for (const auto& k : RangeGen(0, nNeuronsLayer.back())) 
		deltaOut_row(0, k) = ComputeDeltaOutput(k);
	allDelta_row.back() = deltaOut_row;

	//	Internal delta

	//	From the LAST-1 layer to the first layer.
	auto lastHiddenL = (nLayers - 1) - 1;
	for (const auto& layer : RangeGen(lastHiddenL, -1)) {

		mat_r delta_i_row(1, nNeuronsLayer[layer]);
		allDelta_row.front() = delta_i_row;
		AFuncType = AFuncLayer[layer];

		//	For all neurons of the layer i
		for (const auto& neuron : RangeGen(0, nNeuronsLayer[layer])) {

			auto a_i = AValLayer[layer](neuron, 0);
			auto AFuncDer_i = ActivationFunction::AFunction_Der[AFuncType];

			//	Compute the summation, for all the neurons of the next layer
			Real summation = row_by_column(allDelta_row[layer+1], extract_column(weightsLayer[layer + 1], neuron));

			delta_i_row(0, neuron) = AFuncDer_i(a_i) * summation;
		}
		allDelta_row[layer] = delta_i_row;
	}

#pragma endregion

#pragma region Compute Grad E
	vec_r gradE;

	for (const auto& layer : RangeGen(0, nLayers)) {
		Real dE_dparm;
		
		for (const auto& neuron : RangeGen(0, nNeuronsLayer[layer])) {

			//	Compute dE_dparm: biases
			dE_dparm = allDelta_row[layer](0, neuron);// * 1

			gradE.push_back(dE_dparm);

			//	Compute dE_dparm: weights
			for (const auto& conn : RangeGen(0, weightsLayer[layer].size2())) {

				if (layer == 0)
					dE_dparm = allDelta_row[layer](0, neuron) * input[conn];
				else
					dE_dparm = allDelta_row[layer](0, neuron) * outputsLayer[layer - 1](conn, 0);

				gradE.push_back(dE_dparm);
			}
		}
	}

#pragma endregion

	return gradE;
}
