// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "TestFunction.h"
#include "UpdateRule.h"
#include "ReadMNIST.h"

#include <array>
#include <algorithm>
#include <random>

using std::fill;
using std::default_random_engine;
using std::random_device;
using std::cout;
using std::endl;
using std::array;
using std::uniform_int_distribution;

struct Sample1D {
	vector<Real> image1D;
	vec_r labelOneHot{ vec_r(10, 0) };
};

struct NetworkConfig_Evaluated {
	size_t numLayers;
	vector<size_t> numNeurons_PerLayer;
	vector<AFuncType> AFuncType_PerLayer;
	
	Real error_training{ 0 };
	Real error_validation{ 0 };
};

Real ComputeError(NeuralNetworkManager& nnManager, const vector<Sample1D>& dataset, const EFuncType& Etype) {

	Real error{ 0 };
	for (const auto& sample : dataset) {
		nnManager.Run(sample.image1D);

		mat_r columnTarget(sample.labelOneHot.size(), 1);
		for (const auto& t : RangeGen(0, columnTarget.size1()))
			columnTarget(t, 0) = sample.labelOneHot[t];
		error += ErrorFunction::EFunction[Etype](nnManager.GetNetworkOutput(), columnTarget);

	}
	return error;
}

int main() { 

#pragma region Compose dataset
/**
	string sampleImagesFile = "train-images.idx3-ubyte";
	string sampleLabelsFile = "train-labels.idx1-ubyte"; 
	string testImagesFile = "t10k-images.idx3-ubyte";
	string testLabelsFile = "t10k-labels.idx1-ubyte";

	random_device rd;
	unsigned seed = rd();
	auto rng = default_random_engine(seed);	//TOREFACT: move to heap
	uniform_int_distribution<int> uDist_train(0, 59999);
	uniform_int_distribution<int> uDist_test(0, 9999);

	//	Training Set and Validation Set
	vector<ImageLabeled> samples = ReadSample(sampleImagesFile, sampleLabelsFile, 60000);
	shuffle(samples.begin(), samples.end(), rng);

	vector<Sample1D> trainingSet(5000);
	for (auto& s : trainingSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

	vector<Sample1D> validationSet (2500);
	for (auto& s : validationSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

	//	Test Set
	vector<ImageLabeled> tests = ReadSample(testImagesFile, testLabelsFile, 10000);
	shuffle(tests.begin(), tests.end(), rng);

	vector<Sample1D> testSet(2500);
	for (auto& s : testSet) {
		auto sample = tests[uDist_test(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}
/**/
#pragma endregion

#pragma region Learning
/**
	//	Create and set network
 	size_t inputDim = 28 * 28;
	vector<size_t> numNeurons_PerLayer(inputDim, 5);				//	5 neurons per layer
	vector<AFuncType> AFunc_PerLayer(inputDim, AFuncType::SIGMOID);	//	Sigmoid per layer			
	AFunc_PerLayer.back() = AFuncType::IDENTITY;	//	To use the cross-entropy + softmax

	Hyperparameters hyp{
		inputDim,
		numNeurons_PerLayer,
		AFunc_PerLayer
	};

	NeuralNetworkManager& netManager = NeuralNetworkManager::GetNNManager(hyp);

	//	Prepare learning
	size_t numParams{ 0 };
	for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
		numParams += netManager.GetAllParam_PerLayer(layer).size1() * netManager.GetAllParam_PerLayer(layer).size2();

	size_t maxEpoch{ 100 };
	vector<NetworkConfig_Evaluated> networks_evaluated (maxEpoch, {
			netManager.GetNumLayers(),
			netManager.GetAllNumNeurons(),
			netManager.GetAllAFuncType()
		});
	EFuncType EType{ EFuncType::CROSSENTROPY_SOFTMAX };
	vec_r gradE(numParams, 0);
	vec_r oldGradE(numParams, 0);

	//	Batch
	for (const auto& epoch : RangeGen(0, maxEpoch)) {

		// Set the gradient to 0
		fill(gradE.begin(), gradE.end(), 0);	

		for (const auto& sample : trainingSet) {
			netManager.Run(sample.image1D);	//	FP step
			auto gradE_sample = netManager.ComputeGradE_PerSample(EType, sample.labelOneHot); // BP step
			gradE += gradE_sample;
		}

		Rprop(netManager, gradE, oldGradE);

		networks_evaluated[epoch].error_training = ComputeError(netManager, trainingSet, EType);
		networks_evaluated[epoch].error_validation = ComputeError(netManager, validationSet, EType);

		//	Save current gradient
		oldGradE = gradE;
	}

	cout << endl;
#pragma endregion
/**/
}
