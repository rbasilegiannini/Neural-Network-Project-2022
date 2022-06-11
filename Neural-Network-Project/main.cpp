// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "TestFunction.h"
#include "NeuralNetworkManager.h"
#include "ReadMNIST.h"

#include <array>
#include <algorithm>
#include <random>

using std::default_random_engine;
using std::random_device;
using std::cout;
using std::endl;
using std::array;
using std::uniform_int_distribution;

struct sample1D {
	vector<Real> image1D;
	vec_r labelOneHot{ vec_r(9, 0) };
};

vector<Real> ConvertMatToArray(const mat_i& mat) {
	vector<Real> arr;
	for (const auto& row : RangeGen(0, mat.size1()))
		for (const auto& col : RangeGen(0, mat.size2()))
			arr.push_back((Real)mat(row, col));
	return arr;
}

int main() { 
/**
#pragma region Compose dataset

	string sampleImagesFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-images.idx3-ubyte";
	string sampleLabelsFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-labels.idx1-ubyte"; 
	string testImagesFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\t10k-images.idx3-ubyte";
	string testLabelsFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\t10k-labels.idx1-ubyte";

	random_device rd;
	unsigned seed = rd();
	auto rng = default_random_engine(seed);	//TOREFACT: move to heap
	uniform_int_distribution<int> uDist_train(0, 59999);
	uniform_int_distribution<int> uDist_test(0, 9999);

	//	Training Set and Validation Set
	vector<ImageLabeled> samples = ReadSample(sampleImagesFile, sampleLabelsFile, 60000);
	shuffle(samples.begin(), samples.end(), rng);

	array<sample1D, 5000> trainingSet;
	for (auto& s : trainingSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

	array<sample1D, 2500> validationSet;
	for (auto& s : validationSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

	//	Test Set
	vector<ImageLabeled> tests = ReadSample(testImagesFile, testLabelsFile, 10000);
	shuffle(tests.begin(), tests.end(), rng);

	array<sample1D, 2500> testSet;
	for (auto& s : testSet) {
		auto sample = tests[uDist_train(rng)];
		s.image1D = ConvertMatToArray(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

#pragma endregion Compose Dataset

#pragma region Learning

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

	//	Learning with batch
	size_t numParams{ 0 };
	for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
		numParams += netManager.GetAllParam_PerLayer(layer).size1() * netManager.GetAllParam_PerLayer(layer).size2();

	EFuncType EType{ EFuncType::CROSSENTROPY_SOFTMAX };
	for (const auto& epoch : RangeGen(0, 100)) {
		
		vec_r gradE(numParams, 0);
		for (const auto& sample : trainingSet) {
			netManager.Run(sample.image1D);	//	FP step
			auto gradE_sample = netManager.ComputeGradE_PerSample(EType, sample.labelOneHot); // BP step
			gradE = gradE + gradE_sample;
		}

		//	Update parameters (RPROP)

		//	Compute E_ts

		//	Compute E_vs

		//	Save net configuration into an array 
	}

	cout << endl;
#pragma endregion Learning
/**/

	mat_r mat(4, 4);

	for (const auto& row : RangeGen(0, mat.size1()))
		for (const auto& col : RangeGen(0, mat.size2()))
			mat(row, col) = row;

	vec_r vec1 = ConvertMatToArray(mat);
	vec_r vec2 = ConvertMatToArray(mat);
	vec1 += vec2;

	for (const auto& e : vec1)
		cout << e << ' ';
	cout << endl;

	for (const auto& e : vec2)
		cout << e << ' ';
	cout << endl;
	
	//	Test vec + vec

}
