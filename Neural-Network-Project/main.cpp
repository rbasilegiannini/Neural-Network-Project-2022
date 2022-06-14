// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "TestFunction.h"
#include "RPROP.h"
#include "ReadMNIST.h"
#include "AnalysisTools.h"

#include <array>
#include <algorithm>
#include <random>

using std::fill;
using std::min_element;
using std::default_random_engine;
using std::random_device;
using std::cout;
using std::endl;
using std::array;
using std::uniform_int_distribution;

#pragma region Support
struct Sample1D {
	vector<Real> image1D;
	vec_r labelOneHot{ vec_r(10, 0) };
};

struct NetworkConfig_Evaluated {
	size_t numLayers;
	vector<size_t> numNeurons_PerLayer;
	vector<AFuncType> AFuncType_PerLayer;
	vector<mat_r> params;

	size_t epoch{ 0 };
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
#pragma endregion

#pragma region Settings
constexpr size_t MAX_TRAIN_SAMPLE = 60000;	//Default: 60000
constexpr size_t MAX_TEST_SAMPLE = 10000;	//Default: 10000

constexpr size_t NUM_TRAIN_SAMPLE = 5000;	//Request: 5000
constexpr size_t NUM_VAL_SAMPLE = 2500;		//Request: 2500
constexpr size_t NUM_TEST_SAMPLE = 2500;	//Request: 2500
constexpr size_t MAX_EPOCH = 10;

constexpr AFuncType AFUNC_LAYER = AFuncType::RELU;
constexpr size_t NUM_NEURONS_LAYER = 5;
constexpr size_t NUM_LAYERS = 5;

constexpr size_t NUM_CLASS = 10;
#pragma endregion

int main() { 
#pragma region Compose dataset
/**/
	string sampleImagesFile = "train-images.idx3-ubyte";
	string sampleLabelsFile = "train-labels.idx1-ubyte"; 
	string testImagesFile = "t10k-images.idx3-ubyte";
	string testLabelsFile = "t10k-labels.idx1-ubyte";

	random_device rd;
	unsigned seed = rd();
	auto rng = default_random_engine(seed);	//TOREFACT: move to heap
	uniform_int_distribution<int> uDist_train(0, MAX_TRAIN_SAMPLE - 1);
	uniform_int_distribution<int> uDist_test(0, MAX_TEST_SAMPLE - 1);

	//	Training Set and Validation Set
	cout << "Read training samples..." << endl;
	vector<ImageLabeled> samples = ReadSample(sampleImagesFile, sampleLabelsFile, MAX_TRAIN_SAMPLE);
	shuffle(samples.begin(), samples.end(), rng);

	cout << "Compose training set..." << endl;
	vector<Sample1D> trainingSet(NUM_TRAIN_SAMPLE);
	for (auto& s : trainingSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}

	cout << "Compose validation set..." << endl;
	vector<Sample1D> validationSet (NUM_VAL_SAMPLE);
	for (auto& s : validationSet) {
		auto sample = samples[uDist_train(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}
	samples.clear(); // Free memory

	//	Test Set
	cout << "Read test samples..." << endl;
	vector<ImageLabeled> tests = ReadSample(testImagesFile, testLabelsFile, MAX_TEST_SAMPLE);
	shuffle(tests.begin(), tests.end(), rng);

	cout << "Compose test set..." << endl;
	vector<Sample1D> testSet(NUM_TEST_SAMPLE);
	for (auto& s : testSet) {
		auto sample = tests[uDist_test(rng)];
		s.image1D = ConvertMatToArray<Real>(sample.image);
		s.labelOneHot[sample.label] = 1;
	}
	tests.clear(); // Free memory

	cout << "Training set: " << NUM_TRAIN_SAMPLE << " samples." << endl;
	cout << "Validation set: " << NUM_VAL_SAMPLE << " samples." << endl;
	cout << "Test set: " << NUM_TEST_SAMPLE << " samples." << endl;
	cout << endl;


/**/
#pragma endregion

#pragma region Learning
/**/
	//	Create and set network
	size_t inputDim{ 28 * 28 };
	vector<size_t> numNeurons_PerLayer(NUM_LAYERS, NUM_NEURONS_LAYER);
	vector<AFuncType> AFunc_PerLayer(NUM_LAYERS, AFUNC_LAYER);
	numNeurons_PerLayer.back() = NUM_CLASS;	//	Output network
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

	RPROP UpdateRule(numParams, 0.1);

	vector<NetworkConfig_Evaluated> networksEval (MAX_EPOCH, {
			netManager.GetNumLayers(),
			netManager.GetAllNumNeurons(),
			netManager.GetAllAFuncType()
		});

	EFuncType EType{ EFuncType::CROSSENTROPY_SOFTMAX };
	vec_r gradE(numParams, 0);

	auto NetworkParams = [&netManager]()-> vector<mat_r> {
		vector<mat_r> AllParams;
		for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
			AllParams.push_back(netManager.GetAllParam_PerLayer(layer));

		return AllParams;
	};

	//	Batch
	cout << "Learning batch..." << endl;
	for (const auto& epoch : RangeGen(0, MAX_EPOCH)) {
		cout << "Epoch " << epoch << "... ";
		networksEval[epoch].epoch = epoch;

		// Set the gradient to 0
		fill(gradE.begin(), gradE.end(), 0);	

		for (const auto& sample : trainingSet) {
			netManager.Run(sample.image1D);	//	FP step
			auto gradE_sample = netManager.ComputeGradE_PerSample(EType, sample.labelOneHot); // BP step
			gradE += gradE_sample;
		}

		UpdateRule.Run(netManager, gradE);

		networksEval[epoch].error_training = ComputeError(netManager, trainingSet, EType);
		networksEval[epoch].error_validation = ComputeError(netManager, validationSet, EType);
		networksEval[epoch].params = NetworkParams();

		cout << "done." << endl;
	}
	cout << endl;

	cout << "Retrieve best net... ";
	auto cmp = [](const NetworkConfig_Evaluated& lhs, const NetworkConfig_Evaluated& rhs)-> bool {
		return (lhs.error_validation < rhs.error_validation);
	};
	
	NetworkConfig_Evaluated bestNetEval = *min_element(networksEval.begin(), networksEval.end(), cmp);

	//	Rebuilding network
	for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
		netManager.SetAllParam_PerLayer(layer, bestNetEval.params[layer]);

	cout << "done." << endl;
/**/	
#pragma endregion

#pragma region Results
/**/
	vector<double> x_epoch;
	vector<double> y_error_train;
	vector<double> y_error_val;

	for (const auto& epoch : RangeGen(0, MAX_EPOCH)) {
		auto net = networksEval[epoch];
		x_epoch.push_back(net.epoch);
		y_error_train.push_back(net.error_training);
		y_error_val.push_back(net.error_validation);
	}

	if(SavePlot("Training error plot RELU", x_epoch, y_error_train))
		cout << "Training error plot saved." << endl;
	else
		cout << "Error to save training error plot." << endl;

	if(SavePlot("Validation error plot RELU", x_epoch, y_error_val))
		cout << "Validation error plot saved." << endl;
	else
		cout << "Error to save validation error plot." << endl;

	cout << "Best network:" << endl;
	cout << "Epoch: " << bestNetEval.epoch << endl;
	cout << "Training error: " << bestNetEval.error_training << endl;
	cout << "Validation error: " << bestNetEval.error_validation << endl;

	cout << endl;
/**/
#pragma endregion

}
