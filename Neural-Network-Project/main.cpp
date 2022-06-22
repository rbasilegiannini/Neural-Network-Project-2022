// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "RPROP.h"
#include "ReadMNIST.h"
#include "AnalysisTools.h"
#include <array>
#include <algorithm>
#include <numeric>

#pragma region Support

using std::fill;
using std::cout;
using std::endl;
using std::array;
using std::abs;

struct Sample1D {
	vec_r image1D;
	vec_r labelOneHot{ vec_r(10, 0) };
};

typedef vector<Sample1D> Dataset;

struct NetConfig_Evaluated {
	size_t numLayers{ 0 };
	vector<size_t> numNeurons_PerLayer;
	vector<AFuncType> AFuncType_PerLayer;
	vector<mat_r> params;

	size_t epoch{ 0 };
	Real error_training{ 0 };
	Real error_validation{ 0 };
};

void ComposeDataset(Dataset&, Dataset&, Dataset&, const double);
void Learning(NeuralNetworkManager&, vector<NetConfig_Evaluated>&, const Dataset&, const Dataset&, const size_t maxEpoch);
float Accuracy(NeuralNetworkManager&, const Dataset&);
Real ComputeError(NeuralNetworkManager&, const vector<Sample1D>&, const EFuncType&);
NetConfig_Evaluated RetrieveBestNet(const vector<NetConfig_Evaluated>&, NeuralNetworkManager&);

#pragma endregion

#pragma region Settings

constexpr size_t NUM_TRAIN_SAMPLE = 5000;	//Request: 5000
constexpr size_t NUM_VAL_SAMPLE = 2500;		//Request: 2500
constexpr size_t NUM_TEST_SAMPLE = 2500;	//Request: 2500
constexpr size_t MAX_EPOCH = 1000;

constexpr AFuncType AFUNC_LAYER = AFuncType::SIGMOID;
constexpr size_t NUM_HIDDEN_LAYERS = 2;
constexpr array<size_t, NUM_HIDDEN_LAYERS> NUM_NEURONS_HIDDEN{ 5, 10};

constexpr size_t NUM_CLASS = 10;
constexpr size_t PATIENCE = 20;
constexpr double resizeFactor = 0.5;

#define EARLY_STOPPING
constexpr float SENSITIVITY = (float)0.005;

#pragma endregion

int main() {

	//	Prepare network manager
	size_t inputDim{ size_t (resizeFactor*28 * resizeFactor*28) };
	vector<size_t> numNeurons_PerLayer(NUM_HIDDEN_LAYERS+1);
	vector<AFuncType> AFunc_PerLayer(NUM_HIDDEN_LAYERS+1, AFUNC_LAYER);
	NeuralNetworkManager& netManager = NeuralNetworkManager::GetNNManager(Hyperparameters());

	for (const auto& n : RangeGen(0, NUM_NEURONS_HIDDEN.size()))
		numNeurons_PerLayer[n] = NUM_NEURONS_HIDDEN[n];
	numNeurons_PerLayer.back() = NUM_CLASS; // Output network

	AFunc_PerLayer.back() = AFuncType::IDENTITY; // To use the cross-entropy + softmax
	Hyperparameters hyp{
		inputDim,
		numNeurons_PerLayer,
		AFunc_PerLayer
	};

	netManager.ResetHyperparameters(hyp);
	netManager.RandomInitialization(-1, 1);

	cout << "Network with" << endl;
	cout << "Hidden layer: " << netManager.GetNumLayers()-1 << endl;
	
	cout << "Neurons per layer: ";
	for (const auto& n : RangeGen(0, netManager.GetNumLayers()))
		cout << netManager.GetAllNumNeurons()[n] << "/";
	cout << endl;
	cout << "Activation function: " << NameOfAFuncType(netManager.GetAllAFuncType().front()) << endl;
	cout << endl;

	//	Free memory
	numNeurons_PerLayer.clear();
	AFunc_PerLayer.clear();

	//
	
	vector<NetConfig_Evaluated> networksEval;
	NetConfig_Evaluated bestNetEval;
	Dataset trainingSet, validationSet, testSet;

	ComposeDataset(trainingSet, validationSet, testSet, resizeFactor);

	Learning(netManager, networksEval, trainingSet, validationSet, MAX_EPOCH);
	bestNetEval = RetrieveBestNet(networksEval, netManager);
	auto accuracy = Accuracy(netManager, testSet);

	cout << endl;

	// Results
	vector<double> x_epoch;
	vector<double> y_error_train;
	vector<double> y_error_val;

	for (const auto& net : networksEval) {

		x_epoch.push_back((double)net.epoch);
		y_error_train.push_back(net.error_training);
		y_error_val.push_back(net.error_validation);
	}

	if(SavePlot("Plot training error", x_epoch, y_error_train))
		cout << "Training error plot saved." << endl;
	else
		cout << "Error to save training error plot." << endl;

	if(SavePlot("Plot validation error", x_epoch, y_error_val))
		cout << "Validation error plot saved." << endl;
	else
		cout << "Error to save validation error plot." << endl;
	cout << endl;

	cout << "Best network with" << endl;
	cout << "Epoch: " << bestNetEval.epoch << endl;
	cout << "Validation error: " << bestNetEval.error_validation << endl;
	cout << "Accuracy: " << accuracy*100 << "%" << endl;
	cout << endl;

}
// Body

void ComposeDataset(Dataset& trainSet, Dataset& valSet, Dataset& testSet, const double factor) {

	trainSet.resize(NUM_TRAIN_SAMPLE);
	valSet.resize(NUM_VAL_SAMPLE);
	testSet.resize(NUM_TEST_SAMPLE);

	string sampleImagesFile = "train-images.idx3-ubyte";
	string sampleLabelsFile = "train-labels.idx1-ubyte";
	string testImagesFile = "t10k-images.idx3-ubyte";
	string testLabelsFile = "t10k-labels.idx1-ubyte";


	//	Read and resize Training Set and Validation Set
	cout << "Read training samples..." << endl;
	vector<ImageLabeled> samples = ReadSample(sampleImagesFile, sampleLabelsFile, NUM_TRAIN_SAMPLE + NUM_VAL_SAMPLE);
	ResizeDatasetRaw(samples, factor);

	//	Retrieve max and min from known samples. These values will be use also for test set. 
	vector<uint8_t> minmaxFromSamples = RetrieveMinMaxFromDatasetRaw(samples);
	uint8_t min = minmaxFromSamples[0];
	uint8_t max = minmaxFromSamples[1];

	//	First NUM_TRAIN_SAMPLE samples
	cout << "Compose training set..." << endl;
	for (const auto& i : RangeGen(0, NUM_TRAIN_SAMPLE)) {
		auto sample = samples[i];
		trainSet[i].image1D = NormalizeVector(ConvertMatToArray<Real>(sample.image), (Real)max, (Real)min, (Real)(- 0.5), (Real)0.5);
		trainSet[i].labelOneHot[sample.label] = 1;
	}

	//	Last NUM_VAL_SAMPLE samples
	cout << "Compose validation set..." << endl;
	for (const auto& i : RangeGen(0, NUM_VAL_SAMPLE)) {
		auto sample = samples[NUM_TRAIN_SAMPLE+i];
		valSet[i].image1D = NormalizeVector(ConvertMatToArray<Real>(sample.image), (Real)max, (Real)min, (Real)(-0.5), (Real)0.5);
		valSet[i].labelOneHot[sample.label] = 1;
	}
	samples.clear(); // Free memory

	//	Read and resize Test Set
	cout << "Read test samples..." << endl;
	vector<ImageLabeled> tests = ReadSample(testImagesFile, testLabelsFile, NUM_TEST_SAMPLE);
	ResizeDatasetRaw(tests, factor);

	cout << "Compose test set..." << endl;
	for (const auto& i : RangeGen(0, NUM_TEST_SAMPLE)) {
		auto sample = tests[i];
		testSet[i].image1D = NormalizeVector(ConvertMatToArray<Real>(sample.image), (Real)max, (Real)min, (Real)(-0.5), (Real)0.5);
		testSet[i].labelOneHot[sample.label] = 1;
	}
	tests.clear(); // Free memory

	cout << "Training set: " << trainSet.size() << " samples." << endl;
	cout << "Validation set: " << valSet.size() << " samples." << endl;
	cout << "Test set: " << testSet.size() << " samples." << endl;
	cout << endl;
}
void Learning(NeuralNetworkManager& netManager, vector<NetConfig_Evaluated>& networksEval,
	const Dataset& trainSet, const Dataset& valSet, const size_t maxEpoch) {

	// Tools
	auto NetworkParams = [&netManager]()-> vector<mat_r> {
		vector<mat_r> AllParams;
		for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
			AllParams.push_back(netManager.GetAllParam_PerLayer(layer));

		return AllParams;
	};
	//

	try {

		//	Prepare learning
		size_t numParams{ 0 };
		for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
			numParams += netManager.GetAllParam_PerLayer(layer).size1() * netManager.GetAllParam_PerLayer(layer).size2();

		RPROP UpdateRule(numParams, (Real)0.1);
		EFuncType EType{ EFuncType::CROSSENTROPY_SOFTMAX };
		vec_r gradE(numParams, 0);

		// Batch
		Real oldEVal{ (Real)std::numeric_limits<float>::max() };
		size_t pat{ 0 };
		cout << "Learning batch..." << endl;

		for (const auto& epoch : RangeGen(0, maxEpoch)) {
			cout << "Epoch " << epoch + 1 << "... ";

			// Set the gradient to 0
			fill(gradE.begin(), gradE.end(), 0);

			for (const auto& sample : trainSet) {
				netManager.Run(sample.image1D);	//	FP step
				auto gradE_sample = netManager.ComputeGradE_PerSample(EType, sample.labelOneHot); // BP step
				gradE += gradE_sample;
			}

			UpdateRule.Run(netManager, gradE);

			auto eVal = ComputeError(netManager, valSet, EType);
			auto eTrain = ComputeError(netManager, trainSet, EType);

			cout << "validation error: " << eVal << "... ";

			NetConfig_Evaluated netEval{
				netManager.GetNumLayers(),
				netManager.GetAllNumNeurons(),
				netManager.GetAllAFuncType(),
				NetworkParams(),
				(size_t)epoch,
				eTrain,
				eVal
			};

			networksEval.push_back(netEval);

		#ifdef EARLY_STOPPING

			if ((abs(eVal - oldEVal) < SENSITIVITY) || eVal > oldEVal) {
				pat++;
			}
			else {
				pat = 0;
				oldEVal = eVal;
			}

			if (pat == PATIENCE)
				break;

			cout << "done. Patience: " << pat << "/" << PATIENCE << endl;

		#else

			cout << "done. " << endl;

		#endif // EARLY_STOPPING

		}
	}

	catch (InvalidParametersException e) {
		cout << e.getErrorMessage() << endl;
	}

	cout << endl;
}
NetConfig_Evaluated RetrieveBestNet(const vector<NetConfig_Evaluated>& networksEval, NeuralNetworkManager& netManager) {

	auto cmp = [](const NetConfig_Evaluated& lhs, const NetConfig_Evaluated& rhs)-> bool {

		return (lhs.error_validation < rhs.error_validation);
	};

	cout << "Retrieve best net... ";

	NetConfig_Evaluated bestNetEval = *min_element(networksEval.begin(), networksEval.end(), cmp);

	//	Rebuilding network
	for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
		netManager.SetAllParam_PerLayer(layer, bestNetEval.params[layer]);

	cout << "done." << endl;

	return bestNetEval;
}
float Accuracy(NeuralNetworkManager& netManager, const Dataset& testSet) {
	cout << "Compute accuracy... ";

	size_t numCorrect{ 0 };
	for (const auto& sample : testSet) {
		netManager.Run(sample.image1D);

		//	Decision rule
		auto out = netManager.GetNetworkOutput().data();
		size_t predIndex = max_element(out.begin(), out.end()) - out.begin();	//max_element returns an iterator

		if (sample.labelOneHot[predIndex] == 1)
			numCorrect++;
	}
	float accuracy = ((float)numCorrect) / ((float)testSet.size());
	
	cout << "done." << endl;

	return accuracy;
}
Real ComputeError(NeuralNetworkManager& nnManager, const vector<Sample1D>& dataset, const EFuncType& Etype) {

	Real error{ 0 };
	for (const auto& sample : dataset) {
		nnManager.Run(sample.image1D);

		mat_r columnTarget(sample.labelOneHot.size(), 1);
		for (const auto& t : RangeGen(0, columnTarget.size1()))
			columnTarget(t, 0) = sample.labelOneHot[t];
		error += ErrorFunction::EFunction[Etype](nnManager.GetNetworkOutput(), columnTarget);

	}
	return error / dataset.size();
}
