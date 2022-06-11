// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <functional>
#include <numeric>
#include <chrono>
#include <thread>

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"
#include "BackPropagation.h"
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
using std::vector;
using std::array;
using boost::numeric::ublas::matrix;
using boost::numeric::ublas::subrange;
using namespace std::chrono;
using std::uniform_int_distribution;

int main() {

	string sampleImagesFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-images.idx3-ubyte";
	string sampleLabelsFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-labels.idx1-ubyte"; 
	string testImagesFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\t10k-images.idx3-ubyte";
	string testLabelsFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\t10k-labels.idx1-ubyte";

	random_device rd;
	unsigned seed = rd();
	auto rng = default_random_engine(seed);
	uniform_int_distribution<int> uDist_train(0, 59999);
	uniform_int_distribution<int> uDist_test(0, 9999);

	//	Compose the Training Set and the Validation Set
	vector<ImageLabeled> samples = ReadSample(sampleImagesFile, sampleLabelsFile, 60000);
	shuffle(samples.begin(), samples.end(), rng);

	array<ImageLabeled, 5000> trainingSet;
	for (auto& s : trainingSet)
		s = samples[uDist_train(rng)];

	array<ImageLabeled, 2500> validationSet;
	for (auto& s : validationSet)
		s = samples[uDist_train(rng)];

	//	Compose the Test Set
	vector<ImageLabeled> tests = ReadSample(testImagesFile, testLabelsFile, 10000);
	shuffle(tests.begin(), tests.end(), rng);

	array<ImageLabeled, 2500> testSet;
	for (auto& s : testSet)
		s = tests[uDist_test(rng)];

	cout << endl;
}
