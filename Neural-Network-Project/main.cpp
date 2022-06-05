// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"
#include "BackPropagation.h"
#include "TestFunction.h"

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <functional>
#include <numeric>

using std::cout;
using std::endl;
using std::vector;
using boost::numeric::ublas::matrix;
using boost::numeric::ublas::subrange;

#include <chrono>
#include <thread>
using namespace std::chrono;

int main() {
/*
	const vector<size_t> _numNeuronsPerLayer{ 3, 1, 1};
	const vector<Real> input { 1, 1 };

	NeuralNetworkFF net (2, _numNeuronsPerLayer);

	vector<matrix<Real>> weights(3);
	weights[0].resize(3, 2);
	weights[1].resize(1, 3);
	weights[2].resize(1, 1);

	Real add{ 0 };

	///
	weights[0](0, 0) = -0.6 + add;
	weights[0](0, 1) = 0.7 + add;

	weights[0](1, 0) = 0.02 + add;
	weights[0](1, 1) = 0.2 + add;

	weights[0](2, 0) = -0.6 + add;
	weights[0](2, 1) = -0.4 + add;

	// W2
	weights[1](0, 0) = -0.8 + add;
	weights[1](0, 1) = -0.2 + add;
	weights[1](0, 2) = -0.4 + add;

	// W3
	weights[2](0, 0) = 0.6 + add;

	for (size_t i{ 0 }; i < _numNeuronsPerLayer.size(); i++) {
		net.SetWeights(i, weights[i]);

		vector<Real> vecZero(_numNeuronsPerLayer[i], 0);
		net.SetBias(i, vecZero);	// Bias to zero
	}

//	net.PrintNetwork();
/*
	auto params0 = net.GetAllParamPerLayer(0);
	auto params1 = net.GetAllParamPerLayer(1);

	for (size_t i = 0; i < params0.size1(); i++) {
		for (size_t j = 0; j < params0.size2(); j++)
			cout << params0(i, j) << ' ';
		cout << endl;
	}
	cout << endl;

	for (size_t i = 0; i < params1.size1(); i++) {
		for (size_t j = 0; j < params1.size2(); j++)
			cout << params1(i, j) << ' ';
		cout << endl;
	}
	cout << endl;
*/

#pragma region	Test compute network	
/* *
	auto result = net.ComputeNetwork(input);	// NetOutput = 0.5516

	for (int layer = 0; layer < net.GetNumLayers(); layer++) {
		cout << "Output layer " << layer << ": ";
		for (int i = 0; i < result.neuronsOutputPerLayer[layer].size1(); i++)
			cout << result.neuronsOutputPerLayer[layer](i, 0) << ' ';

		cout << endl;
	}

	for (int layer = 0; layer < net.GetNumLayers(); layer++) {
		cout << "Activation layer " << layer << ": ";
		for (int i = 0; i < result.activationsPerLayer[layer].size1(); i++)
			cout << result.activationsPerLayer[layer](i, 0) << ' ';

		cout << endl;
	}

	cout << endl;
/**/
#pragma endregion

#pragma region Test EFunc (Sum of squares)
/*	
	vector<Real> output { 5, 3, 2, 7, 2, 4 };
	vector<Real> target1{ 5+0.3, 3+0.15, 2+1, 7-0.25, 2+0.55, 4+1.1 };
	vector<Real> target2{ 5, 3 + 0.15, 2 + 1, 7, 2, 4 + 0.1 };
	vector<Real> target3{ 5, 3, 2, 7, 2, 4 };
	vector<Real> target4{ 5+1, 3+1, 2+1, 7+1, 2+1, 4+1 };

	cout << "Output to validate: ";
	for (const auto& o : output) cout << o << " ";
	cout << endl;

	cout << "Error with target 1: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target1) << endl;

	cout << "Error with target 2: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target2) << endl;

	cout << "Error with target 3: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target3) << endl;

	cout << "Error with target 4: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target4) << endl;

	cout << endl;
*/
#pragma endregion

#pragma region Test Gradient computation

	for (const auto& nTest : RangeGen(1, 2)) {

		vector<size_t> nNeuronsPerLayer;
		vector<Real> input;
		matrix<Real> target;

		// Set nNeuronPerLayer
		nNeuronsPerLayer.resize(5 * nTest);

		for (auto& nNeuron : nNeuronsPerLayer)
			nNeuron = (rand() % 100) + 1;

		// Set target
		target.resize(nNeuronsPerLayer.back(),1);

		for (const auto& t : RangeGen (0, target.size1()))
			target(t,0) = (rand() % 10) + 1;

		// Set input 
		input.resize(nTest);

		for (auto& i : input)
			i = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1]

		// Create the NN
		NeuralNetworkFF nn(input.size(), nNeuronsPerLayer);

//		nn.SetActivationFunction(nn.GetNumLayers() - 1, AFuncType::IDENTITY);

		// FP step
		auto nnResult = nn.ComputeNetwork(input);
		auto activationsPerLayer = nnResult.activationsPerLayer;
		vector<matrix<Real>> neuronsOutputPerLayer = nnResult.neuronsOutputPerLayer;

		// BP step
		vector<size_t> allNeuronsNumber;
		vector<AFuncType> allAFunc;
		vector<matrix<Real>> weightsPerLayer;

		for (const auto& layer : RangeGen(0, nn.GetNumLayers())) {
			allNeuronsNumber.push_back(nn.GetNumNeuronsPerLayer(layer));
			allAFunc.push_back(nn.GetAFuncPerLayer(layer));
			weightsPerLayer.push_back(nn.GetWeightsPerLayer(layer));
		}

		DataFromNetwork dataNN{
			nn.GetNumLayers(),
			allNeuronsNumber,
			activationsPerLayer,
			weightsPerLayer,
			allAFunc,
			neuronsOutputPerLayer,
			input
		};

		auto start_bp = high_resolution_clock::now();
		auto gradE = BackPropagation::BProp(dataNN, ErrorFuncType::SUMOFSQUARES, target);
		auto stop_bp = high_resolution_clock::now();

		auto duration_bp = duration_cast<seconds>(stop_bp - start_bp);

		//	Testing & compare
		auto start_chk = high_resolution_clock::now();
		bool test = Test_GradientChecking(nn, gradE, ErrorFuncType::SUMOFSQUARES, input, target);
		auto stop_chk = high_resolution_clock::now();

		auto duration_chk = duration_cast<seconds>(stop_chk - start_chk);

		cout << "Test number: " << nTest << ". Result: ";

		if (test)
			cout << "OK! Time saved: " << duration_chk.count() - duration_bp.count() << "s" << endl;
		else
			cout << "Fail!" << endl;

	}

#pragma endregion

} 