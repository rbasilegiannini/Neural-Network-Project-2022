// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

using std::cout;
using std::endl;
using std::vector;
using boost::numeric::ublas::matrix;


int main() {

	// Test network
	const vector<size_t> _numNeuronsPerLayer{ 3, 1};
	const vector<Real> input { 1, 1 };

	NeuralNetworkFF net (2, _numNeuronsPerLayer);

	vector<matrix<Real>> weights(2);
	weights[0].resize(3, 2);
	weights[1].resize(1, 3);

	for (auto c : weights[0].data()) {
	}

	// W1
	weights[0](0, 0) = -0.66016222;
	weights[0](0, 1) = 0.77883251;

	weights[0](1, 0) = 0.02856532;
	weights[0](1, 1) = 0.1876014;

	weights[0](2, 0) = -0.65833175;
	weights[0](2, 1) = -0.43713064;

	// W2
	weights[1](0, 0) = -0.86631423;
	weights[1](0, 1) = -0.2239644;
	weights[1](0, 2) = -0.48749108;

	for (size_t i{ 0 }; i < _numNeuronsPerLayer.size(); i++) {
		net.SetWeights(i, weights[i]);

		vector<Real> vecZero(_numNeuronsPerLayer[i], 0);
		net.SetBias(i, vecZero);
	}

//	net.PrintNetwork();

//	auto result = net.ComputeNetwork(input);
//	for (auto e : result)
//		cout << e << ' ';

//	cout << endl;

	// Test error function (Sum of squares)
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
}