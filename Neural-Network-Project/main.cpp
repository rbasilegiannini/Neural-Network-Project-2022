// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetworkFF.h"

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

	net.PrintNetwork();

	auto result = net.ComputeNetwork(input);
	for (auto e : result)
		cout << e << ' ';

	cout << endl;
}