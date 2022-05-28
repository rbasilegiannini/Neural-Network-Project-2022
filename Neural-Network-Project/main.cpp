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
	const vector<size_t> _numNeuronsPerHiddenLayer{ 3, 1 };
	const vector<Real> input { 1, 1 };

	NeuralNetworkFF net (2, _numNeuronsPerHiddenLayer);

	vector<matrix<Real>> weights(2);
	weights[0].resize(3, 2);
	weights[1].resize(1, 3);

	// W1
	weights[0](0, 0) = 0.06364929;
	weights[0](0, 1) = -0.52697621;

	weights[0](1, 0) = -0.89738668;
	weights[0](1, 1) = 0.63310678;

	weights[0](2, 0) = 0.29764894;
	weights[0](2, 1) = 0.21413957;

	// W2
	weights[1](0, 0) = -0.62903204;
	weights[1](0, 1) = -0.78213938;
	weights[1](0, 2) = 0.93025369;

	net.setWeight(weights);

	net.Print();

	auto result = net.ComputeNetwork(input);
	for (auto e : result)
		cout << e << ' ';

	cout << endl;
}