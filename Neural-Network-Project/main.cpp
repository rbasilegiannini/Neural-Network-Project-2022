// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ActivationFunction.h"

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

using std::cout;
using std::endl;
using std::vector;
using boost::numeric::ublas::matrix;


int main() {

	// Test matrix

	vector<matrix<Real>> _weightPerLayer;
	const vector<size_t> _numNeuronsPerHiddenLayer{5, 4, 5, 3, 2};
	const size_t inputDimension{ 3 };
	const size_t _numHiddenLayers{ _numNeuronsPerHiddenLayer.size() };

	_weightPerLayer.resize(_numHiddenLayers);

	// Resize matrices and set weights for each hidden layer
	size_t idxLayer{ 0 };
	for (auto& weightMatrix : _weightPerLayer) {

		if (idxLayer == 0)
			weightMatrix.resize(_numNeuronsPerHiddenLayer[idxLayer], inputDimension);
		else
			weightMatrix.resize(_numNeuronsPerHiddenLayer[idxLayer], _numNeuronsPerHiddenLayer[idxLayer - 1]);

		// Random initialization of weights
		srand(time(0));
		for (auto i = 0; i < weightMatrix.size1(); i++) {
			for (auto j = 0; j < weightMatrix.size2(); j++)
				weightMatrix(i, j) = (Real)(((rand() % 21) - 10) * 0.1);	// Random value in [-1, 1] 
		}

		idxLayer++;
	}

	// Print all weights
	for (auto& weightMatrix : _weightPerLayer) {

		cout << "Dim: " << weightMatrix.size1() << "x" << weightMatrix.size2() << endl;
		for (auto i = 0; i < weightMatrix.size1(); i++) {
			for (auto j = 0; j < weightMatrix.size2(); j++)
				cout << weightMatrix(i, j) << ' ';
			cout << endl;
		}
		cout << endl;
	}
	
}