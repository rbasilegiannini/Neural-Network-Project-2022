#pragma once
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "ActivationFunction.h"


using std::vector;
using boost::numeric::ublas::matrix;

/**
 *
 * @brief Data structure of a multilayer Neural Network Feed Forward.
 *
 */
class NeuralNetworkFF {
public:
	NeuralNetworkFF(const size_t inputDimension, const vector<size_t>& _nNeuronsPerHiddenLayer);

	vector<Real> ComputeNetwork(const vector<Real>& input);
	/***DEBUG PRINT FUNCTION***/
	void Print() {

		#include <iostream>
		using std::cout;
		using std::endl;

		// Print all weights and bias
		size_t idxLayer{ 0 };
		for (auto& weightMatrix : _weightPerLayer) {

			cout << "Dim: " << weightMatrix.size1() << "x" << weightMatrix.size2() << endl;
			cout << "Weights:" << endl;
			for (auto i = 0; i < weightMatrix.size1(); i++) {
				for (auto j = 0; j < weightMatrix.size2(); j++)
					cout << weightMatrix(i, j) << ' ';
				cout << endl;
			}
			cout << endl;

			cout << "bias:" << endl;
			for (size_t i{ 0 }; i < _biasPerLayer[idxLayer].size1(); i++)
				cout << _biasPerLayer[idxLayer](i, 0) << endl;

			cout << endl;

			idxLayer++;
		}
	}
	
	void setWeight(vector<matrix<Real>> input) {
		_weightPerLayer = input;
	}
	/**********/

private:
	const size_t _numLayers;
	const vector<size_t> _numNeuronsPerLayer;   
	vector<matrix<Real>> _weightPerLayer;
	vector<matrix<Real>> _biasPerLayer;
	vector<AFuncType> _activationFunctionPerLayer;
};

