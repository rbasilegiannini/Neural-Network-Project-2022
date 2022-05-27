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
	NeuralNetworkFF(const size_t inputDimension, const vector<size_t> _nNeuronsPerHiddenLayer);

private:
	const size_t _numLayers{ 1 };
	const vector<size_t> _numNeuronsPerLayer;   
	vector<matrix<Real>> _weightPerLayer;
	matrix<Real> _biasPerLayer;
	vector<AFuncType> _activationFunctionPerLayer;
};

