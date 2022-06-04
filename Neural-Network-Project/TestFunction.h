#pragma once

#include <vector>
#include "Utility.h"
#include "NeuralNetworkFF.h"
#include "BackPropagation.h"

using std::vector;

bool Test_GradientChecking (const NeuralNetworkFF& NN, const vector<Real>& gradToTest, 
	const ErrorFuncType EFuncType, const vector<Real>& input, const vector<Real>& target);




