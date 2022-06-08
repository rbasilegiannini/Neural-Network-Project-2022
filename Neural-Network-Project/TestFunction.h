#pragma once

#include <vector>
#include "Utility.h"
#include "NeuralNetworkFF.h"
#include "BackPropagation.h"

using std::vector;

bool Test_GradientChecking (const NeuralNetworkFF& NN, const vec_r& gradToTest, 
	const EFuncType EType, const vec_r& input, const mat_r& target);




