#pragma once


#include "Utility.h"
#include "NeuralNetworkFF.h"
#include "BackPropagation.h"

bool Test_GradientChecking (const NeuralNetworkFF& NN, const vec_r& gradToTest, 
	const EFuncType EType, const vec_r& input, const mat_r& target);

void TestCase_GradientComputation();
void TestCase_TimingGradientComputation();


