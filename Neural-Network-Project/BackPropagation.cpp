#include "BackPropagation.h"

vector<vector<Real>> BackPropagation::BProp(const DataFromNetwork& dataNN, const ErrorFunction EFuncType) {
	
	vector<vector<Real>> allDelta;

	// Compute the output delta
	vector<Real> deltaOutput;

	for (size_t k{ 0 }; k < dataNN.numNeuronsPerLayer.back() - 1; k++) {
	//
	}

	return allDelta;
}
