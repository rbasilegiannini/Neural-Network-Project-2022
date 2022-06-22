#include "RPROP.h"

using std::min;
using std::max;

RPROP::RPROP(const size_t numParam, const Real initialUpdate) {
	_oldGradE.resize(numParam, 0);
	_updateValue.resize(numParam, initialUpdate);
	_delta.resize(numParam, 0);
}

void RPROP::Run(NeuralNetworkManager& netManager, const vec_r& gradE) {

	vector <mat_r> netParams_PerLayer;

	try {
		//	Update parameters
		size_t offset{ 0 };
		for (const auto& layer : RangeGen(0, netManager.GetNumLayers())) {
			auto matParams = netManager.GetAllParam_PerLayer(layer);
			auto& params = matParams.data();	//	Reference to one-dim version of matrix

			for (const auto& i : RangeGen(0, params.size())) {
				Real& oldGradE_i = _oldGradE[offset + i];	//	It's a reference
				Real currentGradE_i = gradE[offset + i];

				if ((oldGradE_i * currentGradE_i) > 0) {

					_updateValue[offset + i] = min(_updateValue[offset + i] * _etaPlus, _updateMax);
					_delta[offset + i] = -sgn(currentGradE_i) * _updateValue[offset + i];
					params[i] += _delta[offset + i];
					oldGradE_i = currentGradE_i;

				}
				else if ((oldGradE_i * currentGradE_i) < 0) {

					params[i] -= _delta[offset + i];	//	The i-th updateValue of the previously epoch (backtracking)
					_updateValue[offset + i] = max(_updateValue[offset + i] * _etaMinus, _updateMin);
					oldGradE_i = 0;

				}
				else {	//	oldGradE_i * currentGradE_i = 0

					_delta[offset + i] = -sgn(currentGradE_i) * _updateValue[offset + i];
					params[i] += _delta[offset + i];
					oldGradE_i = currentGradE_i;

				}
			}

			//	Set new params
			netManager.SetAllParam_PerLayer(layer, matParams);

			offset += params.size();
		}
	}
	catch (InvalidParametersException e) {
		std::cout << e.getErrorMessage() << std::endl;
	}

}
