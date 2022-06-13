#include "RPROP.h"

using std::min;
using std::max;

RPROP::RPROP(const size_t numParam, const Real initialUpdate) {
	_oldGradE.resize(numParam, 0);
	_updateValue.resize(numParam, initialUpdate);
}

void RPROP::Run(NeuralNetworkManager& netManager, const vec_r& gradE) {
	
	vector <vec_r> netParams_PerLayer;
	
	//	Cache all network's parameters
	for (const auto& layer : RangeGen(0, netManager.GetNumLayers()))
		netParams_PerLayer.push_back(ConvertMatToArray(netManager.GetAllParam_PerLayer(layer)));

	//	Update parameters
	size_t layer{ 0 };
	size_t offset{ 0 };
	for (auto& params : netParams_PerLayer) {
		
		Real delta_i{ 0 };
		for (const auto& i : RangeGen(0, params.size())) {
			Real& oldGradE_i = _oldGradE[layer * offset + i];	//	It's a reference
			Real currentGradE_i = gradE[layer * offset + i];

			if ((oldGradE_i * currentGradE_i) > 0) {

				_updateValue[layer * offset + i] = min(_updateValue[layer * offset + i] * _etaPlus, _updateMax);
				delta_i = -sgn(currentGradE_i) * _updateValue[layer * offset + i];
				params[i] += delta_i;
				oldGradE_i = currentGradE_i;

			}
			else if ((oldGradE_i * currentGradE_i) < 0) {

				params[i] -= _updateValue[layer * offset + i];	//	The i-th updateValue of the previously epoch
				_updateValue[layer * offset + i] = max(_updateValue[layer * offset + i] * _etaMinus, _updateMin);
				oldGradE_i = 0;

			}
			else {

				delta_i = -sgn(currentGradE_i) * _updateValue[layer * offset + i];
				params[i] += delta_i;
				oldGradE_i = currentGradE_i;

			}
		}

		//	Set param
		//	netManager.SetAllParam_PerLayer(layer, matParam);


		offset = params.size();
		layer++;
	}


}
