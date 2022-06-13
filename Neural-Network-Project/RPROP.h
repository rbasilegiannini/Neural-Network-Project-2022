#pragma once
#include "NeuralNetworkManager.h"

/**
 * This class provides the abstractions to compute the ResilientPROP update rule.
 */
class RPROP {

public:
	RPROP(const size_t numParam, const Real initialUpdate);

	/**
	 *	This method performs the RPROP update.
	 * 
	 * \param netManager is the manager of the network.
	 * \param gradE is the gradient of the error.
	 */
	void Run(NeuralNetworkManager& netManager, const vec_r& gradE);

private:
	vec_r _oldGradE;
	vec_r _updateValue;
	Real _updateMax{ 50 };
	Real _updateMin{ 1e-6 };
	Real _etaMinus{ 0.5 };
	Real _etaPlus{ 1.2 };

};

