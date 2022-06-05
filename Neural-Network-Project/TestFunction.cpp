#include "TestFunction.h"

bool Test_GradientChecking(const NeuralNetworkFF& NN, const vector<Real>& gradToTest, 
	const ErrorFuncType EFuncType, const vector<Real>& input, const matrix<Real>& target) {

	bool successful{ false };

	//	Gradient checking
	auto net = NN;

	vector<Real> gradE_checking;
	vector<matrix<Real>> allParamsPerLayer;

	Real e = 0.0001;
	
	for (const auto& layer : RangeGen(0, net.GetNumLayers()))
		allParamsPerLayer.push_back(net.GetAllParamPerLayer(layer)); // Weights and biases

	//	From input layer to the Output layer
	for (const auto& layer : RangeGen(0, net.GetNumLayers())) {

		for (const auto& neuron : RangeGen(0, allParamsPerLayer[layer].size1())) {
			for (const auto& conn : RangeGen(0, allParamsPerLayer[layer].size2())) {

				Real d_E_ij;
				Real originalParam = allParamsPerLayer[layer](neuron, conn);
				Real param_plus_e = allParamsPerLayer[layer](neuron, conn) + e;
				Real param_minus_e = allParamsPerLayer[layer](neuron, conn) - e;

				matrix<Real> output_plus;
				matrix<Real> output_minus;

				// Compute output_plus
				net.SetParamPerNeuron(layer, neuron, conn, param_plus_e);
				auto temp_plus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

				output_plus.resize(temp_plus.size1(),1);
				for (size_t i = 0; i < temp_plus.size1(); i++)
					output_plus(i,0) = temp_plus(i, 0);

				auto error_plus = ErrorFunction::EFunction[EFuncType](output_plus, target);

				// Compute output_minus
				net.SetParamPerNeuron(layer, neuron, conn, param_minus_e);
				auto temp_minus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

				output_minus.resize(temp_minus.size1(),1);
				for (size_t i = 0; i < temp_minus.size1(); i++)
					output_minus(i, 0) = temp_minus(i, 0);

				auto error_minus = ErrorFunction::EFunction[EFuncType](output_minus, target);

				//	Compute dE/dw_ij
				d_E_ij = (error_plus - error_minus) / (2 * e);

				gradE_checking.push_back(d_E_ij);

				//	Restore default values
				net.SetParamPerNeuron(layer, neuron, conn, originalParam);

			}
		}
	}
	
	// Compare results

	Real gradE_quads{ 0 };
	for (auto d : gradToTest)
		gradE_quads += d * d;
	Real gradE_mag = sqrt(gradE_quads);

	Real gradE_check_quads{ 0 };
	for (auto d : gradE_checking)
		gradE_check_quads += d * d;
	Real gradE_checking_mag = sqrt(gradE_check_quads);

	Real gradE_diff_quads{ 0 };
	for (int i = 0; i < gradToTest.size(); i++)
		gradE_diff_quads += (gradToTest[i] - gradE_checking[i]) * (gradToTest[i] - gradE_checking[i]);
	Real difference_num = sqrt(gradE_diff_quads);

	Real difference_denum = gradE_mag + gradE_checking_mag;
	Real difference = (difference_num / difference_denum);

	if (difference < 1.0e-7)
		successful = true;
	else 
		successful = false;

	std::cout << "Gradient diff: " << difference << ". ";

	return successful;
}
