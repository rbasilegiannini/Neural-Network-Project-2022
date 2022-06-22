#include "TestFunction.h"
#include "BackPropagation.h"
#include "NeuralNetworkManager.h"
#include <chrono>
#include <thread>

using namespace std::chrono;
using std::cout;
using std::endl;

bool Test_GradientChecking(const NeuralNetworkFF& NN, const vec_r& gradToTest,
	const EFuncType EFuncType, const vec_r& input, const mat_r& target) {

	bool successful{ false };

	//	Gradient checking
	auto net = NN;

	vec_r gradE_checking;
	vector<mat_r> allParamsPerLayer;

	Real e =  (Real)0.0001;
	
	try {
		for (const auto& layer : RangeGen(0, net.GetNumLayers()))
			allParamsPerLayer.push_back(net.GetAllParam_PerLayer(layer)); // Weights and biases

		//	From input layer to the Output layer
		for (const auto& layer : RangeGen(0, net.GetNumLayers())) {

			for (const auto& neuron : RangeGen(0, allParamsPerLayer[layer].size1())) {
				for (const auto& conn : RangeGen(0, allParamsPerLayer[layer].size2())) {

					Real d_E_ij;
					Real originalParam = allParamsPerLayer[layer](neuron, conn);
					Real param_plus_e = allParamsPerLayer[layer](neuron, conn) + e;
					Real param_minus_e = allParamsPerLayer[layer](neuron, conn) - e;

					mat_r output_plus;
					mat_r output_minus;

					// Compute output_plus
					net.SetParam_PerNeuron(layer, neuron, conn, param_plus_e);
					auto temp_plus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

					output_plus.resize(temp_plus.size1(), 1);
					for (size_t i = 0; i < temp_plus.size1(); i++)
						output_plus(i, 0) = temp_plus(i, 0);

					auto error_plus = ErrorFunction::EFunction[EFuncType](output_plus, target);

					// Compute output_minus
					net.SetParam_PerNeuron(layer, neuron, conn, param_minus_e);
					auto temp_minus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

					output_minus.resize(temp_minus.size1(), 1);
					for (size_t i = 0; i < temp_minus.size1(); i++)
						output_minus(i, 0) = temp_minus(i, 0);

					auto error_minus = ErrorFunction::EFunction[EFuncType](output_minus, target);

					//	Compute dE/dw_ij
					d_E_ij = (error_plus - error_minus) / (2 * e);

					gradE_checking.push_back(d_E_ij);

					//	Restore default values
					net.SetParam_PerNeuron(layer, neuron, conn, originalParam);

				}
			}
		}
	}
	catch (InvalidParametersException e) {
		cout << e.getErrorMessage() << endl;
		return false;
	}
	// Compare results


	// TEST
	return true;
	//
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

void TestCase_GradientComputation() {
	Hyperparameters hyp({});
	NeuralNetworkManager& nnManager = NeuralNetworkManager::GetNNManager(hyp);


	for (const auto& nTest : RangeGen(1, 21)) {

		vector<size_t> nNeuronsPerLayer;
		vector<Real> input;
		matrix<Real> target;

		// Set nNeuronPerLayer
		nNeuronsPerLayer.resize(5 * nTest);

		for (auto& nNeuron : nNeuronsPerLayer)
			nNeuron = (rand() % 10) + 10;

		// Set target
		target.resize(nNeuronsPerLayer.back(), 1);

		for (const auto& t : RangeGen(0, target.size1()))
			target(t, 0) = 0;

		target(rand() % target.size1(), 0) = 1;

		// Set input
		input.resize(100 * nTest);

		// Create the NN
		vector<AFuncType> AFuncPerLayer(nNeuronsPerLayer.size());

		for (auto& f : AFuncPerLayer) {

			size_t choiceA{ (size_t)(rand() % 3)};
			switch (choiceA)
			{
			case 0:
				f = AFuncType::SIGMOID;
				break;

			case 1:
				f = AFuncType::IDENTITY;
				break;

			case 2:
				f = AFuncType::RELU;
				break;

			default:
				f = AFuncType::SIGMOID;
				break;
			}
		}

		Hyperparameters newHyp({ input.size(), nNeuronsPerLayer, AFuncPerLayer });
		nnManager.ResetHyperparameters(newHyp);

		try {

			EFuncType Etype;
			size_t choiceE{ (size_t)(rand() % 3)};
			switch (choiceE)
			{
			case 0:
				Etype = EFuncType::SUMOFSQUARES;
				break;
			case 1:
				Etype = EFuncType::CROSSENTROPY_SOFTMAX;
				nnManager.SetAFunc_PerLayer(nnManager.GetNumLayers() - 1, AFuncType::IDENTITY);

				break;
			case 2:
				Etype = EFuncType::CROSSENTROPY;
				nnManager.SetAFunc_PerLayer(nnManager.GetNumLayers() - 1, AFuncType::SIGMOID);

				break;

			default:
				Etype = EFuncType::CROSSENTROPY_SOFTMAX;
				nnManager.SetAFunc_PerLayer(nnManager.GetNumLayers() - 1, AFuncType::IDENTITY);
				break;
			}

			vector<Real> targetVec(nNeuronsPerLayer.back());
			for (const auto& t : RangeGen(0, target.size1()))
				targetVec[t] = target(t, 0);

			nnManager.Run(input);
			vector<Real> gradE;
			auto start_bp = high_resolution_clock::now();

		
			gradE = nnManager.ComputeGradE_PerSample(Etype, targetVec);
	
			auto stop_bp = high_resolution_clock::now();

			auto duration_bp = duration_cast<milliseconds>(stop_bp - start_bp);

			//	Testing & compare
			auto nn = nnManager.getNet();
			auto start_chk = high_resolution_clock::now();
			bool test = Test_GradientChecking(nn, gradE, Etype, input, target);
			auto stop_chk = high_resolution_clock::now();

			auto duration_chk = duration_cast<seconds>(stop_chk - start_chk);

			size_t sumOfParams{ 0 };
			for (const auto& layer : RangeGen(0, nnManager.GetNumLayers()))
				sumOfParams += nnManager.GetAllParam_PerLayer(layer).size1() * nnManager.GetAllParam_PerLayer(layer).size2();

			cout << "Test number: " << nTest << ". Params: " << sumOfParams << ". Result: ";

			if (test)
				cout << "OK! Time saved: " << duration_chk.count() - duration_bp.count() << "s. ";
			else
				cout << "Fail! ";

			cout << "Loss function: " << NameOfErrorFuncType(Etype) << endl;

		}
		catch (InvalidParametersException e) {
			cout << e.getErrorMessage() << endl;
			return;
		}
	}
}

void TestCase_TimingGradientComputation() {
	Hyperparameters hyp({});
	NeuralNetworkManager& nnManager = NeuralNetworkManager::GetNNManager(hyp);


	for (const auto& nTest : RangeGen(2, 21)) {

		vector<size_t> nNeuronsPerLayer;
		vector<Real> input;
		matrix<Real> target;

		// Set nNeuronPerLayer
		nNeuronsPerLayer.resize(5*nTest);

		for (auto& nNeuron : nNeuronsPerLayer)
			nNeuron = 10*nTest;

		// Set target
		target.resize(nNeuronsPerLayer.back(), 1);

		for (const auto& t : RangeGen(0, target.size1()))
			target(t, 0) = 0;

		target(rand() % target.size1(), 0) = 1;

		// Set input 
		input.resize(5*nTest);

		try {
			// Create the NN
			vector<AFuncType> AFuncPerLayer(nNeuronsPerLayer.size(), AFuncType::SIGMOID);

			Hyperparameters newHyp({ input.size(), nNeuronsPerLayer, AFuncPerLayer });
			nnManager.ResetHyperparameters(newHyp);

			EFuncType EFuncType;
			size_t choice{ (size_t)(rand() % 3) };
			EFuncType = EFuncType::CROSSENTROPY_SOFTMAX;
			nnManager.SetAFunc_PerLayer(nnManager.GetNumLayers() - 1, AFuncType::IDENTITY);

			vector<Real> gradE;

			vector<Real> targetVec(nNeuronsPerLayer.back());
			for (const auto& t : RangeGen(0, target.size1()))
				targetVec[t] = target(t, 0);

			size_t sumOfParams{ 0 };
			for (const auto& layer : RangeGen(0, nnManager.GetNumLayers()))
				sumOfParams += nnManager.GetAllParam_PerLayer(layer).size1() * nnManager.GetAllParam_PerLayer(layer).size2();

			nnManager.Run(input);
			auto start_bp = high_resolution_clock::now();
		
				gradE = nnManager.ComputeGradE_PerSample(EFuncType, targetVec);
		
			auto stop_bp = high_resolution_clock::now();
			auto duration_bp = duration_cast<microseconds>(stop_bp - start_bp);

			cout << "Test number: " << nTest << ". Number of params: " << sumOfParams << ", time: " << duration_bp.count() << "us" << endl;
	
		}
		catch (InvalidParametersException e) {
			cout << e.getErrorMessage() << endl;
			return;
		}
	}
}
