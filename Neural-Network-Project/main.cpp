// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"
#include "BackPropagation.h"

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <functional>
#include <numeric>

using std::cout;
using std::endl;
using std::vector;
using boost::numeric::ublas::matrix;


int main() {


	const vector<size_t> _numNeuronsPerLayer{ 3, 1};
	const vector<Real> input { 1, 1 };

	NeuralNetworkFF net (2, _numNeuronsPerLayer);

	vector<matrix<Real>> weights(2);
	weights[0].resize(3, 2);
	weights[1].resize(1, 3);
	Real add{ 0 };
	// W1
	weights[0](0, 0) = -0.66016222 + add;
	weights[0](0, 1) = 0.77883251 + add;

	weights[0](1, 0) = 0.02856532 + add;
	weights[0](1, 1) = 0.1876014 + add;

	weights[0](2, 0) = -0.65833175 + add;
	weights[0](2, 1) = -0.43713064 + add;

	// W2
	weights[1](0, 0) = -0.86631423 + add;
	weights[1](0, 1) = -0.2239644 + add;
	weights[1](0, 2) = -0.48749108 + add;

	for (size_t i{ 0 }; i < _numNeuronsPerLayer.size(); i++) {
//		net.SetWeights(i, weights[i]);

		vector<Real> vecZero(_numNeuronsPerLayer[i], 0);
		net.SetBias(i, vecZero);	// Bias to zero
	}

//	net.PrintNetwork();

#pragma region	Test compute network	
/*
	auto result = net.ComputeNetwork(input);	// NetOutput = 0.33
	for (int i = 0; i < result.neuronsOutputPerLayer.back().size1(); i++)
		cout << result.neuronsOutputPerLayer.back()(i,0) << ' ';

	cout << endl;
*/
#pragma endregion

#pragma region Test EFunc (Sum of squares)
/*	
	vector<Real> output { 5, 3, 2, 7, 2, 4 };
	vector<Real> target1{ 5+0.3, 3+0.15, 2+1, 7-0.25, 2+0.55, 4+1.1 };
	vector<Real> target2{ 5, 3 + 0.15, 2 + 1, 7, 2, 4 + 0.1 };
	vector<Real> target3{ 5, 3, 2, 7, 2, 4 };
	vector<Real> target4{ 5+1, 3+1, 2+1, 7+1, 2+1, 4+1 };

	cout << "Output to validate: ";
	for (const auto& o : output) cout << o << " ";
	cout << endl;

	cout << "Error with target 1: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target1) << endl;

	cout << "Error with target 2: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target2) << endl;

	cout << "Error with target 3: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target3) << endl;

	cout << "Error with target 4: ";
	cout << ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output, target4) << endl;

	cout << endl;
*/
#pragma endregion

#pragma region Test Gradient computation
	 
	// FP step
	auto netResult = net.ComputeNetwork(input);
	auto activationsPerLayer = netResult.activationsPerLayer;
	vector<matrix<Real>> neuronsOutputPerLayer = netResult.neuronsOutputPerLayer;
	
	// BP step
	vector<size_t> allNeuronsNumber;
	vector<AFuncType> allAFunc;
	vector<matrix<Real>> paramsPerLayer;

	for (size_t i = 0; i < net.GetNumLayers(); i++) {	// For each layer
		allNeuronsNumber.push_back(net.GetNumNeuronsPerLayer(i));
		allAFunc.push_back(net.GetAFuncPerLayer(i));
		paramsPerLayer.push_back(net.GetWeightsPerLayer(i)); // Only weights
		
	}

	DataFromNetwork dataNN{
		net.GetNumLayers(),
		allNeuronsNumber,
		activationsPerLayer,
		paramsPerLayer,
		allAFunc,
		neuronsOutputPerLayer.back()
	};
	
	vector<Real> target{ 1 };
	vector<vector<Real>> allDelta = BackPropagation::BProp(dataNN, ErrorFuncType::SUMOFSQUARES, target);

	cout << "Deltas: " << endl;
	for (const auto& vecD : allDelta) {
		for (const auto& d : vecD) {
			cout << d << ' ';
		}
		cout << endl;
	}

	// Compute gradient of E

	vector<Real> allParams;
	for (const auto& m : paramsPerLayer) {

		for (size_t i = 0; i < m.size1(); i++) {
			for (size_t j = 0; j < m.size2(); j++) {
				allParams.push_back(m(i, j));
			}
		}
	}

	vector<Real> gradE;

	//	From input layer to the Output layer
	for (size_t layer = 0; layer < net.GetNumLayers(); layer++) {

		for (size_t idxNeuron = 0; idxNeuron < paramsPerLayer[layer].size1(); idxNeuron++) {
			for (size_t idxConnection = 0; idxConnection < paramsPerLayer[layer].size2(); idxConnection++) {
				Real d_E_ij;

				//	Compute dE/dw_ij
				if (layer == 0)
					d_E_ij = allDelta[layer][idxNeuron] * input[idxConnection];
				else
					d_E_ij = allDelta[layer][idxNeuron] * neuronsOutputPerLayer[layer - 1](idxConnection);

				gradE.push_back(d_E_ij);
			}
		}
	}

	// Test Gradient Checking 

	/**
	 * Prendo questi 9 parametri. Ogni parametro lo perturbo con epsilon, uno alla volta. Ad ogni perturbazione,
	 * calcolo l'output della rete. Misuro l'errore. Calcolo la differenza dell'errore con perturbazione meno l'errore
	 * senza perturbazione e divido per 2epsilon.
	 * 
	 * \return 
	 */

	vector<Real> gradE_checking;
	Real e = 0.0001;

	cout << "Default network: " << endl;
	net.PrintNetwork();

	//	From input layer to the Output layer
	for (size_t layer = 0; layer < net.GetNumLayers(); layer++) {

		for (size_t idxNeuron = 0; idxNeuron < paramsPerLayer[layer].size1(); idxNeuron++) {
			for (size_t idxConnection = 0; idxConnection < paramsPerLayer[layer].size2(); idxConnection++) {

				cout << "Default network (l: " << layer << ", n: " << idxNeuron << ", c: " << idxConnection << "): " << endl;
				net.PrintNetwork();

				Real d_E_ij;
				Real originalParam = paramsPerLayer[layer](idxNeuron, idxConnection);
				Real param_plus_e = paramsPerLayer[layer](idxNeuron, idxConnection) + e;
				Real param_minus_e = paramsPerLayer[layer](idxNeuron, idxConnection) - e;

				vector<Real> output_plus;
				vector<Real> output_minus;

				// Compute output_plus
				net.SetWeightPerNeuron(layer, idxNeuron, idxConnection, param_plus_e);
				auto temp_plus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

				output_plus.resize(temp_plus.size1());
				for (size_t i = 0; i < temp_plus.size1(); i++)
					output_plus[i] = temp_plus(i, 0);

				auto error_plus = ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output_plus, target);

				cout << "[+]Change weight (" << idxNeuron << "," << idxConnection << ") of layer " << layer << ". New NN: " << endl;
				net.PrintNetwork();

				// Compute output_minus
				net.SetWeightPerNeuron(layer, idxNeuron, idxConnection, param_minus_e);
				auto temp_minus = net.ComputeNetwork(input).neuronsOutputPerLayer.back();

				output_minus.resize(temp_minus.size1());
				for (size_t i = 0; i < temp_minus.size1(); i++)
					output_minus[i] = temp_minus(i, 0);

				auto error_minus = ErrorFunction::EFunction[ErrorFuncType::SUMOFSQUARES](output_minus, target);

				cout << "[-]Change weight (" << idxNeuron << "," << idxConnection << ") of layer " << layer << ". New NN: " << endl;
				net.PrintNetwork();

				//	Compute dE/dw_ij
				d_E_ij = (error_plus - error_minus) / (2 * e);

				gradE_checking.push_back(d_E_ij);

				//	Restore default values
				net.SetWeightPerNeuron(layer, idxNeuron, idxConnection, originalParam);
			}
		}
	}

	cout << "GradE: " << endl;
	for (const auto& d : gradE)
		cout << d << " ";
	cout << endl << endl;

	cout << "GradE (checking): " << endl;
	for (const auto& d : gradE_checking)
		cout << d << " ";
	cout << endl;

	Real gradE_quads{0};
	for (auto d : gradE)
		gradE_quads += d * d;
	Real gradE_mag = sqrt(gradE_quads);

	Real gradE_check_quads{ 0 };
	for (auto d : gradE_checking)
		gradE_check_quads += d * d;
	Real gradE_checking_mag = sqrt(gradE_check_quads);

	Real gradE_diff_quads{ 0 };
	for (int i = 0; i < gradE.size(); i++)
		gradE_diff_quads += (gradE[i] - gradE_checking[i]) * (gradE[i] - gradE_checking[i]);
	Real difference_num = sqrt(gradE_diff_quads);

	auto difference_denum = gradE_mag + gradE_checking_mag;

	cout << "The difference: " << (difference_num / difference_denum) << endl;


	cout << endl;

#pragma endregion

}