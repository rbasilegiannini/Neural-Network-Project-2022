// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <functional>
#include <numeric>
#include <chrono>
#include <thread>

#include "NeuralNetworkFF.h"
#include "ErrorFunction.h"
#include "BackPropagation.h"
#include "TestFunction.h"
#include "NeuralNetworkManager.h"
#include "ReadMNIST.h"

using std::cout;
using std::endl;
using std::vector;
using boost::numeric::ublas::matrix;
using boost::numeric::ublas::subrange;
using namespace std::chrono;

int main() {

	string imagesFiles = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-images.idx3-ubyte";
	string labelsFile = "C:\\Users\\RBG94\\Desktop\\Progetto reti\\train-labels.idx1-ubyte";

	auto samples = ReadSample(imagesFiles, labelsFile, 10000);

	for (const auto& sample : samples) {

		cout << "IMAGE: " << endl << endl;
		for (const auto& row : RangeGen(0, sample.image.size1())) {
			for (const auto& col : RangeGen(0, sample.image.size2())) {
				int temp = sample.image(row, col);
				if (temp > 0)
					cout << 1;
				else
					cout << " ";
			}
			cout << endl;
		}
		cout << endl << "LABEL: " << (unsigned)sample.label << endl << endl;
	}

	cout << endl;

}
