// Neural-Network-Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "ActivationFunction.h"

using namespace boost::numeric::ublas;


int main() {

	// Test sigmoid function
	Real x = 1;
	for (int i = 0; i < 200; i++) {
		std::cout << "x: " <<  x << ", sig(x): " << Sigmoid(x) << std::endl;
		x += 0.1;
	}
}