#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>

using std::ifstream;
using boost::numeric::ublas::matrix;

typedef matrix<unsigned char> mat_c;

/**
 *	This data structure is used to memorize image-sample.
 */
struct ImageLabeled {
	mat_c image;
	unsigned char label{ 0 };

	ImageLabeled() { image.resize(28, 28); }
};


class ReadMNIST {

private:
	/**
	 * This function is used to reverse integer.
	 * 
	 * \param integer is the integer to reverse.
	 * \return The integer reversed.
	 */
	int _reverseInt(int integer);

};

