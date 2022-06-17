#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <string>
#include <vector>

using std::ifstream;
using std::string;
using std::vector;
using boost::numeric::ublas::matrix;

typedef matrix<uint8_t> mat_i;

/**
 *	This data structure is used to memorize image-sample.
 */
struct ImageLabeled {
	mat_i image;
	uint8_t label{ 0 };

	ImageLabeled() { image.resize(28, 28); }
};

/**
 *	This function is able to extract a number of samples from the MNIST dataset.
 *
 * \param	imagesPath is the the filepath of the images file.
 * \param	labelsPath is the the filepath of the labels file.
 * \param	numSamples is the number of samples to extract.
 * \return	a vector with sample data structures.
 */
vector<ImageLabeled> ReadSample(const string& imagesPath, const string& labelsPath, const size_t numSamples);

/**
 * This function retrievs the min and max value from a collection of ImageLabeled.
 * 
 * \param	dataset is the collection of ImageLabeled
 * \return	a vector of two elements: the first is the min, the second is the max.
 */
vector<uint8_t> RetrieveMinMaxFromDatasetRaw(const vector<ImageLabeled>& dataset);




