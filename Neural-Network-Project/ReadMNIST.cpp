#include "ReadMNIST.h"
#include "Utility.h"
#include <fstream>
#include <exception>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using cv::Mat;
using cv::imshow;
using cv::resize;
using std::array;

vector<ImageLabeled> ReadSample(const string& imagesPath, const string& labelsPath, const size_t numSamples) {

	auto reverseInt = [](int32_t toReverse) {
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = toReverse & 255;
		ch2 = (toReverse >> 8) & 255;
		ch3 = (toReverse >> 16) & 255;
		ch4 = (toReverse >> 24) & 255;

		return((int32_t)ch1 << 24) + ((int32_t)ch2 << 16) + ((int32_t)ch3 << 8) + ch4;
	};

	int32_t magicNumber_image{ 0 };
	int32_t magicNumber_label{ 0 };
	int32_t maxNumlabel{ 0 };
	int32_t maxNumImages{ 0 };
	int32_t rows{ 0 };
	int32_t cols{ 0 };

	ifstream imagesFile(imagesPath, std::ios::binary);
	ifstream labelsFile(labelsPath, std::ios::binary);

	if (!imagesFile.is_open() || !labelsFile.is_open()) 
		throw std::runtime_error("[READSAMPLE] Error to open file.");


	//	Read ImagesFile header
	imagesFile.read((char*)&magicNumber_image, sizeof(magicNumber_image));
	magicNumber_image = reverseInt(magicNumber_image); //MSB first

	if (magicNumber_image != 2051)
		throw std::runtime_error("[READSAMPLE] Images file not valid.");

	imagesFile.read((char*)&maxNumImages, sizeof(maxNumImages));
	imagesFile.read((char*)&rows, sizeof(rows));
	imagesFile.read((char*)&cols, sizeof(cols));

	maxNumImages = reverseInt(maxNumImages);
	rows = reverseInt(rows);
	cols = reverseInt(cols);

	if (numSamples > maxNumImages)
		throw std::runtime_error("[READSAMPLE] There are not enough images.");


	//	Read LabelsFile header
	labelsFile.read((char*)&magicNumber_label, sizeof(magicNumber_label));
	magicNumber_label = reverseInt(magicNumber_label);

	if (magicNumber_label != 2049) 
		throw std::runtime_error("[READSAMPLE] Labels file not valid.");

	labelsFile.read((char*)&maxNumlabel, sizeof(maxNumlabel));
	maxNumlabel = reverseInt(maxNumlabel);

	if (numSamples > maxNumlabel)
		throw std::runtime_error("[READSAMPLE] There are not enough labels.");


	//	Read samples
	vector<ImageLabeled> samples(numSamples);

	for (auto& sample : samples) {

		//	Read the sample's image
		for (const auto& row : RangeGen(0, rows)) {
			for (const auto& col : RangeGen(0, cols)) {
				unsigned char pixel{ 0 };
				imagesFile.read((char*)&pixel, sizeof(pixel));
				sample.image(row, col) = (uint8_t)pixel;
			}
		}

		//	Read the sample's label
		unsigned char label{ 0 };
		labelsFile.read((char*)&label, sizeof(label));
		sample.label = (uint8_t)label;
	}

	imagesFile.close();
	labelsFile.close();

	return samples;
}

vector<uint8_t> RetrieveMinMaxFromDatasetRaw(const vector<ImageLabeled>& dataset) {
	vector<uint8_t> minmax(2);

	vector<uint8_t> arrMax;
	vector<uint8_t> arrMin;

	for (const auto& s : dataset) {
		uint8_t sampleMax = *max_element(s.image.data().begin(), s.image.data().end());
		uint8_t sampleMin = *min_element(s.image.data().begin(), s.image.data().end());

		arrMax.push_back(sampleMax);
		arrMin.push_back(sampleMin);
	}

	uint8_t min = *min_element(arrMin.begin(), arrMin.end());
	uint8_t max = *max_element(arrMax.begin(), arrMax.end());

	minmax[0] = min;
	minmax[1] = max;

	return minmax;
}

void ResizeDatasetRaw (vector<ImageLabeled>& dataset, double factor) {
	
	for (auto& sample : dataset) {

		mat_i& imageSample = sample.image;	// It's a reference

		//	Build an image for OpenCV tools
		array<array<uint8_t, 28>, 28> imageRaw;
		for (const auto& row : RangeGen(0, imageSample.size1())) {
		
			array<uint8_t, 28> image_row;
			for (const auto& col : RangeGen(0, imageSample.size2())) 
				image_row[col] = imageSample(row, col);

			imageRaw[row] = image_row;
		}

		Mat image(28, 28, CV_8UC1, &imageRaw);

		//	Resize image 
		Mat imageResized;
		resize(image, imageResized, cv::Size(), factor, factor);

		//	Resize the matrix
		auto oldRows = imageSample.size1();
		auto oldCols = imageSample.size2();
		imageSample.resize(factor *oldRows, factor *oldCols);

		for (const auto& row : RangeGen(0, imageSample.size1())) {
			for (const auto& col : RangeGen(0, imageSample.size2())) {
				imageSample(row, col) = imageResized.at<uint8_t>(row, col);
			}
		}
	}
}
