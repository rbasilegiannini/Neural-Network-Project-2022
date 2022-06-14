#pragma once

#include "pbPlots/pbPlots.hpp"
#include "pbPlots/supportLib.hpp"
#include "Utility.h"

using std::string;
using std::max_element;

template<typename T>
inline bool SavePlot(const string& namePlot, const vector<T>& x_axis, const vector<T>& y_axis) {

	bool success;
	StringReference* errorMessage = new StringReference();
	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	vector<double> xs{ x_axis };
	vector<double> ys{ y_axis };

	ScatterPlotSeries* series = GetDefaultScatterPlotSeriesSettings();
	series->xs = &xs;
	series->ys = &ys;
	series->linearInterpolation = true;
	series->lineType = toVector(L"solid");
	series->lineThickness = 1;
	series->color = GetGray(0.3);

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 1000;
	settings->height = 1000;
	settings->autoBoundaries = false;
	settings->xMin = 0;
	settings->xMax = xs.back() + 5;
	settings->yMin = 0;
	settings->yMax = *max_element(ys.begin(), ys.end());
	settings->autoPadding = true;
	settings->xLabel = toVector(L"Epoch");
	settings->yLabel = toVector(L"Error");
	settings->scatterPlotSeries->push_back(series);

	success = DrawScatterPlotFromSettings(imageReference, settings, errorMessage);
	
	//	Save in png
	if (success) {
		vector<double>* pngData = ConvertToPNG(imageReference->image);
		WriteToFile(pngData, namePlot + ".png");
	}
	return success;
}
