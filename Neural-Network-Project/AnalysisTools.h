#pragma once

#include "pbPlots/pbPlots.hpp"
#include "pbPlots/supportLib.hpp"
#include "Utility.h"

using std::string;
using std::max_element;

template<typename T>
inline void SavePlot(const string& namePlot, const vector<T>& x_axis, const vector<T>& y_axis) {

	vector<double> x_axis_plot;
	vector<double> y_axis_plot;
	StringReference* errorMessage = new StringReference();

	//	Conversion Real to double
	for (const auto& x : x_axis)
		x_axis_plot.push_back((double)x);
	for (const auto& y : y_axis)
		y_axis_plot.push_back((double)y);

	ScatterPlotSeries* series = GetDefaultScatterPlotSeriesSettings();
	series->xs = &x_axis_plot;
	series->ys = &y_axis_plot;
	series->linearInterpolation = true;
	series->lineType = toVector(L"solid");
	series->lineThickness = 2;
	series->color = GetBlack();

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 800;
	settings->height = 400;
	settings->autoBoundaries = false;
	settings->xMin = 0;
	settings->xMax = x_axis_plot.back() + 5;
	settings->yMin = 0;
	settings->yMax = *max_element(y_axis_plot.begin(), y_axis_plot.end()) + y_axis_plot.back() * 0.1;
	settings->autoPadding = true;
	settings->xLabel = toVector(L"Epoch");
	settings->yLabel = toVector(L"Error");
	settings->scatterPlotSeries->push_back(series);

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	DrawScatterPlotFromSettings(imageReference, settings, errorMessage);
	//	Save in png
	vector<double>* pngData = ConvertToPNG(imageReference->image);
	WriteToFile(pngData, namePlot + ".png");
	DeleteImage(imageReference->image);
}
