## Introduction
A simple project for the Neural Network course. The goal is to model the learning process using the Back Propagation algorithm and the RPROP update rule. The dataset used is [MNIST](http://yann.lecun.com/exdb/mnist/) and the programming language used is C++.

## Requirements

 - C++14
 - [Boost 1.79.0](https://www.boost.org/users/download/)
 - [OpenCV 4.6.0](https://opencv.org/releases/)
 - [pBPlots](https://github.com/InductiveComputerScience/pbPlots) (included)

## An overview of the architecture
```mermaid
classDiagram
	Client..>NeuralNetworkManager
	Client..>RPROP
	Client..>ReadMNIST
	Client..>AnalysisTools
	RPROP..>NeuralNetworkManager
	NeuralNetworkManager*--NeuralNetworkFF
	NeuralNetworkManager..> BackPropagation
	BackPropagation..> ActivationFunction
	BackPropagation..> ErrorFunction
	NeuralNetworkFF..>  ActivationFunction
