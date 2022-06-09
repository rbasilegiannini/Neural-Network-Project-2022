#pragma once

#include <exception>
#include <string>

class InvalidParametersException : public std::exception
{
private:
	std::string errorMsg;

public:
	InvalidParametersException(std::string _errorMsg) : errorMsg(_errorMsg) {};
	std::string getErrorMessage() const { return errorMsg; };
};