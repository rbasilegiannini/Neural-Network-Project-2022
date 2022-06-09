
#include "ReadMNIST.h"

int ReadMNIST::_reverseInt(int integer) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = integer & 255;
	ch2 = (integer >> 8) & 255;
	ch3 = (integer >> 16) & 255;
	ch4 = (integer >> 24) & 255;

	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}