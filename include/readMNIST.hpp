#ifndef READ_MNIST
#define READ_MNIST

#include <vector>
#include <string>

int reverseInt(int);

void readMNIST(int, int, std::vector<std::vector<float>>&, std::string);

void readLabels(int, std::vector<int>&, std::string);

#endif