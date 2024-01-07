#ifndef NN_LIB
#define NN_LIB

#include "matrix.hpp"

class NN{
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    float learningRate;

    matrix weights1;
    matrix weights2;

    matrix get_expected_output_matrix(int);

public:
    NN(int,int,int,float);

    matrix query(matrix);
    void train(matrix,int);

};

#endif