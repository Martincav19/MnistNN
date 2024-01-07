#include "../include/NN.hpp"
#include "../include/matrix.hpp"
#include <vector>

NN::NN(int input_nodes,int hidden_nodes,int output_nodes,float learningRate){

    this->input_nodes = input_nodes;
    this->hidden_nodes = hidden_nodes;
    this->output_nodes = output_nodes;
    this->learningRate = learningRate;

    this->weights1 = matrix(hidden_nodes, input_nodes);
    this->weights2 = matrix(output_nodes, hidden_nodes);
}

matrix NN::get_expected_output_matrix(int label){
    std::vector<float> expected;
    expected.resize(this->output_nodes);

    for(int i = 0; i < this->output_nodes; i++){
        if(i == label)
            expected[i] = .99;
        else
            expected[i] = 0.01;
    }

    matrix expectedValues = matrix(expected);
    return expectedValues;
}


matrix NN::query(matrix m1){
    
    matrix hiddenValues = ((this->weights1 * m1.transpose())).sigmoid();

    matrix outputValues = ((this->weights2 * hiddenValues)).sigmoid();
    return outputValues;
}

void NN::train(matrix trainInput, int trainLabel){

    matrix hiddenValues = ((this->weights1 * trainInput.transpose())).sigmoid();

    matrix outputValues = ((this->weights2 * hiddenValues)).sigmoid();

    matrix expectedValues = get_expected_output_matrix(trainLabel);

    matrix outputError = outputValues - expectedValues.transpose();
    matrix hiddenError = this->weights2.transpose() * outputError;

    //back prop

    matrix onesOut(output_nodes,1,1);
    matrix onesHidden(hidden_nodes,1,1);

    matrix w2grad = (outputError % outputValues % (onesOut - outputValues)) * hiddenValues.transpose(); 
    //matrix b2grad = outputError % outputValues % (outputValues * (-1) + 1);

    matrix w1grad = (hiddenError % hiddenValues % (onesHidden - hiddenValues)) * trainInput;
    //matrix b1grad = hiddenError % hiddenValues % (hiddenValues * (-1) + 1);

    this->weights2 = this->weights2 - w2grad * learningRate;   
    //bias2 = bias2 - b2grad * learningRate;

    this->weights1 = this->weights1 - w1grad * learningRate;
}