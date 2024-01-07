#include <iostream>
#include <vector>
#include "../include/myLibrary.hpp"
#include "../include/readMNIST.hpp"
#include "../include/matrix.hpp"

std::vector<float> get_expected(int label, int possibilities){
    std::vector<float> expected;
    expected.resize(possibilities);

    for(int i = 0; i < possibilities; i++){
        if(i == label)
            expected[i] = .99;
        else
            expected[i] = 0.01;
    }

    return expected;
}

int main(){

    int s = 0;
    std::vector<std::vector<float>> ar;
    std::vector<int> labels;

    readMNIST(60000,784,ar, "./mnist/train-images.idx3-ubyte");
    readLabels(60000,labels, "./mnist/train-labels.idx1-ubyte");

    /*
    do {
        std::cout << std::endl;
        std::cout << "Introduce un número: ";
        std::cin >> s;
        std::cout << labels[s] << std::endl;

        for(int i = 1; i < ar[s].size(); i++){
            if(ar[s][i])
                std::cout << "*";
            else
                std::cout << "-";
            
            if(i % 28 == 0)
                std::cout << std::endl;
            
        }
        
    }while(s != 77); 
    */
    
    int input_nodes = 784;

    int hidden_nodes = 50;

    int output_nodes = 10;

    float learningRate = .1;

    matrix* weights1 = new matrix(hidden_nodes, input_nodes);
    matrix bias1(hidden_nodes, 1);

    matrix* weights2 = new matrix(output_nodes, hidden_nodes);
    matrix bias2(output_nodes, 1);

    //training
    for(int inputNum = 0; inputNum < ar.size(); inputNum++){

        matrix rawInput(ar[inputNum]);

        matrix inputMat = rawInput * (.99 / 255.0) + .01;

        //forward prop

        matrix hiddenValues = ((*weights1 * inputMat.transpose())).sigmoid();

        matrix outputValues = ((*weights2 * hiddenValues)).sigmoid();

        //expectedValues and error calculation
        matrix expectedValues(get_expected(labels[inputNum],output_nodes));
        
        matrix outputError = outputValues - expectedValues.transpose();
        matrix hiddenError = weights2->transpose() * outputError;

        //back prop

        matrix onesOut(output_nodes,1,1);
        matrix onesHidden(hidden_nodes,1,1);

        matrix w2grad = (outputError % outputValues % (onesOut - outputValues)) * hiddenValues.transpose(); 
        //matrix b2grad = outputError % outputValues % (outputValues * (-1) + 1);

        matrix w1grad = (hiddenError % hiddenValues % (onesHidden - hiddenValues)) * inputMat;
        //matrix b1grad = hiddenError % hiddenValues % (hiddenValues * (-1) + 1);

        *weights2 = *weights2 - w2grad * learningRate;   
        //bias2 = bias2 - b2grad * learningRate;

        *weights1 = *weights1 - w1grad * learningRate;
        //bias1 = bias1 - b1grad * learningRate; 
    
    }

    //testing
    std::vector<std::vector<float>> arTest;
    std::vector<int> labelsTest;

    readMNIST(10000,784,arTest, "./mnist/t10k-images.idx3-ubyte");
    readLabels(10000,labelsTest, "./mnist/t10k-labels.idx1-ubyte");

    std::cout << std::endl;
    std::cout << "Training done, beginning with testing...";
    do {
        std::cout << std::endl;
        std::cout << "Enter 1 to go to the next test number. Enter 0 to abort: ";
        std::cin >> s;

        //elegir un numero del 0 al 9999 al azar
        srand((unsigned) time(NULL));
        int random = (rand() % 9999);

        //mostrar el numero 
        if(s != 0){
            for(int i = 1; i < arTest[random].size(); i++){
                if(arTest[random][i])
                    std::cout << "*";
                else
                    std::cout << "-";
                
                if(i % 28 == 0)
                    std::cout << std::endl;
            }
        }

        //meter el número como input a la NN
        matrix rawInput(arTest[random]);

        matrix inputMat = rawInput * (.99 / 255.0) + .01;

        matrix hiddenValues = ((*weights1 * inputMat.transpose())).sigmoid();

        matrix outputValues = ((*weights2 * hiddenValues)).sigmoid();

        //mostrar el resultado
        std::cout << std::endl;

        std::cout << "The predicted number is: " << outputValues.columnMaxLoc(0) << std::endl;

        std::cout << "The correct number is: " << labelsTest[random] << std::endl;
        
    }while(s != 0); 

}