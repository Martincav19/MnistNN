#include <iostream>
#include <vector>
#include "../include/readMNIST.hpp"
#include "../include/matrix.hpp"
#include "../include/NN.hpp"

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
    NN neuralNetwork(784,50,10,.1);

    //matrix bias1(hidden_nodes, 1);

    //matrix bias2(output_nodes, 1);

    //training
    for(int inputNum = 0; inputNum < ar.size(); inputNum++){

        matrix rawInput(ar[inputNum]);

        matrix inputMat = rawInput * (.99 / 255.0) + .01;

        neuralNetwork.train(inputMat,labels[inputNum]);
    
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

        matrix outputValues = neuralNetwork.query(inputMat);

        //mostrar el resultado
        std::cout << std::endl;

        std::cout << "The predicted number is: " << outputValues.columnMaxLoc(0) << std::endl;

        std::cout << "The correct number is: " << labelsTest[random] << std::endl;
        
    }while(s != 0); 

}