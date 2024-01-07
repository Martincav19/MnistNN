#ifndef MAT_LIB
#define MAT_LIB

#include <vector>
#include <stdlib.h>
#include <stdio.h>

class matrix{
private:
    int rows;
    int cols;
    std::vector<std::vector<float>> mat;

public:
    matrix(std::vector<std::vector<float>>);
    matrix(std::vector<float>);
    matrix(int,int);
    matrix(int,int,float);


    matrix inverse();
    matrix transpose();
    matrix sigmoid();

    void print();
    void get_info();

    int columnMaxLoc(int);
    int rowMax(int);

    matrix operator+(const matrix& m1);
    matrix operator+(const float& a);
    matrix operator-(const matrix& m1);
    matrix operator-(const float& a);

    matrix operator*(const matrix& m1);
    matrix operator*(const float& a);
    matrix operator%(const matrix& m1);

};

#endif