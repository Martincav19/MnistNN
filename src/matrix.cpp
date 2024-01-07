#include "../include/matrix.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>

matrix::matrix(std::vector<std::vector<float>> mat){
    this->mat = mat;
    this->rows = mat.size();
    this->cols = mat[0].size();
}

matrix::matrix(std::vector<float> vector){
    this->mat = {vector};
    this->rows = mat.size();
    this->cols = mat[0].size();
}

matrix::matrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    int randHelper = 0;

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            srand((unsigned) time(NULL) + randHelper);
            float random = ((1 + rand() % 900) - 450) / 1000.0;
            result[r][c] = random;
            randHelper++;
        }
    }

    this->mat = result;
}

matrix::matrix(int rows, int cols, float fillNumber){
    this->rows = rows;
    this->cols = cols;

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[r][c] = fillNumber;
        }
    }

    this->mat = result;
}

void matrix::print(){
    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            std::cout << std::fixed << std::setprecision(4) << this->mat[r][c] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrix::get_info(){
    std::cout << "Matrix has " << this->rows << " rows " << "and " << this->cols << " columns." << std::endl;
}

int matrix::columnMaxLoc(int col){

    int maxNumberLoc = 0;
    float maxNumber = 0;

    for(int r = 0; r < this->rows; r++){
        if(this->mat[r][col] > maxNumber){
            maxNumber = this->mat[r][col];
            maxNumberLoc = r;
        }
    }

    return maxNumberLoc;
}

matrix matrix::transpose(){

    std::vector<std::vector<float>> result;
    result.resize(this->cols, std::vector<float> (this->rows));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[c][r] = this->mat[r][c];;
        }
    }

    return result;
}

//matrix matrix::inverse(){}

matrix matrix::sigmoid(){

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[r][c] = 1 / (1 + exp(this->mat[r][c] * -1));
        }
    }

    return result;
}

matrix matrix::operator*(const matrix& m1){

    //la primera matriz r x c (this) debe tener el numero de columnas que la segunda matriz r1 x c1 (m1) tiene de renglones
    //es decir, c == r1. La matriz resultante será una matriz de tamaño r x c1. 

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (m1.cols));

    if(this->cols == m1.rows){

        for(int r = 0; r < this->rows; r++){
            for(int c = 0; c < m1.cols; c++){
                float cell = 0;
                for(int cr = 0; cr < this->cols; cr++){
                    cell += this->mat[r][cr] * m1.mat[cr][c];
                }
                result[r][c] = cell;
            }
        }
    }
    else{
        perror("Matrix multiplication error");
        abort();
    }

    return result;
}

matrix matrix::operator*(const float& a){

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[r][c] = this->mat[r][c] * a;
        }
    }

    return result;
}

matrix matrix::operator+(matrix const& m1){

    std::vector<std::vector<float>> result;
    result.resize(m1.rows, std::vector<float> (m1.cols));

    if(m1.rows == this->rows && m1.cols == this->cols){

        for(int r = 0; r < m1.rows; r++){
            for(int c = 0; c < m1.cols; c++){
                result[r][c] = this->mat[r][c] + m1.mat[r][c];
            }
        }
    }
    else{
        perror("Matrix sum error");
        abort();
    }

    return result; 
}

matrix matrix::operator+(const float& a){

    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[r][c] = this->mat[r][c] + a;
        }
    }

    return result;
}

matrix matrix::operator-(matrix const& m1){

    std::vector<std::vector<float>> result;
    result.resize(m1.rows, std::vector<float> (m1.cols));

    if(m1.rows == this->rows && m1.cols == this->cols){

        for(int r = 0; r < m1.rows; r++){
            for(int c = 0; c < m1.cols; c++){
                result[r][c] = this->mat[r][c] - m1.mat[r][c];
            }
        }
    }
    else{
        perror("Matrix substracion error");
        abort();
    }

    return result; 
}

matrix matrix::operator-(const float& a){
    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    for(int r = 0; r < this->rows; r++){
        for(int c = 0; c < this->cols; c++){
            result[r][c] = this->mat[r][c] - a;
        }
    }

    return result;
}

matrix matrix::operator%(const matrix& m1){
    
    std::vector<std::vector<float>> result;
    result.resize(this->rows, std::vector<float> (this->cols));

    if(this->rows == m1.rows && this->cols == m1.cols){


        for(int r = 0; r < this->rows; r++){
            for(int c = 0; c < this->cols; c++){
                result[r][c] = this->mat[r][c] * m1.mat[r][c];
            }
        }
    }
    else{
        perror("Matrix elementwise multiplication error");
        abort();
    }

    return result;
}
