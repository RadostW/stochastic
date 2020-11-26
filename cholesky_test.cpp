#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "tdouble.cpp"
#include "matrix/cholesky.cpp"
#include "matrix/matrix.cpp"
 

void print_cholesky_factor(const matrix<tdouble>& matrix) {
    std::cout << std::fixed <<std::setprecision(2);
    std::cout << "Matrix:\n";
    print(std::cout, matrix);
    std::cout << "Cholesky factor:\n";
    print(std::cout, cholesky_factor(matrix));
}

int main() {
    std::cout << std::fixed <<std::setprecision(2);

    matrix<tdouble> matrix1(3, 3);
    matrix1(0,0) = tdouble(2,0);
    matrix1(0,1) = tdouble(1,0);
    matrix1(1,0) = tdouble(1,0);
    matrix1(1,1) = matrix1(1,1) + 1.;
    matrix1(2,2) = matrix1(2,2) + 1.;
    print(std::cout,matrix1);
    std::cout<<"Mat:\n";
    print(std::cout,matrix1);
    std::cout<<"Chol:\n";
    matrix<tdouble> chol = cholesky_factor(matrix1);
    print(std::cout,chol);
    std::cout<<"Prod:\n";
    print(std::cout,chol*chol.T());
    

    std::cout<<"\n";
    std::cout<<"\n";
    std::cout<<"\n";
    std::cout<<"\n";
    print_cholesky_factor(matrix1);
    std::cout<<"\n";

    return 0;
}
