#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "tdouble.cpp"
#include "cholesky/cholesky.cpp"
#include "cholesky/matrix.cpp"
 

void print_cholesky_factor(const matrix<tdouble>& matrix) {
    std::cout << std::fixed <<std::setprecision(2);
    std::cout << "Matrix:\n";
    print(std::cout, matrix);
    std::cout << "Cholesky factor:\n";
    print(std::cout, cholesky_factor(matrix));
}

int main() {
    matrix<tdouble> matrix1(3, 3);
    matrix1(0,0) = tdouble(2,0);
    matrix1(0,1) = tdouble(1,0);
    matrix1(0,2) = tdouble(1,0);
    matrix1(1,1) = matrix1(1,1) + 1.;
    matrix1(2,2) = matrix1(2,2) + 1.;
    print_cholesky_factor(matrix1);
    std::cout<<"\n";

    return 0;
}
