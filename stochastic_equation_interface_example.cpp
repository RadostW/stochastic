#include "equations.cpp"
#include <iostream>

void test(StochasticDifferentialEquation &x)
{
    std::cout << x.drift(101);
}

int main()
{
    SinhEquation q;
    test(q);    
}
