#include "tdouble.cpp"

/*
TODO: proper visiblity and inheritance
class Equation 
{
    tdouble (*drift)(tdouble);
    tdouble (*volatility)(tdouble);
    double x0;
    double tmax;
    double (*exact_solution)(double x0, double w);
};
*/

// TODO: proper visiblity and inheritance
class SinhEquation
{
    public:

    double a=0.1;
    tdouble drift(tdouble x)
    {
        if(x > 3)
            return -0.5*a*a*exp(-x+2/3)*sech(x);
    
        else if(x < -3)
            return 0.5*a*a*exp(x+2/3)*sech(x);
        else
            return -0.5*a*a*tanh(x)*sech(x)*sech(x);
    }
    tdouble volatility(tdouble x)
    {
        return a*sech(x);
    }
    double exactSolution(double x0, double w)
    {
        return asinh(a*w + sinh(x0)); 
    }
};
