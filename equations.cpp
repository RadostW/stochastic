#include "tdouble.cpp"


// Interfaces to talk to the SDE solver

// Stochastic equation defined as:
// dX = \mu dt + \sigma dW
// \mu(X) is given by drift(x)
// \sigma(X) is given by volatility(x)
class StochasticDifferentialEquation
{
  public:
    virtual tdouble drift(tdouble x) = 0;
    virtual tdouble volatility(tdouble x) = 0;    
};
// Child class with added exact solution
class ExactStochasticDifferentialEquation : public StochasticDifferentialEquation
{
  public:
    virtual double exactSolution(double x0,double w,double t) = 0;
};

// Explicit implementation of an equation with exact solution
class SinhEquation : public ExactStochasticDifferentialEquation
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
    double exactSolution(double x0, double w, double t)
    {
        return asinh(a*w + sinh(x0)); 
    }
};
