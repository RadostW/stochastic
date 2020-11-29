// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.

#include <vector>
#include <stdio.h>
#include "itoprocess.cpp"

double a=0.1;
tdouble a_term(tdouble x)
{
    if(x > 3)
    {
        return -0.5*a*a*exp(-x+2/3)*sech(x);
    }
    else if(x < -3)
    {
        return 0.5*a*a*exp(x+2/3)*sech(x);
    }
    else
    {
        return -0.5*a*a*tanh(x)*sech(x)*sech(x);
    }
}
tdouble b_term(tdouble x)
{
    return a*sech(x);
}


// stochastic equation to be solved:
// dX = a_term(x) dt + b_term(x) dW

int main()
{
    ItoProcess proc = ItoProcess(a_term, b_term);
    
    auto opts = proc.GetIntegrationOptions();
    opts.integratorType = IntegratorType::EulerMaruyama;
    proc.SetIntegrationOptions(opts);

    auto res = proc.SamplePath(0, 10);
    printf("Len = %d\n",(int)res.size());
}
