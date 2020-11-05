// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.

#include <vector>
#include <stdio.h>
#include "itoprocess.cpp"

/*
const double ceiling = 50.;
tdouble mobility(tdouble location)
{
    tdouble x = 1.0 / location;
    tdouble mobdown = 0.986292 - x - 0.00688 * cos(10.86762 + 8.092 * x) + 0.02057 * sin(2.506 + x * (3.074 + 2.227 * x));
    x = 1.0 / (ceiling - location);
    tdouble mobup = 0.986292 - x - 0.00688 * cos(10.86762 + 8.092 * x) + 0.02057 * sin(2.506 + x * (3.074 + 2.227 * x));
    return location > (0.5 * ceiling) ? mobup : mobdown;
}
tdouble dmobility(tdouble location)
{
    tdouble h = location;
    tdouble dmobdown = ((-0.09161878 - 0.06323218 * h) *
                            cos(2.506 + (2.227 + 3.074 * h) / (h * h)) +
                        h * (1. - 0.05567296 * sin(10.86762 + 8.092 / h))) /
                       (h * h * h);
    h = ceiling - location;
    tdouble dmobup = -((-0.09161878 - 0.06323218 * h) *
                           cos(2.506 + (2.227 + 3.074 * h) / (h * h)) +
                       h * (1. - 0.05567296 * sin(10.86762 + 8.092 / h))) /
                     (h * h * h);
    return location > (0.5 * ceiling) ? dmobup : dmobdown;
}
*/

/*
tdouble mobility(tdouble x)
{
    return 1.0-1.0/x;
}
tdouble dmobility(tdouble x)
{
    return 1.0/(x*x);
}

double kbT = 1;
double g = 1;
tdouble a_term(tdouble x)
{
    return  kbT * dmobility(x) - mobility(x)*g;
}
tdouble b_term(tdouble x)
{
    return sqrt(2 * kbT * mobility(x));
}
*/

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

    FILE *out;
    out = fopen("toplot.dat", "w");

    ItoProcess proc = ItoProcess(a_term, b_term);

    const int nproc = 1000;
    double x0 = 0;
    double tmax = 100;
    double dt = 50;

    std::vector<double> res[nproc];
    for(int i=0;i<nproc;i++)
    {
        res[i] = proc.SampleWagnerPlaten(x0, tmax, dt);
        proc.ResetRealization();
    }

    for(int i=tmax/dt - 100;i*dt<tmax;i++)
    {
        for(int j=0;j<nproc;j++)
        {
            fprintf(out,"%lf ",res[j][i]);
        }
        fprintf(out,"\n");
    }

    fclose(out);
}
