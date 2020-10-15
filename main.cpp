#include <vector>
#include <stdio.h>
#include "pmath.cpp"
#include "tdouble.cpp"
#include "itoprocess.cpp"

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

tdouble a_term(tdouble x)
{
    return dmobility(x);
}
tdouble b_term(tdouble x)
{
    return sqrt(2 * mobility(x));
}
// stochastic equation to be solved:
// dX = a_term(x) dt + b_term(x) dW

int main()
{

    FILE *out;
    out = fopen("toplot.dat", "w");

    ItoProcess proc = ItoProcess(a_term, b_term);

    proc.WeinerValue(3.);
    proc.WeinerValue(1.);
    proc.WeinerValue(2.);
    int x;
    /*double x0 = 10;
    double tmax = 100;
    double dt = 1;
    auto res1 = proc.SampleWagnerPlaten(x0, tmax, dt);
    auto res2 = proc.SampleMilstein(x0, tmax, dt);
    auto res3 = proc.SampleMilstein(x0, tmax, dt);
    proc.WeinerResample();
    auto res4 = proc.SampleMilstein(x0, tmax, dt);
    auto res5 = proc.SampleMilstein(x0, tmax, dt);

    for(int i=0;i<100;i++)
    {
        fprintf(out,"%lf\n",proc.WeinerValue(i));
    }

    fclose(out);*/
}
