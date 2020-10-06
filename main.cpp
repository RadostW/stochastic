#include<vector>
#include<stdio.h>
#include "pmath.cpp"
#include "tdouble.cpp"
#include "itoprocess.cpp"



tdouble mobility(tdouble distance)
{
    tdouble x = 1.0 / distance;
    return 0.986292 - x - 0.00688*cos(10.86762 + 8.092*x) + 0.02057*sin(2.506 + x*(3.074 + 2.227*x));
}
tdouble dmobility(tdouble distance)
{
    tdouble h = distance;
    return ((-0.09161878 - 0.06323218*h)*
      cos(2.506 + (2.227 + 3.074*h)/(h*h)) + 
     h*(1. - 0.05567296*sin(10.86762 + 8.092/h)))/(h*h*h);
}

tdouble a_term(tdouble x)
{
    return dmobility(x);
}
tdouble b_term(tdouble x)
{
    return sqrt(2*mobility(x));
}
// stochastic equation to be solved:
// dX = a_term(x) dt + b_term(x) dW



int main()
{

    FILE* out;
    out = fopen("toplot.dat","w");

    ItoProcess proc = ItoProcess(a_term,b_term);
    auto res1 =  proc.SampleEuler(5,50,0.1);
    for (vector<double>::iterator i = res1.begin(); i != res1.end(); i++)
    {
        std::cout << *i << ' ';
        fprintf(out,"%lf\n",*i);
    }

    cout<<"\n\n";
    auto res2 =  proc.SampleMilstein(5,50,0.1);
    for (vector<double>::iterator i = res2.begin(); i != res2.end(); i++)
    {
        std::cout << *i << ' ';
    }


}
