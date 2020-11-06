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

// benchmark equation has exact solution
// X_t = sinh(W_t + arcshinh(X_0))

int main()
{
    ItoProcess proc = ItoProcess(a_term, b_term);

    const int nproc = 100;
    double x0 = 0.2;
    double tmax = 400;
    double dt = 0.5;

    double errWagnerPlaten=0;
    double errMilstein=0;
    double errEuler=0;
    double errAdaptiveEuler=0;

    double stepsAdaptiveEuler=0;

    for(int i=0;i<nproc;i++)
    {
        double valExact        = asinh(a*proc.GetWienerValue(tmax)+sinh(x0));
        double valWagnerPlaten = *proc.SampleWagnerPlaten(x0, tmax, dt).rbegin();
        double valMilstein     = *proc.SampleMilstein(x0, tmax, dt).rbegin();
        double valEuler        = *proc.SampleEuler(x0, tmax, dt).rbegin();
        auto samAdaptiveEuler = proc.SampleAdaptiveEuler(x0, 0, tmax, 10);
        double valAdaptiveEuler= *samAdaptiveEuler.rbegin();
        stepsAdaptiveEuler += samAdaptiveEuler.size();
        
        if(i%10==0) printf("%4d %5.2lf    %5.2lf %5.2lf %5.2lf %5.2lf\n",i,
                                valExact,
                            valWagnerPlaten-valExact,valMilstein-valExact,valEuler-valExact,valAdaptiveEuler-valExact);

        errWagnerPlaten += (valWagnerPlaten-valExact)*(valWagnerPlaten-valExact);
        errMilstein += (valMilstein-valExact)*(valMilstein-valExact);
        errEuler += (valEuler-valExact)*(valEuler-valExact);
        errAdaptiveEuler += (valAdaptiveEuler-valExact)*(valAdaptiveEuler-valExact);

        proc.ResetRealization();
    }

    errWagnerPlaten = sqrt(errWagnerPlaten/nproc);
    errMilstein     = sqrt(errMilstein/nproc);
    errEuler        = sqrt(errEuler/nproc);
    errAdaptiveEuler= sqrt(errAdaptiveEuler/nproc);

    stepsAdaptiveEuler = tmax / (stepsAdaptiveEuler / nproc);

    printf("%20s %s (dt=%lf)\n","Alg","RMSE",dt);
    printf("%20s %lf\n","WagnerPlaten",errWagnerPlaten);
    printf("%20s %lf\n","Milstein",errMilstein);
    printf("%20s %lf\n","Euler",errEuler);
    printf("%20s %lf  (average dt=%lf)\n","AdaptiveEuler",errAdaptiveEuler,stepsAdaptiveEuler);

    return 0;
}
