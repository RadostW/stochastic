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

    const int nproc = 1000;
    double x0 = 0.;
    double tmax = 400;
    double dt = 0.5;

    double errWagnerPlaten=0;
    double errMilstein=0;
    double errEuler=0;
    double errAdaptiveMilstein=0;

    double stepsAdaptiveMilstein=0;

    for(int i=0;i<nproc;i++)
    {
        double valExact        = asinh(a*proc.GetWienerValue(tmax)+sinh(x0));
        double valWagnerPlaten = *proc.SampleWagnerPlaten(x0, tmax, dt).rbegin();
        double valMilstein     = *proc.SampleMilstein(x0, tmax, dt).rbegin();
        double valEuler        = *proc.SampleEuler(x0, tmax, dt).rbegin();
        auto samAdaptiveMilstein = proc.SampleAdaptiveMilstein(x0, 0, tmax, (1./2.41)*dt);
        double valAdaptiveMilstein= *samAdaptiveMilstein.rbegin();
        stepsAdaptiveMilstein += samAdaptiveMilstein.size()-1;
        
        if(i%10==0) printf("%4d %5.2lf    %5.2lf %5.2lf %5.2lf %5.2lf\n",i,
                                valExact,
                            valWagnerPlaten-valExact,valMilstein-valExact,valEuler-valExact,valAdaptiveMilstein-valExact);

        errWagnerPlaten += (valWagnerPlaten-valExact)*(valWagnerPlaten-valExact);
        errMilstein += (valMilstein-valExact)*(valMilstein-valExact);
        errEuler += (valEuler-valExact)*(valEuler-valExact);
        errAdaptiveMilstein += (valAdaptiveMilstein-valExact)*(valAdaptiveMilstein-valExact);

        proc.ResetRealization();
    }

    errWagnerPlaten = sqrt(errWagnerPlaten/nproc);
    errMilstein     = sqrt(errMilstein/nproc);
    errEuler        = sqrt(errEuler/nproc);
    errAdaptiveMilstein= sqrt(errAdaptiveMilstein/nproc);

    stepsAdaptiveMilstein = tmax / (stepsAdaptiveMilstein / nproc);

    printf("%20s %s (dt=%lf)\n","Alg","RMSE",dt);
    printf("%20s %lf\n","WagnerPlaten",errWagnerPlaten);
    printf("%20s %lf\n","Milstein",errMilstein);
    printf("%20s %lf\n","Euler",errEuler);
    printf("%20s %lf  (average dt=%lf)\n","AdaptiveMilstein",errAdaptiveMilstein,stepsAdaptiveMilstein);

    return 0;
}
