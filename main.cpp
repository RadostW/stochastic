// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.

#include <vector>
#include <stdio.h>
#include "itoprocess.cpp"

// dh = a dt + b dW

tdouble mu(tdouble h) // make sure mu->1 as h->inf
{
    return 1.0-(1.0/h);
}
tdouble dmu(tdouble h)
{
    return 1.0/(h*h);
}

double hmean=3;
tdouble a_term(tdouble x)
{
    return 0.5*dmu(x) + 0.5*mu(x)*(1.0/hmean);
}
tdouble b_term(tdouble x)
{
    return sqrt(mu(x));
}

// stochastic equation to be solved:
// dX = a_term(x) dt + b_term(x) dW

int main()
{
    auto eq = SinhEquation();
    ItoProcess proc = ItoProcess(eq);
    double x0 = 0;
    double tmax = 400;

    auto opts = proc.GetIntegrationOptions();
    opts.stepSize = tmax/1000;
    opts.targetMseDensity = 5e-7;

    double errEuler = 0;
    double errMilstein = 0;

    int n_proc = 100;
    for(int i=0;i<n_proc;i++)
    {
        proc.ResetRealization(i);

        // Fixed
        opts.integrationStyle = IntegrationStyle::Fixed;
        opts.integratorType = IntegratorType::EulerMaruyama;

        proc.SetIntegrationOptions(opts);
        double valEuler = proc.SamplePath(x0, tmax).back().value;

        opts.integratorType = IntegratorType::Milstein;
        proc.SetIntegrationOptions(opts);
        double valMilstein = proc.SamplePath(x0, tmax).back().value;
    }
}
