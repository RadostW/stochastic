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

double exact_solution(double x0, double w)
{
    return asinh(a*w + sinh(x0));
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
    double errEulerAdPr = 0;
    double errEulerAd1 = 0;
    double errEulerAd2 = 0;
    double errEulerAd3 = 0;
    double errMilstein = 0;
    double errMilsteinAd1 = 0;
    double errMilsteinAd2 = 0;
    
    double stepsEulerAdPr = 0;
    double stepsEulerAd1 = 0;
    double stepsEulerAd2 = 0;
    double stepsEulerAd3 = 0;
    double stepsMilsteinAd1 = 0;
    double stepsMilsteinAd2 = 0;
    
    double meanDtEulerAdPr = 0;
    double meanDtEulerAd1 = 0;
    double meanDtEulerAd2 = 0;
    double meanDtEulerAd3 = 0;
    double meanDtMilsteinAd1 = 0;
    double meanDtMilsteinAd2 = 0;
    
    int n_proc = 100;
    for(int i=0;i<n_proc;i++)
    {
        proc.ResetRealization(i);
        double valExact    = exact_solution(x0, proc.getW(tmax));

        // Fixed
        opts.integrationStyle = IntegrationStyle::Fixed;

        opts.integratorType = IntegratorType::EulerMaruyama;
        proc.SetIntegrationOptions(opts);
        double valEuler = proc.SamplePath(x0, tmax).back().value;

        opts.integratorType = IntegratorType::Milstein;
        proc.SetIntegrationOptions(opts);
        double valMilstein = proc.SamplePath(x0, tmax).back().value;

        // Adaptive predictive
        opts.integrationStyle = IntegrationStyle::AdaptivePredictive;        
        opts.integratorType = IntegratorType::EulerMaruyama;
        proc.SetIntegrationOptions(opts);
        auto traj = proc.SamplePath(x0, tmax);
        double valEulerAdPr = traj.back().value;
        meanDtEulerAdPr += tmax/(traj.size() - 1);


        // Adaptive
        opts.integrationStyle = IntegrationStyle::Adaptive;
        
        opts.integratorType = IntegratorType::EulerMaruyama;

        opts.errorTerms = 1;
        proc.SetIntegrationOptions(opts);
        traj = proc.SamplePath(x0, tmax);
        double valEulerAd1 = traj.back().value;
        meanDtEulerAd1 += tmax/(traj.size() - 1);

        opts.errorTerms = 2;
        proc.SetIntegrationOptions(opts);
        traj = proc.SamplePath(x0, tmax);
        double valEulerAd2 = traj.back().value;
        meanDtEulerAd2 += tmax/(traj.size() - 1);

        opts.errorTerms = 3;
        proc.SetIntegrationOptions(opts);
        traj = proc.SamplePath(x0, tmax);
        double valEulerAd3 = traj.back().value;
        meanDtEulerAd3 += tmax/(traj.size() - 1);

        opts.integratorType = IntegratorType::Milstein;

        opts.errorTerms = 1;
        proc.SetIntegrationOptions(opts);
        traj = proc.SamplePath(x0, tmax);
        double valMilsteinAd1 = traj.back().value;
        meanDtMilsteinAd1 += tmax/(traj.size() - 1);

        opts.errorTerms = 2;
        proc.SetIntegrationOptions(opts);
        traj = proc.SamplePath(x0, tmax);
        double valMilsteinAd2 = traj.back().value;
        meanDtMilsteinAd2 += tmax/(traj.size() - 1);
        
        errEuler    += (valEuler-valExact)*(valEuler-valExact);
        errEulerAdPr  += (valEulerAdPr-valExact)*(valEulerAdPr-valExact);
        errEulerAd1  += (valEulerAd1-valExact)*(valEulerAd1-valExact);
        errEulerAd2  += (valEulerAd2-valExact)*(valEulerAd2-valExact);
        errEulerAd3  += (valEulerAd3-valExact)*(valEulerAd3-valExact);
        errMilstein += (valMilstein-valExact)*(valMilstein-valExact);
        errMilsteinAd1 += (valMilsteinAd1-valExact)*(valMilsteinAd1-valExact);
        errMilsteinAd2 += (valMilsteinAd2-valExact)*(valMilsteinAd2-valExact);
    }
    
    errEuler = sqrt(errEuler/n_proc);
    errEulerAdPr = sqrt(errEulerAdPr/n_proc);
    errEulerAd1 = sqrt(errEulerAd1/n_proc);
    errEulerAd2 = sqrt(errEulerAd2/n_proc);
    errEulerAd3 = sqrt(errEulerAd3/n_proc);
    errMilstein = sqrt(errMilstein/n_proc);
    errMilsteinAd1 = sqrt(errMilsteinAd1/n_proc);
    errMilsteinAd2 = sqrt(errMilsteinAd2/n_proc);

    meanDtEulerAdPr /= n_proc;
    meanDtEulerAd1 /= n_proc;
    meanDtEulerAd2 /= n_proc;
    meanDtEulerAd3 /= n_proc;
    meanDtMilsteinAd1 /= n_proc;
    meanDtMilsteinAd2 /= n_proc;
    
    printf("errEuler       %lf,      dt = %lf\n", errEuler, opts.stepSize);
    printf("errEulerAdPr   %lf, mean dt = %lf\n", errEulerAdPr, meanDtEulerAdPr);
    printf("errEulerAd1    %lf, mean dt = %lf\n", errEulerAd1, meanDtEulerAd1);
    printf("errEulerAd2    %lf, mean dt = %lf\n", errEulerAd2, meanDtEulerAd2);
    printf("errEulerAd3    %lf, mean dt = %lf\n", errEulerAd3, meanDtEulerAd3);
    printf("errMilstein    %lf,      dt = %lf\n", errMilstein, opts.stepSize);
    printf("errMilsteinAd1 %lf, mean dt = %lf\n", errMilsteinAd1, meanDtMilsteinAd1);
    printf("errMilsteinAd2 %lf, mean dt = %lf\n", errMilsteinAd2, meanDtMilsteinAd2);    
}
