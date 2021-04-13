// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.

#include <vector>
#include <stdio.h>
#include "itoprocess.cpp"
#include "argparser.cpp"

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

int main(int argc, char* argv[])
{
    auto inputParser =  InputParser(argc, argv);
    int seed = 0;
    if(inputParser.cmdOptionExists("--seed"))
    {
        sscanf(inputParser.getCmdOption("--seed").c_str(), "%d", &seed);
        srand(seed);
        seed = rand();
    }
    auto eq = WallEquation();
    ItoProcess proc = ItoProcess(eq);
    double x0 = 3;
    double tmax = 30;

    auto opts = proc.GetIntegrationOptions();
    opts.stepSize = 0.01;
    auto stepSize = 0.01;

    std::vector <ItoProcess::IntegrationOptions> optss = {
        {IntegratorType::EulerMaruyama,  IntegrationStyle::Fixed, .stepSize=stepSize/100},
        {IntegratorType::EulerMaruyama,  IntegrationStyle::Fixed, .stepSize=stepSize},
        {IntegratorType::Milstein,  IntegrationStyle::Fixed, .stepSize=stepSize},
        {IntegratorType::EulerMaruyama,  IntegrationStyle::Adaptive, .stepSize=stepSize},
        {IntegratorType::Milstein,  IntegrationStyle::Adaptive, .stepSize=stepSize},
        {IntegratorType::EulerMirror1,  IntegrationStyle::Fixed, .stepSize=stepSize},
    }; 

    int n_proc = 20;
    for(int i=0;i<n_proc;i++)
    {
        proc.ResetRealization(seed + i);
        for(auto const& opts: optss)
        {
            proc.SetIntegrationOptions(opts);
            auto path = proc.SamplePath(x0, tmax);
            for(auto const& datapoint: path)
                printf("%lf %lf ", datapoint.time, datapoint.value);
            printf("\n");   
        }
        printf("\n");
    }    
}
