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
    double x0 = 1.5;
    double tmax = 100;

    auto stepSize = 0.1;

    std::vector <ItoProcess::IntegrationOptions> optss = {
        //{IntegratorType::EulerMaruyama,  IntegrationStyle::Fixed, .stepSize=stepSize/100},
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
            if(inputParser.cmdOptionExists("--full_paths"))
                for(auto const& datapoint: path)
                    printf("%lf %lf ", datapoint.time, datapoint.value);
            else if(inputParser.cmdOptionExists("--resample"))
            {
                double dt;
                sscanf(inputParser.getCmdOption("--resample").c_str(), "%lf", &dt);
                for(double t = 0; t <= tmax; t+=dt)
                    printf("%lf ", proc.GetItermediateValue(t));
            }
            else
                printf("%lf", path.back().value);
            printf("\n");
        }
        printf("\n");
    }
}
