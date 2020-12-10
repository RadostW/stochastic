#include "equations.cpp"
#include "itoprocess.cpp"
#include <iostream>

void test(ExactStochasticDifferentialEquation &eq)
{
    ItoProcess proc = ItoProcess(eq.drift, eq.volatility);
    
    auto opts = proc.GetIntegrationOptions();
    opts.integratorType = IntegratorType::Milstein;
    opts.integrationStyle = IntegrationStyle::Adaptive;
    proc.SetIntegrationOptions(opts);

    auto res = proc.SamplePath(0.1, 10);
    printf("Len = %d\n",(int)res.size());
}

int main()
{
    SinhEquation q;
    test(q);    
}
