#include<stdio.h>
#include<array>
#include<string>
#include<iostream>
#include<cmath>
#include<random>
#include<vector>




class ItoProcess
{
    // Used for obtaining trajectories from equation of type
    // dX = a(x) dt + b(x) dW
    public:
    ItoProcess( tdouble nfa(tdouble),tdouble nfb(tdouble) )
    {
        fa = nfa;
        fb = nfb;
        normal = normal_distribution<double>(0.0,1.0);
    }

    vector<double> SampleEuler( double x0, double tmax , double dt)
    {
        double t=0;
        tdouble x=tdouble(x0,0);
        vector<double> res;
        
        for(int i=0;dt*i < tmax;i++)
        {
            res.push_back(x.get_value());
            double a = fa(x).get_value();
            double b = fb(x).get_value();
            double dW = draw_normal()*sqrt(dt);
            x = tdouble(x.get_value() + a*dt + b*dW,0);
        }
        return res;       
    }

    vector<double> SampleMilstein( double x0, double tmax , double dt)
    {
        double t=0;
        tdouble x=tdouble(x0,0);
        vector<double> res;
        
        for(int i=0;dt*i < tmax;i++)
        {
            res.push_back(x.get_value());
            double a = fa(x).get_value();
            tdouble fbval = fb(x);
            double b = fbval.get_value();
            double bp = fbval.get_gradient()[0];
            double dW = draw_normal()*sqrt(dt);
            x = tdouble(x.get_value() + a*dt + b*dW + 0.5*b*bp*(dW*dW - dt),0);
        }
        return res;
    }

    vector<double> SampleWagnerPlaten( double x0, double tmax , double dt)
    {
        double t=0;
        tdouble x=tdouble(x0,0);
        vector<double> res;
        
        for(int i=0;dt*i < tmax;i++)
        {
            res.push_back(x.get_value());
            tdouble faval = fa(x);
            double a = faval.get_value();
            double ap = faval.get_gradient()[0];
            double app = faval.get_hessian()[0][0];

            tdouble fbval = fb(x);
            double b = fbval.get_value();
            double bp = fbval.get_gradient()[0];
            double bpp = fbval.get_hessian()[0][0];

            double z1 = draw_normal();
            double z2 = draw_normal();
            double dW = z1*sqrt(dt);
            double dZ = 0.5*(z1+z2/sqrt(3))*dt*sqrt(dt);


            x = tdouble(
                        x.get_value() + a*dt + b*dW + 0.5*b*bp*(dW*dW - dt) + 
                        b*ap*dZ + 0.5*(a*ap+0.5*b*b*app)*dt*dt + 
                        (a*bp+0.5*b*b*bpp)*(dW*dt-dZ) +
                        0.5*b*(b*bpp+bp*bp)*((1./3.)*dW*dW-dt)*dW
                        ,0);
        }
        return res;
    }

    private:
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);
    default_random_engine generator;
    normal_distribution<double> normal;
    double draw_normal()
    {   
        return normal(generator);
    }
};
