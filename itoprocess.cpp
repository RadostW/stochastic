// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <map>
#include "tdouble.cpp"

// Provides abstraction of an Ito Process,
// See https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes for details
// Allows for integration and sampling
class ItoProcess
{
    // Used for obtaining trajectories from equation of type
    // dX = a(x) dt + b(x) dW
public:
    ItoProcess(tdouble nfa(tdouble), tdouble nfb(tdouble))
    {
        fa = nfa;
        fb = nfb;
        normal = std::normal_distribution<double>(0.0, 1.0);
        WienerPath[0.] = 0.;
        ZetReported[0.] = 0.;
    }
    double WeinerValue(double t)
    {

        if (t < 0)
        {
            return 0;
        }
        else if (WienerPath.count(t) == 1) // Repeated ask for the same value
        {
        }
        else if ( WienerPath.lower_bound(t) == WienerPath.end() ) // Beyond mesh on the right
        {
            ExtendMesh(t);
        }
        else // Between mesh points
        {
            RefineMesh(t);
        }

        if(WienerPath.count(t) != 1)
        {
             throw std::logic_error("WienerPath: Subsampling failure"); 
        }
        else
        {   
            return WienerPath[t];
        }
    }
    void WeinerResample()
    {
        WienerPath.clear();
        WienerPath[0.] = 0.;
        ZetReported.clear();
        ZetReported[0.] = 0;
    }
    double ZetValue(double a, double b)
    {
        // Return first irreducible double integral:
        // int_a^b int_a^q ds dW_q
        auto lower = ZetReported.lower_bound(a);
        auto upper = ZetReported.upper_bound(b);
        if (lower == ZetReported.end()) // Interval beyond mesh
        {
            ZetValue((ZetReported.rbegin())->first,a); // Sample interval on the left of requested
            return ZetValue(a,b);
        }
        else if (next(lower) == ZetReported.end() && upper == ZetReported.end()) // Interval at left edge of mesh
        {
            ExtendMesh(b);
            if(ZetReported.count(b) != 1)
            {
                throw std::logic_error("ZetReported: Subsampling failure"); 
            }
            else
            {
                return ZetReported[b];
            }   
        }
        else if (lower->first == a && upper->first == b && next(lower) == upper) // Re-report mesh interval
        {
            return upper->second;
        }
        else
        {
            if(lower->first!=a) // Missing meshpoint
            {
                RefineMesh(a);
            }
            if(upper->first!=b) // Missing meshpoint
            {
                RefineMesh(b);
            }

            lower = ZetReported.lower_bound(a);
            upper = ZetReported.upper_bound(b);
            if(lower->first != a || upper->first != b)
            {
                throw std::logic_error("ZetReported: Subsampling failure"); 
            }
            
            double accumulator = 0;
            double tm = b;
            double t1;
            double t0;
            for(auto it = lower;lower!=upper;lower++)
            {
                t0 = it->first;
                t1 = next(it)->first;
                accumulator += next(it)->second;
                accumulator += (WeinerValue(t1)-WeinerValue(t0))*(tm - t1);
            }
            return accumulator;
        }
    }

    std::vector<double> SampleEuler(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.GetValue());
            double a = fa(x).GetValue();
            double b = fb(x).GetValue();
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            x = tdouble(x.GetValue() + a * dt + b * dW, 0);
        }
        return res;
    }

    std::vector<double> SampleMilstein(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.GetValue());
            double a = fa(x).GetValue();
            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient()[0];
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            x = tdouble(x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt), 0);
        }
        return res;
    }

    std::vector<double> SampleWagnerPlaten(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.GetValue());
            tdouble faval = fa(x);
            double a = faval.GetValue();
            double ap = faval.GetGradient()[0];
            double app = faval.GetHessian()[0][0];

            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient()[0];
            double bpp = fbval.GetHessian()[0][0];

            double z1 = DrawNormal();
            double z2 = DrawNormal();
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            double dZ = ZetValue(i * dt, (i + 1) * dt);
            
            x = tdouble(
                x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt) +
                    b * ap * dZ + 0.5 * (a * ap + 0.5 * b * b * app) * dt * dt +
                    (a * bp + 0.5 * b * b * bpp) * (dW * dt - dZ) +
                    0.5 * b * (b * bpp + bp * bp) * ((1. / 3.) * dW * dW - dt) * dW,
                0);
        }
        return res;
    }

private:
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);
    std::default_random_engine generator;
    std::normal_distribution<double> normal;
    void RefineMesh(double s)
    {
        // TODO implement
        // Add new meshpoint at s
        if(WienerPath.count(s) != 0) throw std::logic_error("mesh refine wrong arg");
        auto lower = --WienerPath.lower_bound(s);
        auto upper = WienerPath.upper_bound(s);
        double dW = upper->second - lower->second;
        double Z = ZetReported[upper->first];

        double t0 = lower->first;
        double t1 = s;
        double t2 = upper->first;
        auto means = conditionalMean(t1-t0, t2-t0, dW, Z);
        auto varsAndcorr = conditionalVarsAndCov(t1-t0, t2-t0, dW, Z);

        double midW;
        double midZ;
        DrawCovaried(varsAndcorr[0], varsAndcorr[2], varsAndcorr[1], midW, midZ);
        midW += means[0];
        midZ += means[1];

        WienerPath[s] = lower->second + midW;
        ZetReported[s] = midZ;
        ZetReported[upper->first] = Z-midZ-midW*(upper->second -s);
    }
    void ExtendMesh(double t)
    {
        auto last = WienerPath.rbegin();
        if(t<=last->first) throw std::logic_error("extend meash wrong arg");
        double dt = t - last->first;
        double dW, newZ;
        DrawCovaried(dt, dt*dt/2, dt*dt*dt/3, dW, newZ);
        WienerPath[t] = dW + last->second;
        ZetReported[t] = newZ;
    }
    double DrawNormal()
    {
        return normal(generator);
    }
    void DrawCorrelated(double cor, double &x,double &y)
    {
        // Set x,y to correlated normal samples with Cor(x,y)=cor
        double z1 = DrawNormal();
        double z2 = DrawNormal();
        x = sqrt(1-cor*cor)*z1 + cor*z2;
        y = z2;
    }
    void DrawCovaried(double xx,double xy,double yy, double &x,double &y)
    {
        // Set x,y to sample of N(0,{{xx,xy},{xy,yy}}) dist
        DrawCorrelated(xy/sqrt(xx*yy),x,y);
        x = sqrt(xx) * x;
        y = sqrt(yy) * y;
    }
    std::map<double, double> WienerPath;
    std::map<double, double> ZetReported; // Values of reported I(0,1) integral at given intervals, saved on rhs.
    
    std::array<double, 2> conditionalMean(double s, double t, double w, double z)
    {
        // E(W0s, Z0s | W0t=w, Z0t=z)
        double mean_w0s = s*(w*(3*s*t-2*t*t)+z*6*(t-s))/(t*t*t);
        double mean_z0s = s*s*(w*(s*t-t*t)+z*(3*t-2*s))/(t*t*t);
        return std::array<double, 2> {mean_w0s, mean_z0s};
    }

    std::array<double, 3> conditionalVarsAndCov(double s, double t, double w, double z)
    {
        // Var(W0s), Var(Z0s) and Corr(W0s, Z0s) given W0t=w, Z0t=z
        double var_w0s = -s*(s-t)*(3*s*s-3*s*t+t*t)/(t*t*t);
        double var_z0s = -s*s*s*(s-t)*(s-t)*(s-t)/(3*t*t*t);
        double corr = sqrt(3.)*(t-2*s)/(2*sqrt(3*s*s-3*s*t+t*t));
        double cov = corr*sqrt(var_w0s*var_z0s);
        return std::array<double, 3> {var_w0s, var_z0s, cov};
    }
};

