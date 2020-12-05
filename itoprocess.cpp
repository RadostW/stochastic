// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <map>
#include <time.h>
#include <functional>
#include "tdouble.cpp"
#include "wiener.cpp"
#include "utils.cpp"

// Provides abstraction of an Ito Process,
// See https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes for details
// Allows for integration and sampling

enum IntegratorType { EulerMaruyama, Milstein, WagnerPlaten };
enum IntegrationStyle { Fixed, Adaptive, AdaptivePredictive };

class ItoProcess
{
  public:
    struct IntegrationOptions
    {
        IntegratorType integratorType = Milstein;
        IntegrationStyle integrationStyle = Fixed;
        double stepSize = 0.01;
        int errorTerms = 1;
        double targetErrorDensity = 1e-2;
    };
    struct PathPoint
    {
        double time;
        double value;
        PathPoint(double time_,double value_)
        {
            time = time_;
            value = value_;
        }
    };

  private:
    // Function pointers defining SDE:
    // dX = fa(X) dt + fb(X) dW
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);

    // Underlying random process.
    Wiener W;

    // Stores integration settings
    IntegrationOptions IntegrationOptions_;

    std::vector<PathPoint> SampleEulerMaruyama(double x0, double tmax)
    {
        tdouble x = tdouble::Variable(x0);
        std::vector<PathPoint> res;
        double step = IntegrationOptions_.stepSize;
        double tmin = 0;

        for (int i = 0; step * i + tmin < tmax; i++)
        {
            res.push_back(PathPoint(step*i,x.GetValue()));// Push value BEFORE each step to have initial value in response vector
            double sbegin = step*i+tmin;
            double send = std::min(step*(i+1)+tmin,tmax);
            double dt = send-sbegin;
            double a = fa(x).GetValue();
            double b = fb(x).GetValue();
            double dW = W.GetValue(send) - W.GetValue(sbegin);
            x = tdouble::Variable(x.GetValue() + a * dt + b * dW);
        }
        res.push_back(PathPoint(tmax,x.GetValue())); //Push final value
        return res;
    }

    std::vector<PathPoint> SampleEulerMaruyamaAdaptive(double x0, double tmax)
    {
        tdouble x = tdouble::Variable(x0);
        std::vector<PathPoint> res;
        double dt = IntegrationOptions_.stepSize;
        double tmin = 0;
        double t = tmin;

        if(IntegrationOptions_.errorTerms > 3)
            throw std::logic_error("SampleEulerMaruyamaAdaptive can't handle that much error terms");

        while(t < tmax)
        {
            res.push_back(PathPoint(t, x.GetValue()));// Push value BEFORE each step to have initial value in response vector
            
            tdouble faval = fa(x);
            tdouble fbval = fb(x); 
            
            double a   = faval.GetValue();
            double ap  = faval.GetGradient();
            double app = faval.GetHessian();
            double b   = fbval.GetValue();
            double bp  = fbval.GetGradient();
            double bpp = fbval.GetHessian();
            
            // E R^2/dt^2 = mse_density_polynomial_coefs[0] + mse_density_polynomial_coefs[1]*dt + ...
            double mse_density_polynomial_coefs[] = {
                pow(b*bp, 2),
                pow(b*ap, 2) + pow(a*bp + b*b*bpp/2., 2) + pow(b*(b*bpp + bp*bp), 2),
                pow(a*ap + b*b*app/2., 2) + pow(b*(b*app + ap*bp), 2) // TODO: I_101 and I_110 should be here but require third derivative
            };
            
            // Solving sqrt(E R^2/dt^2) = target_error_density
            mse_density_polynomial_coefs[0] -= pow(IntegrationOptions_.targetErrorDensity, 2);
            dt = solve_increasing_poly(
                mse_density_polynomial_coefs, 
                IntegrationOptions_.errorTerms,
                IntegrationOptions_.stepSize/10,
                IntegrationOptions_.stepSize*10,
                dt,
                IntegrationOptions_.targetErrorDensity/100
            );
            dt = std::min(dt, tmax-t);
            double dW = W.GetValue(t+dt) - W.GetValue(t);
            x = tdouble::Variable(x.GetValue() + a * dt + b * dW);
            t += dt;
        }
        res.push_back(PathPoint(tmax,x.GetValue())); //Push final value
        return res;
    }

    std::vector<PathPoint> SampleMilstein(double x0, double tmax)
    {
        tdouble x = tdouble::Variable(x0);
        std::vector<PathPoint> res;
        double step = IntegrationOptions_.stepSize;
        double tmin = 0;


        for (int i = 0; tmin + step * i < tmax; i++)
        {
            res.push_back(PathPoint(step*i,x.GetValue()));// Push value BEFORE each step to have initial value in response vector
            double sbegin = tmin + step*i;
            double send = std::min(tmin + step*(i+1),tmax);
            double dt = send-sbegin;            
            double a = fa(x).GetValue();
            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient();
            double dW = W.GetValue(send) - W.GetValue(sbegin);
            x = tdouble::Variable(x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt));
        }
        res.push_back(PathPoint(tmax,x.GetValue())); //Push final value
        return res;
    }    

    std::vector<PathPoint> SampleMilsteinAdaptive(double x0, double tmax)
    {
        tdouble x = tdouble::Variable(x0);
        std::vector<PathPoint> res;
        double dt = IntegrationOptions_.stepSize;
        double tmin = 0;
        double t = tmin;

        if(IntegrationOptions_.errorTerms > 2)
            throw std::logic_error("SampleMilsteinAdaptive can't handle that much error terms");

        while(t < tmax)
        {
            res.push_back(PathPoint(t, x.GetValue()));// Push value BEFORE each step to have initial value in response vector
            
            tdouble faval = fa(x);
            tdouble fbval = fb(x); 
            
            double a   = faval.GetValue();
            double ap  = faval.GetGradient();
            double app = faval.GetHessian();
            double b   = fbval.GetValue();
            double bp  = fbval.GetGradient();
            double bpp = fbval.GetHessian();
            
            // E R^2/dt^2 = mse_density_polynomial_coefs[0] + mse_density_polynomial_coefs[1]*dt + ...
            double mse_density_polynomial_coefs[] = {
                0,
                pow(b*ap, 2) + pow(a*bp + b*b*bpp/2., 2) + pow(b*(b*bpp + bp*bp), 2),
                pow(a*ap + b*b*app/2., 2) + pow(b*(b*app + ap*bp), 2) // TODO: I_101 and I_110 should be here but require third derivative
            };
            
            // Solving sqrt(E R^2/dt^2) = target_error_density
            mse_density_polynomial_coefs[0] -= pow(IntegrationOptions_.targetErrorDensity, 2);
            dt = solve_increasing_poly(
                mse_density_polynomial_coefs, 
                IntegrationOptions_.errorTerms + 1,
                IntegrationOptions_.stepSize/10,
                IntegrationOptions_.stepSize*10,
                dt,                
                IntegrationOptions_.targetErrorDensity/100
            );
            dt = std::min(dt, tmax-t);
            double dW = W.GetValue(t+dt) - W.GetValue(t);
            x = tdouble::Variable(x.GetValue() + a * dt + b * dW + 0.5*b*bp*(dW*dW-dt));
            t += dt;
        }
        res.push_back(PathPoint(tmax,x.GetValue())); //Push final value
        return res;
    }

  public:
    // Constructs Ito process out of two tdouble->tdouble functions.
    // Consistent with definition:
    // dX = drift(X) dt + volitality(X) dW
    ItoProcess(tdouble drift(tdouble), tdouble volitality(tdouble))
    {
        fa = drift;
        fb = volitality;
        W = Wiener();
    }
    ItoProcess(tdouble drift(tdouble), tdouble volitality(tdouble),IntegrationOptions integrationOptions)
    {
        fa = drift;
        fb = volitality;
        W = Wiener();
        IntegrationOptions_ = integrationOptions;
    }

    IntegrationOptions GetIntegrationOptions()
    {
        return IntegrationOptions_;
    }
    void SetIntegrationOptions(IntegrationOptions integrationOptions)
    {
        IntegrationOptions_ = integrationOptions;
    }

    // Destroys underlying process and creates a new one.
    // If you want another sample from the same process use this.
    void ResetRealization()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        W = Wiener(ts.tv_nsec);
    }
    
    void ResetRealization(int seed)
    {
        W = Wiener(seed);
    }

    double getW(double t)
    {
        return W.GetValue(t);
    }

    std::vector<PathPoint> SamplePath(double x0, double tmax)
    {
        if(IntegrationOptions_.integratorType == EulerMaruyama && 
            IntegrationOptions_.integrationStyle == Fixed) return SampleEulerMaruyama(x0,tmax);
        else if(IntegrationOptions_.integratorType == EulerMaruyama && 
            IntegrationOptions_.integrationStyle == Adaptive) return SampleEulerMaruyamaAdaptive(x0,tmax);
        else if(IntegrationOptions_.integratorType == Milstein && 
            IntegrationOptions_.integrationStyle == Fixed) return SampleMilstein(x0,tmax);
        else if(IntegrationOptions_.integratorType == Milstein && 
            IntegrationOptions_.integrationStyle == Adaptive) return SampleMilsteinAdaptive(x0,tmax);
    }
};
