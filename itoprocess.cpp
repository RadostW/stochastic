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

    std::vector<PathPoint> SampleEulerAdaptive(double x0, double tmax)
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
    
    /*double GetDt(){
        if(IntegrationOptions_.integrationStyle == Fixed)
            return IntegrationOptions_.stepSize;
        
        if(IntegrationOptions_.integrationStyle == Adaptive){
            dt = find_root_bin_search(
                [this](double dt){return sqrt(LocalMSEEstimate(dt))/dt - target_error_density;},
                IntegrationOptions_.stepSize/10.,
                IntegrationOptions_.stepSize*10.,
                dt
            );
            return dt;
        }
        throw std::logic_error("AdaptivePredictive not implemented!");
    }

    double LocalMSEEstimate(double dt)
    {
        if(IntegrationOptions_.integrationStyle == Adaptive)
        {
            double mse = 0;
            if(error_order >= 2 & IntegrationOptions_.integratorType == EulerMaruyama)
                mse += coef11*coef11*dt*dt;
            if(error_order >= 3 & IntegrationOptions_.integratorType != WagnerPlaten)
                mse += (coef01*coef01+coef10*coef10+coef111*coef111)*dt*dt*dt;
            if(error_order >= 4)
                mse += coef00*coef00*dt*dt*dt*dt;  // TODO: address the fact that coef_<1x0, 2x1> should also be here!
            
            return mse;
        }
        throw std::logic_error("AdaptivePredictive not implemented!");
    }

    double GetDx(){
        dx = 0;
        dw = W.GetValue(t+dt) - W.GetValue(t);
        dz = W.GetZ(t, t+dt);
        
        if(IntegrationOptions_.integratorType == EulerMaruyama)
            dx += coef0*dt + coef1*dw;
        if(IntegrationOptions_.integratorType == Milstein)
            dx += coef11*(dw*dw-dt);
        if(IntegrationOptions_.integratorType == WagnerPlaten)
            dx += coef01*dz + coef10*(dw*dt-dz) + coef111*((1. / 3.) * dw * dw - dt) * dw;
        
        return dx;        
    }

    std::vector<PathPoint> SamplePathGeneral(double x0, double tmax){
        double dx;        
        
        n_steps = 0;
        t = 0;
        x = tdouble::Variable(x0);        
        std::vector<PathPoint> res;

        while(t < tmax)
        {
            // Store most recent entry.
            res.push_back(PathPoint(t,x.GetValue()));// Push value BEFORE each step to have initial value in response vector

            // Evaluete coeficients
            faval = fa(x);
            fbval = fb(x);

            a   = faval.GetValue();
            ap  = faval.GetGradient();
            app = faval.GetHessian();
            b   = fbval.GetValue();
            bp  = fbval.GetGradient();
            bpp = fbval.GetHessian();

            coef0   = a;
            coef1   = b;
            coef00  = a*ap + b*b*app/2.;
            coef01  = b*ap;
            coef10  = a*bp + b*b*bpp/2.;
            coef11  = b*bp;
            coef111 = b*(b*bpp + bp*bp);

            // Determine dt
            dt = GetDt();
            dt = std::min(dt, tmax-t); // Don't go over tmax

            // Update x
            dx = GetDx();
            x = tdouble::Variable(x.GetValue() + dx);

            // Update t
            n_steps++;
            if(IntegrationOptions_.integrationStyle == Fixed)
                t = n_steps*IntegrationOptions_.stepSize;
            else
                t += dt;
        }

        // Push final value
        res.push_back(PathPoint(tmax,x.GetValue()));

        return res;
    }
    */

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

    /*
    std::vector<PathPoint> SamplePath(double x0, double tmax)
    {
        if(IntegrationOptions_.integratorType == EulerMaruyama && 
            IntegrationOptions_.integrationStyle == Fixed) return SampleEulerMaruyama(x0,tmax);
        else if(IntegrationOptions_.integratorType == Milstein && 
            IntegrationOptions_.integrationStyle == Fixed) return SampleMilstein(x0,tmax);
    }
    */
};
