// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <map>
#include <time.h>
#include "tdouble.cpp"
#include "wiener.cpp"

// Provides abstraction of an Ito Process,
// See https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes for details
// Allows for integration and sampling
class ItoProcess
{
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

    // Destroys underlying process and creates a new one.
    // If you want another sample from the same process use this.
    void ResetRealization()
    {
        //TODO: this is ugly now, RW 2020-10-28
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        W = Wiener(ts.tv_nsec);
    }

    std::vector<double> SampleAdaptiveMilstein(double x0, double tmin, double tmax, double step)
    {

        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        double t=0;

        for (int i = 0; t < tmax; i++)
        {
            res.push_back(x.GetValue()); // Push value BEFORE each step to have initial value in response vector
            double sbegin = t;
            double x0 = x.GetValue();
            double stepscaling = std::min(    fabs( (1.*pow(cosh(x0),5.))/(-2. + cosh(2.*x0))  )    , 4.);
            double send = std::min( sbegin + step * stepscaling  , tmax );
            double dt = send-sbegin;            
            double a = fa(x).GetValue();
            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient()[0];
            double dW = W.GetValue(send) - W.GetValue(sbegin);
            x = tdouble(x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt), 0);

            t = send;
        }
        res.push_back(x.GetValue()); //Push final value
        return res;
        
    }

    std::vector<double> SampleEuler(double x0, double tmax, double step)
    {
        return SampleEuler(x0,0,tmax,step);
    }
    std::vector<double> SampleEuler(double x0, double tmin, double tmax, double step)
    {
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; step * i + tmin < tmax; i++)
        {
            res.push_back(x.GetValue()); // Push value BEFORE each step to have initial value in response vector
            double sbegin = step*i+tmin;
            double send = std::min(step*(i+1)+tmin,tmax);
            double dt = send-sbegin;
            double a = fa(x).GetValue();
            double b = fb(x).GetValue();
            double dW = W.GetValue(send) - W.GetValue(sbegin);
            x = tdouble(x.GetValue() + a * dt + b * dW, 0);
        }
        res.push_back(x.GetValue()); //Push final value
        return res;
    }

    std::vector<double> SampleMilstein(double x0, double tmax, double step)
    {
        return SampleMilstein(x0,0,tmax,step);
    }
    std::vector<double> SampleMilstein(double x0, double tmin, double tmax, double step)
    {
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; tmin + step * i < tmax; i++)
        {
            res.push_back(x.GetValue()); // Push value BEFORE each step to have initial value in response vector
            double sbegin = tmin + step*i;
            double send = std::min(tmin + step*(i+1),tmax);
            double dt = send-sbegin;            
            double a = fa(x).GetValue();
            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient()[0];
            double dW = W.GetValue(send) - W.GetValue(sbegin);
            x = tdouble(x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt), 0);
        }
        res.push_back(x.GetValue()); //Push final value
        return res;
    }

    std::vector<double> SampleWagnerPlaten(double x0, double tmax, double step)
    {
        return SampleWagnerPlaten(x0, 0, tmax, step);
    }
    std::vector<double> SampleWagnerPlaten(double x0, double tmin, double tmax, double step)
    {
        tdouble x = tdouble(x0, 0);
        std::vector<double> res;

        for (int i = 0; tmin + step * i < tmax; i++)
        {
            res.push_back(x.GetValue()); // Push value BEFORE each step to have initial value in response vector
            double sbegin = tmin + step*i;
            double send = std::min(tmin + step*(i+1),tmax);
            double dt = send-sbegin;
            tdouble faval = fa(x);
            double a = faval.GetValue();
            double ap = faval.GetGradient()[0];
            double app = faval.GetHessian()[0][0];

            tdouble fbval = fb(x);
            double b = fbval.GetValue();
            double bp = fbval.GetGradient()[0];
            double bpp = fbval.GetHessian()[0][0];

            double dW = W.GetValue(send) - W.GetValue(sbegin);
            double dZ = W.GetZ(sbegin,send);

            x = tdouble(
                x.GetValue() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt) +
                    b * ap * dZ + 0.5 * (a * ap + 0.5 * b * b * app) * dt * dt +
                    (a * bp + 0.5 * b * b * bpp) * (dW * dt - dZ) +
                    0.5 * b * (b * bpp + bp * bp) * ((1. / 3.) * dW * dW - dt) * dW,
                0);
        }
        res.push_back(x.GetValue()); //Push final value
        return res;
    }

    double GetWienerValue(double time)
    {
        return W.GetValue(time);
    }

 private:
    // Function pointers defining SDE:
    // dX = fa(X) dt + fb(X) dW
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);

    // Underlying random process.
    Wiener W;


    // Handy things
    double sech(double x)
    {
        return 1/cosh(x);
    }
    
    template<typename T>
    T last(std::vector<T> A)
    {
        return *A.rbegin();
    }
};

