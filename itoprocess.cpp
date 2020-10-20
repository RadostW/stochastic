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
        W = Wiener(time(0));
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
            double dW = W.GetValue((i+1)*dt) - W.GetValue(i*dt);
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
            double dW = W.GetValue((i+1)*dt) - W.GetValue(i*dt);
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

            double dW = W.GetValue((i+1)*dt) - W.GetValue(i*dt);
            double dZ = W.GetZ(i*dt,(i+1)*dt);
            
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
    // Function pointers defining SDE:
    // dX = fa(X) dt + fb(X) dW
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);

    // Underlying random process.
    Wiener W;
};

