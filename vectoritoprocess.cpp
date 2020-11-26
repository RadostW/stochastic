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
#include "matrix/matrix.cpp"

// Provides abstraction of an vector Ito Process,
// See https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes for details
// Allows for integration and sampling

class ItoProcess
{
 public:
    // Constructs Ito process out of drift and volitality functions.
    // Consistent with definition:
    // dX_j = drift(X_i)_j dt + volitality(X_i)_jk dW_k
    // If X \in R^n then drift is 1xn and volitality is nxn and dimension is n
    ItoProcess(matrix<tdouble> drift(matrix<tdouble>), matrix<tdouble> volitality(matrix<tdouble>),unsigned int dimension)
    {
        fa = drift;
        fb = volitality;
        W = std::vector<Wiener>(dimension);

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        auto nsec = ts.tv_nsec;

        for(auto it=W.begin();it!=W.end();it++)
        {
            *it = Wiener(nsec++);
        }
    }

    // Destroys underlying process and creates a new one.
    // If you want another sample from the same process use this.
    void ResetRealization()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        auto nsec = ts.tv_nsec;

        for(auto it=W.begin();it!=W.end();it++)
        {
            *it = Wiener(nsec++);
        }
    }

    std::vector<matrix<double>> SampleEuler(matrix<double> x0, double tmax, double step)
    {
        return SampleEuler(x0,0,tmax,step);
    }
    std::vector<matrix<double>> SampleEuler(matrix<double> x0, double tmin, double tmax, double step)
    {
        assert(x0.columns()==1);
        matrix<tdouble> x = Fromdouble(x0);

        std::vector<matrix<double>> res;

        for (int i = 0; step * i + tmin < tmax; i++)
        {
            res.push_back(Fromtdouble(x)); // Push value BEFORE each step to have initial value in response vector
            double sbegin = step*i+tmin;
            double send = std::min(step*(i+1)+tmin,tmax);
            double dt = send-sbegin;
            auto a = Fromtdouble(fa(x));
            auto b = Fromtdouble(fb(x));
            auto dW = GetWienerValue(send) - GetWienerValue(sbegin);
            x = Fromdouble(Fromtdouble(x) + a * dt + b * dW);
        }
        res.push_back(Fromtdouble(x)); //Push final value
                
        return res;
    }

    // Get value of i^th Wiener process at time t
    double GetWienerValue(int i,double t)
    {
        Wiener tmp =  W[i];
        return tmp.GetValue(t);
    }
    matrix<double> GetWienerValue(double t)
    {
        matrix<double>ret(W.size(),1);
        for(int i=0;i<W.size();i++)
        {
            ret(i,0) = GetWienerValue(i,t);
        }
        return ret;
    }

 private:
    // Function pointers defining SDE:
    // dX = fa(X) dt + fb(X) dW
    matrix<tdouble> (*fa)(matrix<tdouble>);
    matrix<tdouble> (*fb)(matrix<tdouble>);

    // Underlying random process.
    std::vector<Wiener> W;

    matrix<double> Fromtdouble(matrix<tdouble> x) const
    {
        matrix<double> ret(x.rows(),x.columns());
        for(size_t i=0;i<x.rows();i++)for(size_t j=0;j<x.columns();j++)
        {
            ret(i,j) = x(i,j).GetValue();
        }
        return ret;
    }
    matrix<tdouble> Fromdouble(matrix<double> x) const
    {
        matrix<tdouble> ret(x.rows(),x.columns());
        for(size_t i=0;i<x.rows();i++)for(size_t j=0;j<x.columns();j++)
        {
            ret(i,j) = tdouble(x(i,j),i*x.columns()+j);
        }
        return ret;
    }

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

