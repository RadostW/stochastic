// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include<map>
#include<random>


// Provides abstraction a realization of Wiener process,
// See https://en.wikipedia.org/wiki/Wiener_process for details.
// Allows for persistent, consistent sampling and subsampling
// Implements point value and double Stochastic integral int_a^b int_a^s dWds
class Wiener
{
 public:
    Wiener()
    {
        samplePoints[0.].w = 0;
        samplePoints[0.].zToPrevPoint = 0;
    }

    // Returns Wiener value at time t
    double GetValue(double t)
    {
        EnsureSamplePoint(t);
        return samplePoints[t].w;
    }

    // Returns double Stochastic integral Z between points t1<t2
    // See "Numerical Solution of Stochastic Differential Equations" for reference
    double GetZ(double t1, double t2)
    {
        EnsureSamplePoint(t1);
        EnsureSamplePoint(t2);

        auto it = samplePoints.lower_bound(t1);
        double Z = 0, Wt1 = it->second.w, dW, t, dt;
        while(it->first != t2)
        {
            t = it->first;
            dW = it->second.w - Wt1;
            it++;
            dt = it->first - t;
            Z += it->second.zToPrevPoint + dt*dW;
        }
        return Z;
    }

 private:

    // A wrapper
    struct samplePoint
    {
        double w;
        double zToPrevPoint;
    };

    // Stores information about already reported samples.
    std::map<double, samplePoint> samplePoints;

    // Takes care of the underlying mesh.
    // Subsamples the process if necessary.
    void EnsureSamplePoint(double t)
    {
        if (t < 0) return;

        auto lower = samplePoints.lower_bound(t);
        if (lower->first == t) return;  // Exact match.
        if (lower == samplePoints.end())  // Beyond known values.
        {
            samplePoint newSamplePoint;
            double tMax = samplePoints.rbegin()->first;
            double lastW = samplePoints.rbegin()->second.w;
            DrawIndependentWZ(tMax, t, lastW, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            samplePoints[t] = newSamplePoint;
        }
        else  // Between known values
        {
            double nextT = lower->first;
            samplePoint & nextSamplePoint = lower->second;
            lower--;
            double prevT = lower->first;
            samplePoint prevSamplePoint = lower->second;

            samplePoint newSamplePoint;
            DrawDependentWZ(prevT, t, nextT, prevSamplePoint.w, nextSamplePoint.w, nextSamplePoint.zToPrevPoint, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            nextSamplePoint.zToPrevPoint = nextSamplePoint.zToPrevPoint - newSamplePoint.zToPrevPoint - (nextT-t)*(newSamplePoint.w-prevSamplePoint.w);
            samplePoints[t] = newSamplePoint;
        }
    }

    // Draws a new sample point from an unconditioned distribution.
    void DrawIndependentWZ(double t1, double t2, double wt1,
                           double &wt2, double &zt1t2)
    {
        // assigns to wt2, zt1t2 sample from W(t2), Z(t1, t2) | W(t1); t1<t2
        double dt = t2 - t1;
        DrawCovaried(dt, dt*dt/2, dt*dt*dt/3, wt2, zt1t2);
        wt2 += wt1;
    }

    // Draws a new sample point inbetween of existing sample points.
    void DrawDependentWZ(double t1, double t2, double t3,
                         double wt1, double wt3, double zt1t3,
                         double &wt2, double &zt1t2)
    {
        // Assigns to wt2, zt1t2 a sample from distribution:
        // W(t2), Z(t1, t2) | W(t1), W(t3), Z(t1, t3); t1<t2<t3.
        double i1 = t2-t1;
        double i2 = t3-t2;
        double I = t3-t1;

        double varwt1t2 =  i1*i2*(i1*i1-i1*i2+i2*i2)/(I*I*I);
        double cov = i1*i1*i2*i2*(i2-i1)/(2*I*I*I);
        double varzt1t2 = i1*i1*i1*i2*i2*i2/(3*I*I*I);

        double wt1t2;

        // sample centered gaussians with approprioate covs
        DrawCovaried(varwt1t2, cov, varzt1t2, wt1t2, zt1t2);

        // add conditional mean
        double wt1t3 = wt3-wt1;
        wt1t2 += wt1t3*i1*(i1-2*i2)/(I*I)  + zt1t3*6*i1*i2/(I*I*I);
        zt1t2 += wt1t3*(-1)*i1*i1*i2/(I*I) + zt1t3*i1*i1*(i1+3*i2)/(I*I*I);

        wt2 = wt1+wt1t2;
    }

    // Randomness generators
    std::default_random_engine generator;
    std::normal_distribution<double> normal;
    double DrawNormal()
    {
        return normal(generator);
    }
    void DrawCorrelated(double cor, double &x, double &y)
    {
        // Set x,y to correlated normal samples with Cor(x,y)=cor
        double z1 = DrawNormal();
        double z2 = DrawNormal();
        x = sqrt(1-cor*cor)*z1 + cor*z2;
        y = z2;
    }
    void DrawCovaried(double xx, double xy, double yy, double &x, double &y)
    {
        // Set x,y to sample of N(0,{{xx,xy},{xy,yy}}) dist
        DrawCorrelated(xy/sqrt(xx*yy), x, y);
        x = sqrt(xx) * x;
        y = sqrt(yy) * y;
    }
};
