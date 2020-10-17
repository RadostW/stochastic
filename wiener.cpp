#include<map>
#include<random>

using namespace std;

struct samplePoint
{
    double w;
    double zToPrevPoint;
};

class Wiener
{
public:
    Wiener()
    {
        samplePoints[0.].w = 0;
        samplePoints[0.].zToPrevPoint = 0;        
    }
    double getValue(double t)
    {
        ensureSamplePoint(t);
        return samplePoints[t].w;
    }
    double getZ(double t1, double t2)
    {
        ensureSamplePoint(t1);
        ensureSamplePoint(t2);

        auto lower = samplePoints.lower_bound(t1);
        auto upper = samplePoints.lower_bound(t2);
        auto it = samplePoints.begin();
        
            
        double accumulator = 0;
        double tm = t2;
        double ta, tb;
        for(it = lower;lower!=upper;lower++)
        {
            ta = it->first;
            tb = next(it)->first;
            accumulator += next(it)->second.zToPrevPoint;
            accumulator += (getValue(tb)-getValue(ta))*(tm - ta);
        }
        return accumulator;

    }

private:
    map<double, samplePoint> samplePoints;
    void ensureSamplePoint(double t)
    {
        if (t < 0) return;

        auto lower = samplePoints.lower_bound(t);
        if (lower->first == t) return; // exact match
        if (lower == samplePoints.end()) // beyond known values
        {
            samplePoint newSamplePoint;
            double tMax = samplePoints.rbegin()->first;
            double lastW = samplePoints.rbegin()->second.w;
            drawIndependentWZ(tMax, t, lastW, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            samplePoints[t] = newSamplePoint;
        }
        else // between known values
        {
            double nextT = lower->first;
            samplePoint & nextSamplePoint = lower->second;
            lower--;
            double prevT = lower->first;
            samplePoint prevSamplePoint = lower->second;

            samplePoint newSamplePoint;
            drawDependentWZ(prevT, t, nextT, prevSamplePoint.w, nextSamplePoint.w, nextSamplePoint.zToPrevPoint, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            nextSamplePoint.zToPrevPoint = nextSamplePoint.zToPrevPoint - newSamplePoint.zToPrevPoint - (nextT-t)*(newSamplePoint.w-prevSamplePoint.w);
            samplePoints[t] = newSamplePoint;
        }
    }
    void drawIndependentWZ(double t1, double t2, double wt1, double &wt2, double &zt1t2)
    {
        // assigns to wt2, zt1t2 sample from W(t2), Z(t1, t2) | W(t1); t1<t2
        double dt = t2 - t1;
        DrawCovaried(dt, dt*dt/2, dt*dt*dt/3, wt2, zt1t2);
        wt2 += wt1;
    }
    void drawDependentWZ(double t1, double t2, double t3, double wt1, double wt3, double zt1t3, double &wt2, double &zt1t2) 
    {
        // assigns to wt2, zt1t2 sample from W(t2), Z(t1, t2) | W(t1), W(t3), Z(t1, t3); t1<t2<t3
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

    // sampling routines
    default_random_engine generator;
    normal_distribution<double> normal;
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
};
