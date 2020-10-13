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
        double Z = 0.;
        //it = samplePoints.lower_bound(t1);
        //for(auto it = samplePoints.lower_bound(t1); it.;)
        // TODO return Z
        return 1.;
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
            drawIndependentWZ(tMax, t, lower->second.w, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            samplePoints[t] = newSamplePoint;
        }
        else // between known values
        {
            double nextT = lower->first;
            samplePoint nextSamplePoint = lower->second;
            lower--;
            double prevT = lower->first;
            samplePoint prevSamplePoint = lower->second;

            samplePoint newSamplePoint;
            drawDependentWZ(prevT, t, nextT, prevSamplePoint.w, nextSamplePoint.w, nextSamplePoint.zToPrevPoint, newSamplePoint.w, newSamplePoint.zToPrevPoint);
            nextSamplePoint.zToPrevPoint = nextSamplePoint.zToPrevPoint - newSamplePoint.zToPrevPoint - (nextT-t)*newSamplePoint.w;
            samplePoints[t] = newSamplePoint;
        }
    }
    void drawIndependentWZ(double t1, double t2, double wt1, double &wt2, double &zt1t2)
    {
        // W(t2), Z(t1, t2) | W(t1)
        double dt = t2 - t1;
        DrawCovaried(dt, dt*dt/2, dt*dt*dt/3, wt2, zt1t2);
        wt2 += wt1;
    }
    void drawDependentWZ(double t1, double t2, double t3, double wt1, double wt3, double zt1t3, double &wt2, double &zt1t2) 
    {
        // W(t2), Z(t1, t2) | W(t1), W(t3), Z(t1, t3)
        double i1 = t2-t1;
        double i2 = t3-t2;
        double I = t3-t1;
        double varwt2 =  i1*i2*(i1*i1-i1*i2+i2*i2)/(I*I*I);
        double cov = i1*i1*i2*i2*(i2-i1)/(2*I*I*I);
        double warzt1t2 = i1*i1*i1*i2*i2*i2/(3*I*I*I);
        double dw = wt3-wt1;
        DrawCovaried(varwt2, cov, warzt1t2, wt2, zt1t2);
        zt1t2 += i1*i1*(-i1*(i1+i2)*dw+(i1+3*i2)*zt1t3)/(I*I*I);
        wt2 += (i1*(i1-2*i2)*(i1+i2)*dw + 6*i1*i2*zt1t3)/(I*I*I);
        wt2 += wt1;
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

int main()
{
    Wiener w;
    FILE *out;
    out = fopen("toplot.dat", "w");

    /*for(int i=0;i<10;i++)
    {
        double x = rand()%100;
        double tmp = w.getValue(x);
        printf("%d: %lf %lf\n",i,x,tmp);
    }
    for(int i=0;i<100;i++)
    {
        w.getValue(i);
    }*/

    for(int i=0;i<100;i++)
    {
        fprintf(out,"%lf\n",w.getValue(i));
    }

    fclose(out);
}