#include<stdio.h>
#include<array>
#include<string>
#include<iostream>
#include<cmath>
using namespace std;
#define maxvars 10

class tdouble //double with taylor expansion
{
    private:
    double x;
    array<double,maxvars> gr;
    array< array<double, maxvars>, maxvars> hes;
    tdouble(double nx,array<double,maxvars> ngr, array<array<double, maxvars>,maxvars> nhes)
    {
        x = nx;
        gr = ngr;
        hes = nhes;
    }
    public:
    tdouble(double val,int id) //create a new variable
    {
        if(id > maxvars) throw 0xBAD;
        x = val;
        for(int i=0;i<maxvars;i++)gr[i]=0;
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)hes[i][j]=0;
        gr[id]=1;
    }
    tdouble apply(double fun(double),double der(double),double dder(double))
    {
        double nx = fun(x);
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = gr[i]*der(x);
        
        // TODO: hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    tdouble inverse() const
    {
        double nx = 1/x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = -gr[i]*(1/x)*(1/x);
        
        // TODO: hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    
    
    
    //ARITHMETIC OPERATORS
    //addition with type
    tdouble operator+(const tdouble& rhs) const
    {
        double nx = this->x + rhs.x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (rhs.gr)[i] + (this->gr)[i];
        
        //TODO: add hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    //addition with scalar
    tdouble operator+(const double rhs) const
    {
        double nx = this->x + rhs;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (this->gr)[i];
        
        //TODO: add hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    //multiplicaiton with type 
    tdouble operator*(const double& rhs) const
    {
        double nx = x * rhs;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = rhs*(this->gr)[i];
        
        //TODO: add hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    //multiplication with scalar
    tdouble operator*(const tdouble& rhs) const
    {
        double nx = this->x * rhs.x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = x*(rhs.gr)[i] + rhs.x * (this->gr)[i];
        
        //TODO: add hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    //division by type
    tdouble operator/(const tdouble&rhs) const
    {
        double nx = x / rhs.x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (-x*(rhs.gr)[i] + rhs.x * (this->gr)[i])/(rhs.x*rhs.x);
        
        //TODO: add hessian support
        
        return tdouble(nx,ngr,nhes);
    }
    //division by scalar
    tdouble operator/(const double rhs) const
    {
        return (*this)*(1/rhs);
    }
    
    //printer
    friend std::ostream & operator <<(std::ostream &s, const tdouble q)
    {
        string text;
        text += std::to_string(q.x);
        for(int i=0;i<maxvars;i++)
        {
            if(q.gr[i]!=0)
            {
                text+=q.gr[i]>0?"+":"";
                text+=std::to_string(q.gr[i]);
                text+="D";
                text+=std::to_string(i);
            }
        }
        return s << text;
    }
};


tdouble operator+(double lhs,const tdouble& rhs)
{
    return rhs+lhs;
} 

tdouble operator-(tdouble& lhs,const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(double lhs,const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(const tdouble&lhs, double rhs)
{
    return lhs+(-rhs);
}

tdouble operator*(double lhs,const tdouble& rhs)
{
    return (rhs*lhs);
}
tdouble operator/(double lhs,const tdouble& rhs)
{
    return lhs*(rhs.inverse());
}




double nsin(double x)
{
    return -sin(x);
}
double ncos(double x)
{
    return -cos(x);
}
double sec2(double x)
{
    return 1.0/(cos(x)*cos(x));
}
double tanpp(double x)
{
    return 2.0*sec2(x)*tan(x);
}
double logp(double x)
{
    return 1.0/x;
}
double logpp(double x)
{
    return -1.0/(x*x);
}
double sqrtp(double x)
{
    return 0.5/sqrt(x);
}
double sqrtpp(double x)
{
    return -0.25/(x*sqrt(x));
}

tdouble cos(tdouble x)
{
    return x.apply(cos,nsin,ncos);
}
tdouble sin(tdouble x)
{
    return x.apply(sin,cos,nsin);
}
tdouble tan(tdouble x)
{
    return x.apply(tan,sec2,tanpp);
}
tdouble exp(tdouble x)
{
    return x.apply(exp,exp,exp);
}
tdouble log(tdouble x)
{
    return x.apply(log,logp,logpp);
}
tdouble sqrt(tdouble x)
{
    return x.apply(sqrt,sqrtp,sqrtpp);
}


int main()
{
    tdouble x(1,0);
    tdouble y(1,1);
    cout << 1-sqrt(x+2*y)/x;
}
