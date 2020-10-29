// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include<array>
#include"pmath.cpp"

// Stores a value together with a Taylor expansion.
// Supports all arithmetic operations of double.
// Supports many functions of <cmath>.
class tdouble
{
 private:
    // TODO(2020 October 19) make maxvars a variable rather than a const
    static const int maxvars = 2;
    double x;
    std::array<double, maxvars> gr;
    std::array< std::array<double, maxvars>, maxvars> hes;
    tdouble(double nx, std::array<double, maxvars> ngr, std::array<std::array<double, maxvars>, maxvars> nhes)
    {
        x = nx;
        gr = ngr;
        hes = nhes;
    }
 public:
    const std::array<double, maxvars> GetGradient()
    {
        return gr;
    }
    const std::array< std::array<double, maxvars>, maxvars> GetHessian()
    {
        return hes;
    }
    const double GetValue()
    {
        return x;
    }
    tdouble(double val, int id)
    {
        if(id > maxvars) throw 0xBAD;
        x = val;
        for(int i=0;i<maxvars;i++)gr[i]=0;
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)hes[i][j]=0;
        gr[id]=1;
    }

    // Provides interface for creating tdouble functions out of double(double) functions
    // Applies transformation given by fun, when provided with derivative and second derivative
    tdouble Apply(double fun(double), double der(double), double dder(double))
    {
        double nx = fun(x);
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = gr[i]*der(x);
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)nhes[i][j] = hes[i][j]*der(x) + gr[i]*gr[j]*dder(x);
        
        return tdouble(nx, ngr, nhes);
    }

    // Inbuild 1/x function
    // Helps with implementations of arithmetic operators
    tdouble Inverse() const
    {
        double nx = 1/x;
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = -gr[i]*(1/x)*(1/x);
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)nhes[i][j] = -hes[i][j]*(1/x)*(1/x) + gr[i]*gr[j]*(2/(x*x*x));
        
        return tdouble(nx, ngr, nhes);
    }
    
    //ARITHMETIC OPERATORS
    // Addition with type.
    tdouble operator+(const tdouble& rhs) const
    {
        double nx = this->x + rhs.x;
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (rhs.gr)[i] + (this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=rhs.hes[i][j] + (this->hes)[i][j];
        
        return tdouble(nx, ngr, nhes);
    }
    // Addition with scalar.
    tdouble operator+(const double rhs) const
    {
        double nx = this->x + rhs;
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=(this->hes)[i][j];
        
        return tdouble(nx, ngr, nhes);
    }
    // Multiplicaiton with scalar.
    tdouble operator*(const double& rhs) const
    {
        double nx = x * rhs;
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = rhs*(this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=rhs*(this->hes)[i][j];
        
        return tdouble(nx, ngr, nhes);
    }
    // Multiplication with type.
    tdouble operator*(const tdouble& rhs) const
    {
        double nx = this->x * rhs.x;
        std::array<double, maxvars> ngr;
        std::array< std::array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = x*(rhs.gr)[i] + rhs.x * (this->gr)[i];
        auto x1 = this->x;
        auto x2 = rhs.x;
        auto hes1 = this->hes;
        auto hes2 = rhs.hes;
        auto g1 = this->gr;
        auto g2 = rhs.gr;
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j] = g1[i]*g2[j]+g2[i]*g1[j]+x1*hes2[i][j]+x2*hes1[i][j];
        
        return tdouble(nx, ngr, nhes);
    }
    // Division by type.
    tdouble operator/(const tdouble&rhs) const
    {
        return (*this)*rhs.Inverse();
    }
    // Division by scalar.
    tdouble operator/(const double rhs) const
    {
        return (*this)*(1/rhs);
    }

    // Comparison operators tdouble.
    bool operator< (const tdouble &y) const {
        return x < y.x;
    }
    bool operator<= (const tdouble &y) const {
        return x <= y.x;
    }
    bool operator> (const tdouble &y) const {
        return x > y.x;
    }
    bool operator>= (const tdouble &y) const {
        return x >= y.x;
    }
    bool operator== (const tdouble &y) const {
        return x == y.x;
    }
    bool operator!= (const tdouble &y) const {
        return x != y.x;
    }
    
    // Comparison operators with double.
    bool operator< (double y) const {
        return x < y;
    }
    bool operator<= (double y) const {
        return x <= y;
    }
    bool operator> (double y) const {
        return x > y;
    }
    bool operator>= (double y) const {
        return x >= y;
    }
    bool operator== (double y) const {
        return x == y;
    }
    bool operator!= (double y) const {
        return x != y;
    }
    
    friend std::ostream & operator <<(std::ostream &s, const tdouble q)
    {
        std::string text;
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
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)
        {
            if(q.hes[i][j]!=0)
            {
                text+=q.hes[i][j]>0?"+":"";
                text+=std::to_string(q.hes[i][j]);
                text+="D";
                text+=std::to_string(i);
                text+=std::to_string(j);
            }
        }
        return s << text;
    }
};

// Comparison operators with double.
bool operator< (double x, const tdouble &y){
    return y > x;
}
bool operator<= (const double &x, const tdouble &y){
    return y >= x;
}
bool operator> (const double &x, const tdouble &y){
    return y < x;
}
bool operator>= (const double &x, const tdouble &y){
    return y <= x;
}
bool operator== (const double &x, const tdouble &y){
    return y == x;
}
bool operator!= (const double &x, const tdouble &y){
    return y != x;
}


tdouble operator-(const tdouble &q){
    return q*(-1.);
}

tdouble operator+(double lhs, const tdouble& rhs)
{
    return rhs+lhs;
} 

tdouble operator-(const tdouble& lhs, const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(double lhs, const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(const tdouble&lhs, double rhs)
{
    return lhs+(-rhs);
}

tdouble operator*(double lhs, const tdouble& rhs)
{
    return (rhs*lhs);
}
tdouble operator/(double lhs, const tdouble& rhs)
{
    return lhs*(rhs.Inverse());
}


// Common functions
tdouble cos(tdouble x)
{
    return x.Apply(cos, nsin, ncos);
}
tdouble sin(tdouble x)
{
    return x.Apply(sin, cos, nsin);
}
tdouble tan(tdouble x)
{
    return x.Apply(tan, sec2, tanpp);
}
tdouble exp(tdouble x)
{
    return x.Apply(exp, exp, exp);
}
tdouble log(tdouble x)
{
    return x.Apply(log, logp, logpp);
}
tdouble sqrt(tdouble x)
{
    return x.Apply(sqrt, sqrtp, sqrtpp);
}
tdouble sech(tdouble x)
{
    return x.Apply(sech,sechp,sechpp);
}
tdouble tanh(tdouble x)
{
    return x.Apply(tanh,tanhp,tanpp);
}
