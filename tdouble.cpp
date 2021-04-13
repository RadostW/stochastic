// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include<array>
#include<iostream>
#include"pmath.cpp"

// Stores a value together with a Taylor expansion.
// Supports all arithmetic operations of double.
// Supports many functions of <cmath>.
class tdouble
{
    
 private:
    static const int maxvars = 5;
    double x;
    double gr;
    double hes;
    tdouble(double nx, double ngr, double nhes)
    {
        x = nx;
        gr = ngr;
        hes = nhes;
    }

 public:

    std::string ToString()
    {
        auto q = *this;
        std::string text;
        text += std::to_string(q.x);
        text+=q.gr>0?"+":"";
        text+=std::to_string(q.gr);
        text+="Dx";
        text+=q.hes>0?"+":"";
        text+=std::to_string(q.hes);
        text+="Dxx";
        
        return text;
    }

    tdouble()
    {
        x = 0;
        gr = 0;
        hes = 0;
    }   

    const double GetGradient()
    {
        return gr;
    }
    const double GetHessian()
    {
        return hes;
    }
    const double GetValue()
    {
        return x;
    }
    tdouble(double val)
    {
        x = val;
        gr = 0;
        hes = 0;
    }

    static tdouble Variable(double val)
    {
        return tdouble(val,1,0);
    }

    // Provides interface for creating tdouble functions out of double(double) functions
    // Applies transformation given by fun, when provided with derivative and second derivative
    tdouble Apply(double fun(double), double der(double), double dder(double))
    {
        double nx = fun(x);
        double ngr;
        double nhes;
        
        ngr = gr*der(x);
        nhes = hes*der(x) + gr*gr*dder(x);
        
        return tdouble(nx, ngr, nhes);
    }

    // Inbuild 1/x function
    // Helps with implementations of arithmetic operators
    tdouble Inverse() const
    {
        double nx = 1/x;
        double ngr;
        double nhes;
        
        ngr = -gr*(1/x)*(1/x);
        nhes = -hes*(1/x)*(1/x) + gr*gr*(2/(x*x*x));
        
        return tdouble(nx, ngr, nhes);
    }
    
    //ARITHMETIC OPERATORS
    // Addition with type.
    tdouble operator+(const tdouble& rhs) const
    {
        double nx = this->x + rhs.x;
        double ngr;
        double nhes;
        ngr = (rhs.gr) + (this->gr);
        nhes = rhs.hes + (this->hes);
        
        return tdouble(nx, ngr, nhes);
    }
    
    // Multiplication with type.
    tdouble operator*(const tdouble& rhs) const
    {
        double nx = this->x * rhs.x;
        double ngr;
        double nhes;
        ngr = x*(rhs.gr) + rhs.x * (this->gr);
        auto x1 = this->x;
        auto x2 = rhs.x;
        auto hes1 = this->hes;
        auto hes2 = rhs.hes;
        auto g1 = this->gr;
        auto g2 = rhs.gr;
        nhes = g1*g2+g2*g1+x1*hes2+x2*hes1;
        
        return tdouble(nx, ngr, nhes);
    }
    
    // Division by type.
    tdouble operator/(const tdouble&rhs) const
    {
        return (*this)*rhs.Inverse();
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
    
    tdouble& operator+=(const tdouble& rhs)
    {
        *this = *this + rhs;
        return *this;
    }
    tdouble& operator-=(const tdouble& rhs)
    {
        *this = *this + rhs*(-1.);
        return *this;
    }
    tdouble& operator*=(const tdouble& rhs)
    {
        *this = *this * rhs;
        return *this;
    }
    tdouble& operator/=(const tdouble& rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    friend std::ostream & operator <<(std::ostream &s, const tdouble q)
    {
        s << q.x;
        s << (q.gr>=0?"+":"");
        s << q.gr;
        s << "dx ";
        s << (q.hes>=0?"+":"");
        s << q.hes;
        s << "dxx";
        return s;
    }
    
};

tdouble operator-(const tdouble &q){
    return q*(-1.);
}

tdouble operator-(const tdouble& lhs, const tdouble& rhs)
{
    return lhs+rhs*(-1);
}

tdouble operator*(const double& lhs, const tdouble& rhs)
{
    return rhs*lhs;
}

tdouble operator/(const double& lhs, const tdouble& rhs)
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
    return x.Apply(tanh,tanhp,tanhpp);
}
