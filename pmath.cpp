// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include<cmath>

// This file is dedicated to defining function literals.
// These functions help define taylor expansion relations for cmath functions

// Trig functions
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

// Hyperbolic functions
double sech(double x)
{
    return 1.0/cosh(x);
}
double sechp(double x)
{
    return -sech(x)*tanh(x);
}
double sechpp(double x)
{
    return sech(x)-2*sech(x)*sech(x)*sech(x);
}

double tanhp(double x)
{
    return sech(x)*sech(x);
}
double tanhpp(double x)
{
    return -2*sech(x)*sech(x)*tanh(x);
}

