// Copyright 2020, Radost Waszkiewicz and Maciej Bartczak
// This project is licensed under the terms of the MIT license.
#pragma once
#include<cmath>

// This file is dedicated to defining function literals.
// These functions help define taylor expansion relations for cmath functions

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
