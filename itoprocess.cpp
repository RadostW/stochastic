#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <map>

using namespace std;

class ItoProcess
{
    // Used for obtaining trajectories from equation of type
    // dX = a(x) dt + b(x) dW
public:
    ItoProcess(tdouble nfa(tdouble), tdouble nfb(tdouble))
    {
        fa = nfa;
        fb = nfb;
        normal = normal_distribution<double>(0.0, 1.0);
        WeinerPath[0.] = 0.;
        ZetReported[0.] = 0.;
    }
    double WeinerValue(double t)
    {
        if (t < 0)
        {
            return 0;
        }
        else if (WeinerPath.count(t) == 1) // Repeated ask for the same value
        {
            return WeinerPath[t];
        }

        auto lower = WeinerPath.lower_bound(t);
        if (lower == WeinerPath.end()) // Beyond rightmost sample
        {
            lower--;
            double z = DrawNormal();
            WeinerPath[t] = sqrt(t - lower->first) * z + lower->second;
            return WeinerPath[t];
        }
        else
        {
            throw logic_error("Weiner subsampling not implemented");
        }
    }
    void WeinerResample()
    {
        WeinerPath.clear();
        WeinerPath[0.] = 0.;
        ZetReported.clear();
        ZetReported[0.] = 0;
    }
    double ZetValue(double a, double b)
    {
        // Return first irreducible double integral:
        // int_a^b int_a^q ds dW_q
        auto lower = ZetReported.lower_bound(a);
        auto upper = ZetReported.upper_bound(b);
        if (lower == ZetReported.end())
        {
            throw logic_error("Weiner extrapolation not implemented");
        }
        else if (next(lower) == ZetReported.end() && upper == ZetReported.end())
        {
            double dW = WeinerValue(b) - WeinerValue(a);
            double dt = b - a;

            double z1 = dW * (1. / sqrt(b - a));
            double z2 = DrawNormal();

            double dZ = 0.5 * (z1 + z2 / sqrt(3)) * dt * sqrt(dt);
            ZetReported[b] = dZ;
            return dZ;
        }
        else if (lower->first == a && upper->first == b && next(lower) == upper)
        {
            // Alreay reported this interval, report again
            return upper->second;
        }
        else
        {
            throw logic_error("Weiner subsampling not implemented");
        }
    }

    vector<double> SampleEuler(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.get_value());
            double a = fa(x).get_value();
            double b = fb(x).get_value();
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            x = tdouble(x.get_value() + a * dt + b * dW, 0);
        }
        return res;
    }

    vector<double> SampleMilstein(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.get_value());
            double a = fa(x).get_value();
            tdouble fbval = fb(x);
            double b = fbval.get_value();
            double bp = fbval.get_gradient()[0];
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            x = tdouble(x.get_value() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt), 0);
        }
        return res;
    }

    vector<double> SampleWagnerPlaten(double x0, double tmax, double dt)
    {
        double t = 0;
        tdouble x = tdouble(x0, 0);
        vector<double> res;

        for (int i = 0; dt * i < tmax; i++)
        {
            res.push_back(x.get_value());
            tdouble faval = fa(x);
            double a = faval.get_value();
            double ap = faval.get_gradient()[0];
            double app = faval.get_hessian()[0][0];

            tdouble fbval = fb(x);
            double b = fbval.get_value();
            double bp = fbval.get_gradient()[0];
            double bpp = fbval.get_hessian()[0][0];

            double z1 = DrawNormal();
            double z2 = DrawNormal();
            double dW = WeinerValue((i + 1) * dt) - WeinerValue(i * dt);
            double dZ = ZetValue(i * dt, (i + 1) * dt);
            
            x = tdouble(
                x.get_value() + a * dt + b * dW + 0.5 * b * bp * (dW * dW - dt) +
                    b * ap * dZ + 0.5 * (a * ap + 0.5 * b * b * app) * dt * dt +
                    (a * bp + 0.5 * b * b * bpp) * (dW * dt - dZ) +
                    0.5 * b * (b * bpp + bp * bp) * ((1. / 3.) * dW * dW - dt) * dW,
                0);
        }
        return res;
    }

private:
    tdouble (*fa)(tdouble);
    tdouble (*fb)(tdouble);
    default_random_engine generator;
    normal_distribution<double> normal;
    double DrawNormal()
    {
        return normal(generator);
    }
    map<double, double> WeinerPath;
    map<double, double> ZetReported; // Values of reported I(0,1) integral at given intervals, saved on rhs.
};
