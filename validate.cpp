#include "tdouble.cpp"
#include "itoprocess.cpp"
#include "pmath.cpp"
#include <cmath>

// calculates trajectries integrated by various methods and different precision
// and compares against some "true" trajectory
void sample_some(tdouble a_term(tdouble), tdouble b_term(tdouble), double x0, double tmax, double dt)
{
    FILE *out;
    out = fopen("toplot.dat", "w");

    ItoProcess proc = ItoProcess(a_term, b_term);
    int N = 100;

    auto traj_true = proc.SampleWagnerPlaten(x0, tmax, dt / double(N));
    auto traj_euler = proc.SampleEuler(x0, tmax, dt);
    auto traj_milst = proc.SampleMilstein(x0, tmax, dt);
    auto traj_wagnp = proc.SampleWagnerPlaten(x0, tmax, dt);
    
    auto traj_euler5 = proc.SampleEuler(x0, tmax, dt/5);
    auto traj_milst5 = proc.SampleMilstein(x0, tmax, dt/5);
    auto traj_wagnp5 = proc.SampleWagnerPlaten(x0, tmax, dt/5);

    double euler_error_sum = 0;
    double milst_error_sum = 0;
    double wagnp_error_sum = 0;

    double euler5_error_sum = 0;
    double milst5_error_sum = 0;
    double wagnp5_error_sum = 0;

    fprintf(out, "t true euler milstein wagnerp\n");
    for (int i = 0; i < traj_true.size() - 10; i++)
    {
        if (i % N)
            fprintf(out, "%lf %lf\n", dt * i / double(N), traj_true[i]);
        else
        {
            fprintf(out, "%lf %lf %lf %lf %lf\n", dt * i, traj_true[i], traj_euler[i / N], traj_milst[i / N], traj_wagnp[i / N]);
            euler_error_sum += fabs(traj_true[i] - traj_euler[i / N]);
            milst_error_sum += fabs(traj_true[i] - traj_milst[i / N]);
            wagnp_error_sum += fabs(traj_true[i] - traj_wagnp[i / N]);
        }
    }
    fclose(out);

    //euler_error_sum /= double(tmax)/dt;
    //milst_error_sum /= double(tmax)/dt;
    //wagnp_error_sum /= double(tmax)/dt;
    printf("euler_error_sum=%lf\n", euler_error_sum);
    printf("milst_error_sum=%lf\n", milst_error_sum);
    printf("wagnp_error_sum=%lf\n\n", wagnp_error_sum);

    printf("euler_error_end=%lf\n", traj_euler.back() - traj_true.back());
    printf("milst_error_end=%lf\n", traj_milst.back() - traj_true.back());
    printf("wagnp_error_end=%lf\n", traj_wagnp.back() - traj_true.back());
}

void viz_order(tdouble a_term(tdouble), tdouble b_term(tdouble), double x0, double tmax)
{
    FILE *out;
    out = fopen("toplot.dat", "w");
    fprintf(out, "steps euler milst wagnp\n");
    
    ItoProcess proc = ItoProcess(a_term, b_term);
    
    auto traj_true = proc.SampleWagnerPlaten(x0, tmax, tmax/1000.);
    double dt;
    for(int steps=10; steps<1000; steps++)
    {
        dt = tmax/steps;
        auto traj_euler = proc.SampleEuler(x0, tmax, dt);
        auto traj_milst = proc.SampleMilstein(x0, tmax, dt);
        auto traj_wagnp = proc.SampleWagnerPlaten(x0, tmax, dt);
        
        double error_euler = fabs(traj_euler.back() - traj_true.back());
        double error_milst = fabs(traj_milst.back() - traj_true.back());
        double error_wagnp = fabs(traj_wagnp.back() - traj_true.back());
        fprintf(out, "%lf %lf %lf %lf\n", steps, error_euler, error_milst, error_wagnp);
        printf("%lf %lf %lf %lf\n", steps, error_euler, error_milst, error_wagnp);
    }
}
const double ceiling = 50.;
tdouble mobility(tdouble location)
{
    tdouble x = 1.0 / location;
    tdouble mobdown = 0.986292 - x - 0.00688 * cos(10.86762 + 8.092 * x) + 0.02057 * sin(2.506 + x * (3.074 + 2.227 * x));
    x = 1.0 / (ceiling - location);
    tdouble mobup = 0.986292 - x - 0.00688 * cos(10.86762 + 8.092 * x) + 0.02057 * sin(2.506 + x * (3.074 + 2.227 * x));
    return location > (0.5 * ceiling) ? mobup : mobdown;
}
tdouble dmobility(tdouble location)
{
    tdouble h = location;
    tdouble dmobdown = ((-0.09161878 - 0.06323218 * h) *
                            cos(2.506 + (2.227 + 3.074 * h) / (h * h)) +
                        h * (1. - 0.05567296 * sin(10.86762 + 8.092 / h))) /
                        (h * h * h);
    h = ceiling - location;
    tdouble dmobup = -((-0.09161878 - 0.06323218 * h) *
                            cos(2.506 + (2.227 + 3.074 * h) / (h * h)) +
                        h * (1. - 0.05567296 * sin(10.86762 + 8.092 / h))) /
                        (h * h * h);
    return location > (0.5 * ceiling) ? dmobup : dmobdown;
}

tdouble a_term(tdouble x)
{
    return dmobility(x) - 1;
}
tdouble b_term(tdouble x)
{
    return sqrt(2 * mobility(x));
}

int main()
{
    viz_order(a_term, b_term, 1, 10);
    return 0;
}