#include "tdouble.cpp"
#include "itoprocess.cpp"
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

tdouble a_term(tdouble x)
{
    return -x;
};

tdouble b_term(tdouble x)
{
    return 1 / (x + 1);
};

int main()
{
    sample_some(a_term, b_term, 0, 1, 0.1);
    return 0;
}