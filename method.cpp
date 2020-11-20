#include "tdouble.cpp"
#include "wiener.cpp"
#include <functional>

using namespace std;

// assuming f is continuous and increasing, returns boundary with value closest to 0 if there is no root
double find_root_bin_search(function<double(double)> func, double x_min, double x_max, double x_guess, double f_abs_tol=1e-4)
{
  double func_max = func(x_max);
  double func_min = func(x_min);
  if(func_max < f_abs_tol)
    return x_max;
  if(func_min > -f_abs_tol)
    return x_min;
  
  int n_iters = 0;
  double x = x_guess;
  while(1)
  {
    double f = func(x);
    if(f > f_abs_tol)
    {
      x_max = x;
      x = (x+x_min)/2;
    }
    else if(f < -f_abs_tol)
    {
      x_min = x;
      x = (x+x_max)/2;
    }
    else
      return x;
    
    x = (x_min+x_max)/2;    
    n_iters += 1;
    if(n_iters > 30)
      throw logic_error("find root bin search infinite loop");
  }
}

/*
given Ito-Taylor expansion
x = x_0 + I_0 L_0 x + I_1 L_1 x
  = x_0 + I_0 a + I_1 b
  = x_0 + I_0 a_0 + I_00 L_0 a + I_01 L_1 a
        + I_1 b_0 + I_10 L_0 b + I_11 L_1 b
  = x_0 + I_0 a_0 + I_00 (aa'+b^2a''/2)_0 
*/


typedef struct {double t; double x; double w;} datapoint;

vector<datapoint> method(
    Wiener wiener,
    double x0,
    double T,
    tdouble drift(tdouble),
    tdouble volatility(tdouble),
    char method='e',
    bool adaptive=false,
    int err_order=0,
    int n_steps=1000,
    double target_error_density=0.1
    //local_error_variance_order=None
    )
{
    auto fa = drift;
    auto fb = volatility;

    tdouble x = tdouble(x0, 0);
    double t = 0;
    double dt = T/n_steps;
    double dt_max = dt*100.;
    double dt_min = dt/100.;
    
    double w = 0;
    double dx;
    double dw;
    double dz;

    vector<datapoint> trajectory;

    double a;
    double ap;
    double app;
    double b;
    double bp;
    double bpp;
    double coef0;
    double coef1;
    double coef00;
    double coef01;
    double coef10;
    double coef11;

    while(1)
    {
        datapoint d = datapoint();
        d.t = t;
        d.x = x.GetValue();
        d.w = w;
        trajectory.push_back(d);
        // printf("%lf %lf %lf\n", t, x.GetValue(), w);

        if(t>=T)
            break;

        tdouble ta = fa(x);
        tdouble tb = fb(x);

        a   = ta.GetValue();
        ap  = ta.GetGradient()[0];
        app = ta.GetHessian()[0][0];
        
        b   = tb.GetValue();
        bp  = tb.GetGradient()[0];
        bpp = tb.GetHessian()[0][0];
        
        coef0  = a;                     // sq order = 2
        coef1  = b;                     // sq order = 1
        coef00 = a*ap + b*b*app/2.;                  // sq order = 4
        coef01 = b*ap;                  // sq order = 3
        coef10 = a*bp + b*b*bpp/2.;     // sq order = 3
        coef11 = b*bp;                  // sq order = 2
        // double coef110 = (b*bpp + bp*bp)*a + (b*bppp + 3*bp*bpp)*b*b/2  // sq order = 4
        // double coef111 = b*(b*bpp + bp*bp);     // sq order = 3

        printf("%lf %lf %lf %lf\n", x, b, bp, b*bp, coef11);

        /*if(adaptive)
        {
            function<double(double)> local_error_var_estimate = [](double dt){return 0.;};
            
            if( err_order >= 2 & method == 'e')
                local_error_var_estimate = [coef11](double dt){
                    return coef11*coef11*dt*dt;
                };
            if(err_order >= 3 & (method == 'e' | method == 'm'))
                local_error_var_estimate = [local_error_var_estimate, coef01, coef10, coef111](double dt){
                    return local_error_var_estimate(dt) + (coef01*coef01+coef10*coef10+coef111*coef111)*dt*dt*dt;
                };
            if(err_order >= 4 & (method == 'e' | method == 'm' | method == 'w'))
                local_error_var_estimate = [local_error_var_estimate, coef00](double dt){
                    return local_error_var_estimate(dt) +coef00*coef00*dt*dt*dt*dt;
                };
            
            
            dt = find_root_bin_search(
                [local_error_var_estimate, target_error_density](double dt){return local_error_var_estimate(dt) - target_error_density*target_error_density*dt*dt;},
                dt_min, dt_max, dt
            );            
        }*/

        dt = min(dt, T-t);

        dx = 0;
        dw = wiener.GetValue(t+dt) - wiener.GetValue(t);
        dz = wiener.GetZ(t, t+dt);
        
        if(method == 'e')
            dx = coef0*dt + coef1*dw;
        if(method == 'm')
            dx = coef0*dt + coef1*dw + coef11*(dw*dw-t)/2;
        /*switch (method)
        {
            case 'w':
                dx += coef01*dz + coef10*(dw*dt-dz) + coef111*((1. / 3.) * dw * dw - dt) * dw;
            case 'm':
                dx += coef11*(dw*dw-t)*0.5;
            case 'e':
                dx += coef0*dt + coef1*dw;
        }     */   
        t += dt;
        x = x+dx;
        w += dw;
    }
    
    datapoint d = datapoint();
    d.t = t;
    d.x = x.GetValue();
    d.w = w;
    trajectory.push_back(d);
    
    return trajectory;
}

double a=0.5;
tdouble a_term(tdouble x)
{
    return -0.5*a*a*tanh(x)*sech(x)*sech(x);
}
tdouble b_term(tdouble x)
{
    return a*sech(x);
}
double exact(double w, double x0){
    return asinh(a*w + sinh(x0));
}

int main(){
    Wiener w = Wiener();
    double x0 = 0.1;
    double T = 100;
    
    auto traj = method(w, x0, T, a_term, b_term, 'e');
    
    FILE *out;
    out = fopen("toplot.dat", "w");
    for(auto it = traj.begin(); it != traj.end(); it++)
    {
        fprintf(out, "%lf %lf %lf\n", it->x, it->w, exact(it->w, x0));
    }
    fclose(out);
    return 0;
    
    double errEuler = 0;
    double errMilstein = 0;
    
    int n_proc = 100;
    for(int i=0;i<n_proc;i++)
    {
        Wiener w = Wiener(i);    
        double valExact    = exact(w.GetValue(T), x0);
        double valEuler    = method(w, x0, T, a_term, b_term, 'e').back().x;
        double valMilstein = method(w, x0, T, a_term, b_term, 'm').back().x;
        
        errEuler    += (valEuler-valExact)*(valEuler-valExact);
        errMilstein += (valMilstein-valExact)*(valMilstein-valExact);
    }
    errEuler = sqrt(errEuler/n_proc);
    errMilstein = sqrt(errMilstein/n_proc);
    printf("%lf\n%lf", errEuler, errMilstein);
/*
    double errWagnerPlaten=0;
    double errMilstein=0;
    double errEuler=0;
    double errAdaptiveMilstein=0;

    double stepsAdaptiveMilstein=0;

        double valWagnerPlaten = *proc.SampleWagnerPlaten(x0, tmax, dt).rbegin();
        double valMilstein     = *proc.SampleMilstein(x0, tmax, dt).rbegin();
        double valEuler        = *proc.SampleEuler(x0, tmax, dt).rbegin();
        auto samAdaptiveMilstein = proc.SampleAdaptiveMilstein(x0, 0, tmax, (1./2.41)*dt);
        double valAdaptiveMilstein= *samAdaptiveMilstein.rbegin();
        stepsAdaptiveMilstein += samAdaptiveMilstein.size()-1;
        
        if(i%10==0) printf("%4d %5.2lf    %5.2lf %5.2lf %5.2lf %5.2lf\n",i,
                                valExact,
                            valWagnerPlaten-valExact,valMilstein-valExact,valEuler-valExact,valAdaptiveMilstein-valExact);

        errWagnerPlaten += (valWagnerPlaten-valExact)*(valWagnerPlaten-valExact);
        errMilstein += (valMilstein-valExact)*(valMilstein-valExact);
        errEuler += (valEuler-valExact)*(valEuler-valExact);
        errAdaptiveMilstein += (valAdaptiveMilstein-valExact)*(valAdaptiveMilstein-valExact);

        proc.ResetRealization();
    }*/
}