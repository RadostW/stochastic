#include <functional>

// assuming f is continuous and increasing, returns boundary with value closest to 0 if there is no root
double find_root_bin_search(std::function<double(double)> func, double x_min, double x_max, double x_guess, double f_abs_tol = 1e-4)
{
  double func_max = func(x_max);
  double func_min = func(x_min);
  if (func_max < f_abs_tol)
    return x_max;
  if (func_min > -f_abs_tol)
    return x_min;

  int n_iters = 0;
  double x = x_guess;
  while (1)
  {
    double f = func(x);
    if (f > f_abs_tol)
    {
      x_max = x;
      x = (x + x_min) / 2;
    }
    else if (f < -f_abs_tol)
    {
      x_min = x;
      x = (x + x_max) / 2;
    }
    else
      return x;

    x = (x_min + x_max) / 2;
    n_iters += 1;
    if (n_iters > 30)
      throw std::logic_error("find root bin search infinite loop");
  }
}
