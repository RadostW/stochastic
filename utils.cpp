double pow(double x, int n)
{
  if(x==0 & n == 0) throw std::logic_error("Attempting 0^0!");
  double y = 1;
  while(n-- > 0) y *= x;
  return y;
}

double eval_poly(double x, double* coefs, int order)
{
  double y = 0;
  for(int i = 0; i <= order; i++) y += coefs[i]*pow(x, i);
  return y;
}

// Assuming f is an increasing polynomial on [x_min, x_max] returs its unique root or boundary argument with value closest to 0.
double solve_increasing_poly(double* coefs, int order, double x_min, double x_max, double x_guess, double f_abs_tol = 1e-4)
{
  double val_max = eval_poly(x_max, coefs, order);
  double val_min = eval_poly(x_min, coefs, order);
  if (val_max < f_abs_tol)
    return x_max;
  if (val_min > -f_abs_tol)
    return x_min;

  int n_iters = 0;
  double x = x_guess;
  while (1)
  {
    double f = eval_poly(x, coefs, order);
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
    if (n_iters > 100)
      throw std::logic_error("solve_increasing_poly infinite loop");
  }
}
