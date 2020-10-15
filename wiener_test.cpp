#include "wiener.cpp"
#include <iostream>

/*
Moze napiszesz automatycznego unit testa tego wienera?
Np: niech weźmie przepyta punkty losowo z przedziału 1-1000 a potem kolejno z przedziału 1-1000
I potem test czy przyrosty maja dobrą sigme (np sprawdzi jaki odsetek > niz 2\sigma)
Czy kolejne są independent (np korelacja jest odpowiednio mała)
I potem to samo dla Z
i w końcu czy Z na przedziałach ma dobrą korelacje z W (i tutaj przedziały minimalnej wielkości najpierw a potem na przykład wielkości 5)
*/


bool testWiener(Wiener w, double T, double dt)
{
    int N = T/dt;

    //obtain values
    double increments[N];
    double z_values[N];
    for (int i = 0; i < N + 1; i++)
    {
        increments[i] = w.getValue(dt*(i+1)) - w.getValue(dt*i);
        z_values[i] = w.getZ(dt*i, dt*(i+1));
    }

    double Edw_squared = 0;
    double Ez_squared = 0;
    double Edw_z = 0;
    double Edw_lagdw = 0;
    for (int i = 0; i < N; i++)
    {
        Edw_squared+=increments[i]*increments[i];
        Ez_squared+=z_values[i]*z_values[i];
        Edw_z += increments[i] * z_values[i];
        Edw_lagdw += increments[i + 1] * increments[i];
    }
    Edw_squared /= N;
    Ez_squared /= N;
    Edw_z /= N;
    Edw_lagdw /= N;

    if (
        abs(Edw_squared - dt)/dt                  > 0.01 ||
        abs(Edw_z - dt*dt/2)/(dt*dt/2)            > 0.01 ||
        abs(Ez_squared - dt*dt*dt/3)/(dt*dt*dt/3) > 0.01 ||
        abs(Edw_lagdw)                            > 0.01)
    {
        std::cout << "dt=" << dt << std::endl 
            << "Edw_squared=" << Edw_squared << ",\tshould be " << dt << std::endl
            << "Edw_z=" << Edw_z << ",\t\tshould be " << dt*dt/2 << std::endl
            << "Ez_squared=" << Ez_squared << ",\tshould be " << dt*dt*dt/3 << std::endl
            << "Edw_lagdw=" << Edw_lagdw << ",\tshould be " << 0 << std::endl;
        return false;
    }
    else return true;
}

int main()
{
    Wiener w;
    int n = 1000;
    int T = 100000;

    //sample at n random points
    for (int i = 0; i < n; i++)
        w.getValue((rand() % 100*T)/100.);

    bool test1 = testWiener(w, T, 1);
    bool test5 = testWiener(w, T, 5); 
    if( !(test1 && test5) )
        throw logic_error("test failed");

    return 0;
}