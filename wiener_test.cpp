#include "wiener.cpp"
#include<iostream>

/*
Moze napiszesz automatycznego unit testa tego wienera?
Np: niech weźmie przepyta punkty losowo z przedziału 1-1000 a potem kolejno z przedziału 1-1000
I potem test czy przyrosty maja dobrą sigme (np sprawdzi jaki odsetek > niz 2\sigma)
Czy kolejne są independent (np korelacja jest odpowiednio mała)
I potem to samo dla Z
i w końcu czy Z na przedziałach ma dobrą korelacje z W (i tutaj przedziały minimalnej wielkości najpierw a potem na przykład wielkości 5)
*/

int main()
{
    Wiener w;
    int n = 100;
    int N = 1000;
    
    //sample at random points
    for(int i=0;i<n;i++) w.getValue( rand()%N + (rand()%100)/100. );
    
    //obtain values
    double increments[N];
    double z_values[N];
    for(int i=0;i<N+1;i++) 
    {
        increments[i] = w.getValue(i+1)-w.getValue(i);
        z_values[i] = w.getZ(i, i+1);
    }

    double w_quantile = 0;
    double z_quantile = 0;
    double cov_wz = 0;
    double cov_dw = 0;
    for(int i=0;i<N;i++) 
    {
        if(increments[i]>2)  w_quantile+=1./N;
        if(z_values[i]>2/3.) z_quantile+=1./N;
        cov_wz += increments[i]*z_values[i]/N;
        cov_dw += increments[i+1]*increments[i]/N;
    }
    int x;
    x++;

    
    return 0;
}