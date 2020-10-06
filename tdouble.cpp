#include<array>

using namespace std;
#define maxvars 10

class tdouble //double with taylor expansion
{
    private:
    double x;
    array<double,maxvars> gr;
    array< array<double, maxvars>, maxvars> hes;
    tdouble(double nx,array<double,maxvars> ngr, array<array<double, maxvars>,maxvars> nhes)
    {
        x = nx;
        gr = ngr;
        hes = nhes;
    }
    public:
    const array<double,maxvars> get_gradient()
    {
        return gr;
    }
    const array< array<double,maxvars>, maxvars> get_hessian()
    {
        return hes;
    }
    const double get_value()
    {
        return x;
    }
    tdouble(double val,int id) //create a new variable
    {
        if(id > maxvars) throw 0xBAD;
        x = val;
        for(int i=0;i<maxvars;i++)gr[i]=0;
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)hes[i][j]=0;
        gr[id]=1;
    }
    tdouble apply(double fun(double),double der(double),double dder(double))
    {
        double nx = fun(x);
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = gr[i]*der(x);
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)nhes[i][j] = hes[i][j]*der(x) + gr[i]*gr[j]*dder(x);
        
        return tdouble(nx,ngr,nhes);
    }
    tdouble inverse() const
    {
        double nx = 1/x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        
        for(int i=0;i<maxvars;i++)ngr[i] = -gr[i]*(1/x)*(1/x);
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)nhes[i][j] = -hes[i][j]*(1/x)*(1/x) + gr[i]*gr[j]*(2/(x*x*x));
        
        return tdouble(nx,ngr,nhes);
    }
    
    
    
    //ARITHMETIC OPERATORS
    //addition with type
    tdouble operator+(const tdouble& rhs) const
    {
        double nx = this->x + rhs.x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (rhs.gr)[i] + (this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=rhs.hes[i][j] + (this->hes)[i][j];
        
        return tdouble(nx,ngr,nhes);
    }
    //addition with scalar
    tdouble operator+(const double rhs) const
    {
        double nx = this->x + rhs;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = (this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=(this->hes)[i][j];
        
        return tdouble(nx,ngr,nhes);
    }
    //multiplicaiton with scalar
    tdouble operator*(const double& rhs) const
    {
        double nx = x * rhs;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = rhs*(this->gr)[i];
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j]=rhs*(this->hes)[i][j];
        
        return tdouble(nx,ngr,nhes);
    }
    //multiplication with type
    tdouble operator*(const tdouble& rhs) const
    {
        double nx = this->x * rhs.x;
        array<double,maxvars> ngr;
        array< array<double, maxvars>, maxvars> nhes;
        for(int i=0;i<maxvars;i++)ngr[i] = x*(rhs.gr)[i] + rhs.x * (this->gr)[i];
        auto x1 = this->x;
        auto x2 = rhs.x;
        auto hes1 = this->hes;
        auto hes2 = rhs.hes;
        auto g1 = this->gr;
        auto g2 = rhs.gr;
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++) nhes[i][j] = g1[i]*g2[j]+g2[i]*g1[j]+x1*hes2[i][j]+x2*hes1[i][j];
        
        return tdouble(nx,ngr,nhes);
    }
    //division by type
    tdouble operator/(const tdouble&rhs) const
    {
        return (*this)*rhs.inverse();
    }
    //division by scalar
    tdouble operator/(const double rhs) const
    {
        return (*this)*(1/rhs);
    }

    //comparison operators
    bool operator< (const tdouble &y){
        return x < y.x;
    }
    bool operator<= (const tdouble &y){
        return x <= y.x;
    }
    bool operator> (const tdouble &y){
        return x > y.x;
    }
    bool operator>= (const tdouble &y){
        return x >= y.x;
    }
    bool operator== (const tdouble &y){
        return x == y.x;
    }
    bool operator!= (const tdouble &y){
        return x != y.x;
    }
    
    
    
    //printer
    friend std::ostream & operator <<(std::ostream &s, const tdouble q)
    {
        string text;
        text += std::to_string(q.x);
        for(int i=0;i<maxvars;i++)
        {
            if(q.gr[i]!=0)
            {
                text+=q.gr[i]>0?"+":"";
                text+=std::to_string(q.gr[i]);
                text+="D";
                text+=std::to_string(i);
            }
        }
        for(int i=0;i<maxvars;i++)for(int j=0;j<maxvars;j++)
        {
            if(q.hes[i][j]!=0)
            {
                text+=q.hes[i][j]>0?"+":"";
                text+=std::to_string(q.hes[i][j]);
                text+="D";
                text+=std::to_string(i);
                text+=std::to_string(j);
            }
        }
        return s << text;
    }
};


tdouble operator+(double lhs,const tdouble& rhs)
{
    return rhs+lhs;
} 

tdouble operator-(const tdouble& lhs,const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(double lhs,const tdouble& rhs)
{
    return lhs+rhs*(-1);
}
tdouble operator-(const tdouble&lhs, double rhs)
{
    return lhs+(-rhs);
}

tdouble operator*(double lhs,const tdouble& rhs)
{
    return (rhs*lhs);
}
tdouble operator/(double lhs,const tdouble& rhs)
{
    return lhs*(rhs.inverse());
}

tdouble cos(tdouble x)
{
    return x.apply(cos,nsin,ncos);
}
tdouble sin(tdouble x)
{
    return x.apply(sin,cos,nsin);
}
tdouble tan(tdouble x)
{
    return x.apply(tan,sec2,tanpp);
}
tdouble exp(tdouble x)
{
    return x.apply(exp,exp,exp);
}
tdouble log(tdouble x)
{
    return x.apply(log,logp,logpp);
}
tdouble sqrt(tdouble x)
{
    return x.apply(sqrt,sqrtp,sqrtpp);
}