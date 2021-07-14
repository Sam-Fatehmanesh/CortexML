#include <vector>
#include <string>
#include <arrayfire.h>
#include <af/util.h>

using namespace std;

class Layerspecs
{
public:
    int incount;
    int Ncount;
    string func;
    af::dim4 dims;

    Layerspecs(int ncount,  string Func, int IN = 0)
    {
        Ncount = ncount;
        func = Func;
        incount = IN;
        dims = af::dim4(Ncount,incount+1);
    }
};

class NetSpecs {
public:
    int ins;
    vector<Layerspecs> layers;
    int laynum = 0;
    int outs;
    pair<int,int> max;
    explicit NetSpecs(int Inputs,int Outs = 0)
    {
        outs = Outs;
        ins = Inputs;
    }
    void addlayer(int ncount, const string& Func = "")
    {
        outs = ncount;
        laynum = laynum + 1;
        if (layers.empty())
        {
            layers.emplace_back(ncount,Func,ins);
        }
        else
        {
            int temp = layers.size()-1;
            int x = layers[temp].Ncount;
            layers.emplace_back(ncount,Func,x);
        }
    }
};
