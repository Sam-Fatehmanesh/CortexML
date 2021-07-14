#include "Cortex.hpp"
#include <vector>
#include <iostream>
#include "CSVReader.hpp"

//implement return regular NN
//combine darwin + calc

using namespace af;
using namespace std;


int main(){
    vector<vector<float>> trainin;
    vector<vector<float>> trainout;
    vector<vector<float>> testin;
    vector<vector<float>> testout;

    CSVReader read = CSVReader("training.csv");
    read.importData();
    trainout = read.saveOutput(32);
    trainin = read.saveInput(1,31);

    CSVReader readtest = CSVReader("test.csv");
    readtest.importData();
    testout = readtest.saveOutput(32);
    testin = readtest.saveInput(1,31);

    /*
    in.push_back(vector<float>{1,1});
    in.push_back(vector<float>{1,0});
    in.push_back(vector<float>{0,1});
    in.push_back(vector<float>{0,0});
    out.push_back(vector<float>{1});
    out.push_back(vector<float>{0});
    out.push_back(vector<float>{0});
    out.push_back(vector<float>{1});
    */

    auto neuralstructure = NetSpecs(31);
    neuralstructure.addlayer(31,"sig");
    //neuralstructure.addlayer(31,"relu");
    neuralstructure.addlayer(1,"sig");
    int pop = 1;
    CortexNN model = CortexNN(neuralstructure,2,pop);
    model.installData(trainin, trainout, testin, testout);
    //model.evolveD(pop,2,.05,10,true);

    model.evolveG(20,.0001,.3, true);
    /*
    cout << in[0][0] << in[0][1] << endl;
    model.run(0);
    cout << in[1][0] << in[1][1] << endl;
    model.run(1);
    cout << in[2][0] << in[2][1] << endl;
    model.run(2);
    cout << in[3][0] << in[3][1] << endl;
    model.run(3);
    cout << "accuracy: " << model.accuracyClass() << endl;
    */

}