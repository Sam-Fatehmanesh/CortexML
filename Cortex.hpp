#include <arrayfire.h>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>
#include "NeuralTensor.hpp"

using namespace af;
using namespace std;

struct ActFuncs
{
    string Sig = "sig";
    string Relu = "relu";
    string None = "";
};
struct Optimizer
{
    string Adam = "adam";
    string Static = "static";
};

//class for traditional non recurrent supervised learning NNs
class CortexNN
{
private:
    //initializing
    int popN;//number of NNs
    //random engine
    af::randomEngine rengine = af::randomEngine(af::randomEngineType::AF_RANDOM_ENGINE_PHILOX,randint());
    //main tensor with all NNs
    NeuralTensor * tensor;
    //the error of the NN/NNs
    af::array error;
    //derivative error
    af::array dError;
    //Z for gradient decent
    vector<af::array> Z;
    //the finished or most optimal NN
    NeuralTensor * optimal{};
    //momentum tensor terms
    NeuralTensor * mtensor;
    //sets neural net structure
    NetSpecs netstructure = NetSpecs(0);
    //the output of each layer, necessary for gradient decent
    vector<af::array> layerinput;
    //sets train input and output data
    vector<af::array> trainIn;
    vector<af::array> trainOut;


    static int randint()
    {
        std::random_device r;
        std::uniform_int_distribution<int> dist;
        return dist(r);
    }

    //converts a 2d vector to an af tensor within a vector
    static vector<af::array> datatogpu(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        return x;
    }


    //converts a 2d vector to an af tensor
    static af::array matrixtogpu(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        af::array y(in.size(),x[0].dims(0));
        for (int i = 0; i < x.size();i++)
        {
            y(i,span) = x[i];
        }
        return y;
    }

    //applies the activation function to all of a tensor then returns the result
    static af::array act(const af::array& x, string func,bool der = false)
    {
        //output tensor
        af::array y;
        //checks which function to use
        if (func == "sig")
        {
            y = af::sigmoid(x);
            if(der)
            {
                return y * (1-y);
            }
            else
            {
                return y;
            }
        }
        else if(func == "relu") {
            y = af::max(0,x);
            if(der)
            {
                cout << "not ready yet"<<endl;
            }
            else
            {
                return y;
            }
        }
        else {
            if(der)
            {
                y = x;
                y = 1;
                return y;
            }
            return x;
        }
        return af::array();
    }

public:
    //Properties
    static Optimizer optimizers;
    static ActFuncs activations;
    //sets input and output test data
    vector<af::array> testIn;
    vector<af::array> testOut;

    //constructors
    explicit CortexNN(const NetSpecs& NetStructure, int initrange = 1, int initpopcount = 1)
    {
        //netstructure is the structure and composition of the neural network
        netstructure = NetStructure;
        //the population size is first set to the smallest size it can be when inicialized
        popN = initpopcount;
        //Here each layer is initialized thus initializing the neural tensor
        tensor = new NeuralTensor(netstructure,rengine,popN,initrange);
        //makes a 0 value clone of the NN tensor in order to hold momentum terms
        mtensor = new NeuralTensor(netstructure,rengine,popN,0);
        //holds the individual errors of all NNs
        error = af::array(1,1,1,popN);
    }
    CortexNN()
    {

    }


    //exports an NN model
    vector<af::array> Export(int num = 0, bool print = true)
    {
        if (print) {
            for (auto & i : tensor->tensor)
            {
                af_print(i(span, span, 0, num));
            }
        }
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor->tensor)
        {
            output.push_back(i(span, span, 0, num));
        }
        return output;
    }

    //imports an NN model
    void import(const vector<vector<vector<float>>>& in)
    {
        vector<af::array> newnet;
        newnet.reserve(in.size());
        for (auto & i : in)
        {
            newnet.push_back(matrixtogpu(i));
        }
        int i;
        //might need to change the u > 0
        for (int u = netstructure.laynum; u > 0; u--)
        {
            i = netstructure.laynum - u - 1;
            tensor->tensor[i](span, span, 0, span) = newnet[i];
        }

    }


    //trains by use of evolution based modeling
    //int initPopN is currently not in use
    void evolveD(int initPopN,float maxmute, float topFrac,int generations,bool debug = false, bool batched = false, int batchsize = 1)
    {
        //checks if set population size is under 4, if so stops training and states the issue
        if (popN<4)
        {
            cout << "Will not run due to population count under 4" << endl;
        }
        else
        {
            //the batchsize is used to determin if the model is trained with everysingle example for each round
            // or one example per round

            //if not batched the population of NNs is trained with a batch size equal to the total number of training examples
            if (!batched)
            {
                traindarwinian(maxmute,topFrac,generations,debug,trainIn.size());
            }
            //if batched the population of NNs is trained with a specific batch size
            else
            {
                traindarwinian(maxmute,topFrac,generations,debug,batchsize);
            }

        }
    }

    //trains by use of traditional gradient decent and back propagation
    void evolveG(int epochs,float learnrate,float momentum,bool debug = false,int batchnum = 1)
    {

        //dimensions of the Z tensor containing Z values
        vector<af::dim4> Zdims;
        Zdims.reserve(netstructure.laynum);
        for (int i = 0; i < netstructure.laynum; i++)
        {
            //sets Zdims to the output of each layer of the NN
            Zdims.emplace_back(netstructure.layers[i].Ncount,1,1,1);
        }
        //sets Z to the Z dims
        for (auto & i : Zdims)
        {
            Z.emplace_back(i);
        }
        //sets output to the Z dims
        layerinput.emplace_back(trainIn[0].dims(0));
        for (auto & i : Zdims)
        {
            layerinput.emplace_back(i);
        }
        //sets the dError to the correct dims for getting the derivative of errors
        dError = af::array(netstructure.outs,1,1,popN);
        //trains the model with gradient decent and back propagation
        trainnewtonian(epochs,learnrate,momentum,debug,batchnum);
    }

    //trains by use of mutiple gradient trained NNs and determining the best structure though
    void evolveDG()
    {

    }

    //calculates accuracy of model
    float accuracyClass()
    {
        //output tensor
        af::array out;
        //error small tensor
        af::array errorsmall;
        //sum of error
        af::array sum;
        float a = 0;
        for (int i = 0; i < testIn.size(); i++)
        {
            //computes model output
            out = compute(testIn[i],false);
            //gets difference
            errorsmall = out - testOut[i];
            //gets absolute of error
            errorsmall = af::abs(errorsmall);
            //sums error
            sum = af::sum(errorsmall);
            //adds the proportion value of correct output to a
            a += 1-((sum.scalar<float>())/testOut[i].dims(0));
        }
        //divides by number of test cases to get an average
        a /= testIn.size();
        return a;
    }

    //installs seqencial data
    /*
    void installSeqTrainData(vector<vector<vector<float>>> trainin, vector<vector<vector<float>>> trainout)
    {
        trainIn = datatogpu(move(trainin));
        trainOut = datatogpu(move(trainout));
    }*/

    //installs all data
    void installData(vector<vector<float>> trainin, vector<vector<float>> trainout, vector<vector<float>> testin, vector<vector<float>> testout)
    {
        trainIn = datatogpu(move(trainin));
        trainOut = datatogpu(move(trainout));
        testIn = datatogpu(move(testin));
        testOut = datatogpu(move(testout));
    }

    //installs test data
    void installtestdata(vector<vector<float>> testin, vector<vector<float>> testout)
    {
        testIn = datatogpu(move(testin));
        testOut = datatogpu(move(testout));
    }

    //prints the best current error
    void optimalError()
    {
        af_print(error(0));
    }

    //runs the model through the input data to produce an output tensor
    af::array compute(const af::array& data,bool computeall = true)
    {
        //is the input turned output of the layers
        af::array intraout;
        //is a tensor used to format the intraout
        af::array base;
        //tiles the intraout tensor to the number of NNs in use
        intraout = tile(data,1,1,1,popN);
        //modifier is the size the base needs to be to format the intraout
        int modifier;
        for (int u = 0; u < netstructure.laynum; u++)
        {
            //sets the modifier equal to the difference between the
            modifier = int(tensor->tensor[u].dims(1) - intraout.dims(0));
            if (modifier > 0)
            {
                //if the modifer is greater than 0 the base is created, equal to 1, and appended to the  intraout tensor.
                base = af::array(1,1,1,popN);
                //base(modifier-1,0,0,span) = 1;
                base = 1.0;
                intraout = af::join(0,intraout,base);
            }
            //intraout becomes the porduct of the matrix multiplication between the preious
            // intraout,the input, and the layer tensor
            //af_print(tensor->tensor[u]);
            //af_print(intraout);
            intraout = matmul(tensor->tensor[u], intraout);
            if (!computeall)
            {
                Z[u] += intraout;
            }
            //applies the activation function to the output making the result the new output/intraout
            intraout = act(intraout,netstructure.layers[u].func);
            if (!computeall)
            {
                layerinput[1+u] += intraout;
            }

        }
        return intraout(seq(netstructure.outs),span,span,span);
    }

    // runs/computes the model once to get output then prints
    af::array run(int num)
    {
        af::array x = compute(testIn[num], false);
        af_print(x);
        return x;
    }

    //computes the error or d error of the model/s
    void computeError(bool gradient, int batchsize = 0, int batch = 0)
    {
        //sets the error tensor to 0 for all NNs
        error = 0;
        dError = 0;
        //sets the layerinputs to 0
        for (auto & i : layerinput)
        {
            i = 0;
        }
        for (auto & i : Z)
        {
            i = 0;
        }
        //the true output values
        af::array correct;
        af::array sum;
        af::array out;
        //the shift is the value needed to reach the batch set for this particular batch set
        int shift =  batchsize * (batch);
        //loops through the batch's set of examples
        //if batch size = 1 then it will loop through each batch normally
        for (int i = 0; i < batchsize; i++)
        {
            //gets output from the computation
            out = compute(trainIn[i+shift],!gradient);
            //takes correct output and tiles it to the size of the output from all the NNs
            correct = tile(trainOut[i+shift],1,1,1,popN);
            //sum is the difference between the correct output and all the provided outputs
            //added negative for gradint decent
            sum = -(correct - out);
            //if this is not a calc/gradient based training func then the output differences are squared
            if(gradient)
            {
                //if it is a gradient based training the non squared difference is summed and preserved in dError
                dError += sum * 2;
                layerinput[0] += trainIn[i+shift];
            }
            //
            sum = pow(sum,2);
            //here the sum is summed along the 0 dimension
            //af_print(sum);
            sum = af::sum(sum,0);
            //here the sum is added to the error
            //af_print(error);
            //af_print(sum);
            error += sum;
        }
        //the error is multiplied by 2
        error = error * 2;
        //then divided by the numbers of examples it was created with
        error = error / batchsize;
    }


    // genetic algo specific
    //performs darwinian training of population to reach an optimal model
    void traindarwinian(float maxmute, float topFrac,int generations,bool debug = false, int batchsize = 1)
    {
        //examples at a time rather than everything at once per epoch
        //loops through consequtive sets of examples
        for (int o = 0; o < generations; o++)
        {

            for (int i = 0; i < int(trainIn.size()/batchsize); i++)
            {
                //calculates the error of each NN of that spesific batch
                computeError(false, batchsize, i);
                //selects and keeps the best fit NNs and removes the rest
                selection(topFrac);
                //prints the debug info if wanted
                if(debug){cout << "gen: " << (o+1) << " batch: " << (i+1) << endl; optimalError();}
                //clones the remaining NNs to form a new population
                reproduce(topFrac);
                //mutates the new population
                mutate(maxmute);
                //sets the current optimal to previous gen optimal
                tensor->setnetequal(optimal,0);
            }
        }
    }

    //mutates population of NNs
    void mutate(float maxmute)
    {
        //sets a new random seed to the random engine
        rengine.setSeed(randint());
        //mutagen tensor that contains random floats to be added to the tensor
        auto * mutagen = new NeuralTensor(netstructure,rengine,popN,maxmute);
        //sets the mutagen term for the most optimal NN to 0 to preserve it
        mutagen->setnetequal(0.0,0);
        //adds the mutagen to the tensor and all NNs
        tensor->add(mutagen);
        //sets the optimal NN to the 0st position in the tensor
        tensor->setnetequal(optimal,0);
    }

    //performs preservation of high performance NNs while discarding others
    void selection(float topFrac)
    {
        //indexing
        //creates an index array which is just some consecutive integers (1,2,3...)
        af::array index = af::array(1,1,1,popN);
        gfor(seq i, popN)
        {
            //sets the index array to consecutive integers (1,2,3...)
            index(span,span,span,i) = i;
        }
        //is an array with errors of the strongest/(best fit) of the NNs
        af::array strong;
        //sets the strong array to integers representing the strongest/(best fit) of the NNs
        //sorts the index array to an order representing the best fit NN to the least fit NN
        af::sort(strong,index, error,3, true);
        //sets the error to the str
        error = strong;
        //final selection
        //is the number of NNs that will be used to create the next generation
        int topnum = topFrac * popN;
        //takes the topnum of the index and removes the rest
        index = index(span,span,span,seq(topnum));
        //formats index to the size required
        index = moddims(index,topnum);
        //sets tensor to the best fit NNs only as defined by their errors and the top fraction
        tensor->tensor = tensor->extractNN(index);
        //sets the current gen's best fit NN to be preserved to the next gen
        optimal = new NeuralTensor(tensor->extractNN());
    }

    //perfroms replication of current NNs in population
    void reproduce(float topfrac)
    {
        //reproduce simply makes 1/topfrac copies of every NN in the tensor
        for (auto & i : tensor->tensor)
        {
            //looping through each layer to copy each then uses the tile function to copy them
            i = tile(i, 1, 1, 1, int(1 / topfrac));
        }
    }


    // Gradient Decent Specific
    //performs gradient decent training
    void trainnewtonian(int epochs,float learnrate,float momentum,bool debug = false,int batchnum = 1)
    {
        for (int i = 0; i < epochs; i++)
        {
            for(int o=0; o < trainIn.size(); o+=batchnum)
            {
                //computes the errors of the model
                computeError(true, batchnum, o);
                //back propagates to update all values
                backpropUpdate(learnrate,momentum);
                if(debug && o % 100 == 0){cout << "epoch: " << i << "   accuracy: "<< accuracyClass() << endl;}
            }
            //debugging
            //prints debug info
        }
    }

    //performs back propagating updates
    void backpropUpdate(float learnrate,float momentum)
    {
        //intraupdate passes the partial derivative from the last layer to the first
        af::array intraupdate;
        //dFunc is the output of the derivative of the activation functions
        af::array dFunc;
        //update term to NN values
        af::array updateTerm;

        //loops from the 1st to last layer to the first of the NN
        for (int o = tensor->layercount-1; o >= 0; o--)
        {
            //dFunc is the output of the derivative of the activation functions
            dFunc = af::array(netstructure.layers[o].Ncount);
            //update term to NN values
            updateTerm = af::array(netstructure.layers[o].dims);
            //dFunc gains the value of the fractional derivative of the activation function
            //Z debug layer input
            dFunc = act(Z[o],netstructure.layers[o].func,true);
           // af_print(dError);
            //af_print(dFunc);
            if (o != tensor->layercount-1)
            {
                //dFunc is multiplied with the fractional derivative of the output
                dFunc = intraupdate * dFunc;
            }
            else
            {

                dFunc = dError * dFunc;
            }
            //af_print(dFunc);
            //sets updateTerm to the -learnrate
            updateTerm = -learnrate;
            //af_print(updateTerm);
            //multiplies all inputs of all weights and biases of the NN by the act function derivative according to the N count
            gfor(seq i,netstructure.layers[o].Ncount)
            {
                updateTerm(i,span) = updateTerm(i,span) * dFunc(i,span);
            }
            //af_print(updateTerm);
            //multiples only the weights of the NN by the output of the previous neurons
            gfor(seq i,netstructure.layers[o].incount)
            {
                updateTerm(span,i) = updateTerm(span,i) * layerinput[o](i,span);
            }
            intraupdate = af::array(netstructure.layers[o].incount);
            //placed the intraupdates here and below because they work best like this
            if(o == tensor->layercount-1)
            {
                //creates the cost partial derivative for the next neural layer
                gfor(seq i,netstructure.layers[o].incount)
                {
                    intraupdate(i) = layerinput[o](i,span) * af::sum((tensor->tensor[o](span,i)*dFunc));
                }
            }
            //updates the momentum tensor
            mtensor->tensor[o] = (mtensor->tensor[o] * momentum) + updateTerm;
            //updates the NN tensor
            tensor->tensor[o] += mtensor->tensor[o];
            //af_print(mtensor->tensor[o]);
            if(o != tensor->layercount-1)
            {
                //creates the cost partial derivative for the next neural layer
                gfor(seq i,netstructure.layers[o].incount)
                {
                    intraupdate(i) = layerinput[o](i,span) * af::sum((tensor->tensor[o](span,i)*dFunc));
                }
            }


        }
    }
};
/*
//class for Recurrent NNs
class CortexRNN
{
private:
    //initializing
    int popN;//number of NNs
    //random engine
    af::randomEngine rengine = af::randomEngine(af::randomEngineType::AF_RANDOM_ENGINE_PHILOX,randint());
    //main tensor with all NNs
    NeuralTensor * tensor;
    //the error of the NN/NNs
    af::array error;
    //derivative error
    af::array dError;
    //Z for gradient decent
    vector<af::array> Z;
    //the finished or most optimal NN
    NeuralTensor * optimal{};
    //momentum tensor terms
    NeuralTensor * mtensor;
    //
    //sets neural net structure
    NetSpecs netstructure = NetSpecs(0);
    //the output of each layer, necessary for gradient decent
    vector<af::array> layerinput;
    //sets train input and output data
    vector<af::array> trainIn;
    vector<af::array> trainOut;


    static int randint()
    {
        std::random_device r;
        std::uniform_int_distribution<int> dist;
        return dist(r);
    }

    //converts a 2d vector to an af tensor within a vector
    static vector<af::array> datatogpu(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        return x;
    }
    //converts a 2d vector to an af tensor
    static af::array matrixtogpu(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        af::array y(in.size(),x[0].dims(0));
        for (int i = 0; i < x.size();i++)
        {
            y(i,span) = x[i];
        }
        return y;
    }

    //applies the activation function to all of a tensor then returns the result
    static af::array act(const af::array& x, string func,bool der = false)
    {
        //output tensor
        af::array y;
        //checks which function to use
        if (func == "sig")
        {
            y = af::sigmoid(x);
            if(der)
            {
                return y * (1-y);
            }
            else
            {
                return y;
            }
        }
        else if(func == "relu") {
            y = af::max(0,x);
            if(der)
            {
                cout << "not ready yet"<<endl;
            }
            else
            {
                return y;
            }
        }
        else {
            if(der)
            {
                y = x;
                y = 1;
                return y;
            }
            return x;
        }
        return af::array();
    }

public:
    //Properties
    static Optimizer optimizers;
    static ActFuncs activations;
    //sets input and output test data
    vector<af::array> testIn;
    vector<af::array> testOut;

    //constructors
    explicit CortexRNN(const NetSpecs& NetStructure, int initrange = 1)
    {
        //netstructure is the structure and composition of the neural network
        netstructure = NetStructure;
        //the population size is first set to the smallest size it can be when inicialized
        popN = 1;
        //Here each layer is initialized thus initializing the neural tensor
        tensor = new NeuralTensor(netstructure,rengine,popN,initrange, true);
        //makes a 0 value clone of the NN tensor in order to hold momentum terms
        mtensor = new NeuralTensor(netstructure,rengine,popN,0,true);
        //holds the individual errors of all NNs
        error = af::array(1,1,1,popN);
    }
    CortexRNN()
    {

    }


    //exports an NN model
    vector<af::array> Export(int num = 0, bool print = true)
    {
        if (print) {
            for (auto & i : tensor->tensor)
            {
                af_print(i(span, span, 0, num));
            }
        }
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor->tensor)
        {
            output.push_back(i(span, span, 0, num));
        }
        return output;
    }

    //imports an NN model
    void import(const vector<vector<vector<float>>>& in)
    {
        vector<af::array> newnet;
        newnet.reserve(in.size());
        for (auto & i : in)
        {
            newnet.push_back(matrixtogpu(i));
        }
        int i;
        //might need to change the u > 0
        for (int u = netstructure.laynum; u > 0; u--)
        {
            i = netstructure.laynum - u - 1;
            tensor->tensor[i](span, span, 0, span) = newnet[i];
        }

    }


    //trains by use of evolution based modeling
    void evolveD(float initPopN,float maxmute, float topFrac,int generations,bool debug = false, bool batched = false, int batchsize = 1)
    {
        //checks if set population size is under 4, if so stops training and states the issue
        if (popN<4)
        {
            cout << "Will not run due to population count under 4" << endl;
        }
        else
        {
            //the batchsize is used to determin if the model is trained with everysingle example for each round
            // or one example per round

            //if not batched the population of NNs is trained with a batch size equal to the total number of training examples
            if (!batched)
            {
                traindarwinian(maxmute,topFrac,generations,debug,trainIn.size());
            }
                //if batched the population of NNs is trained with a specific batch size
            else
            {
                traindarwinian(maxmute,topFrac,generations,debug,batchsize);
            }

        }
    }

    //calculates accuracy of model
    float accuracyClass()
    {
        //output tensor
        af::array out;
        //error small tensor
        af::array errorsmall;
        //sum of error
        af::array sum;
        float a = 0;
        for (int i = 0; i < testIn.size(); i++)
        {
            //computes model output
            out = compute(testIn[i],false);
            //gets difference
            errorsmall = out - testOut[i];
            //gets absolute of error
            errorsmall = af::abs(errorsmall);
            //sums error
            sum = af::sum(errorsmall);
            //adds the proportion value of correct output to a
            a += 1-((sum.scalar<float>())/testOut[i].dims(0));
        }
        //divides by number of test cases to get an average
        a /= testIn.size();
        return a;
    }

    //installs seqencial data
    /*
    void installSeqTrainData(vector<vector<vector<float>>> trainin, vector<vector<vector<float>>> trainout)
    {
        trainIn = datatogpu(move(trainin));
        trainOut = datatogpu(move(trainout));
    }

    //installs all data
    void installData(vector<vector<float>> trainin, vector<vector<float>> trainout)
    {
        trainIn = datatogpu(move(trainin));
        trainOut = datatogpu(move(trainout));
    }

    //installs test data
    void installtestdata(vector<vector<float>> testin, vector<vector<float>> testout)
    {
        testIn = datatogpu(move(testin));
        testOut = datatogpu(move(testout));
    }

    //prints the best current error
    void optimalError()
    {
        af_print(error(0));
    }

    //runs the model through the input data to produce an output tensor
    af::array compute(const af::array data,bool computeall = true)
    {
        //is the input turned output of the layers
        af::array intraout;
        //is a tensor used to format the intraout
        af::array base;
        //tiles the intraout tensor to the number of NNs in use
        intraout = tile(data,1,1,1,popN);
        //modifier is the size the base needs to be to format the intraout
        int modifier;
        for (int u = 0; u < netstructure.laynum; u++)
        {
            //sets the modifier equal to the difference between the
            modifier = int(tensor->tensor[u].dims(1) - intraout.dims(0));
            if (modifier > 0)
            {
                //if the modifer is greater than 0 the base is created, equal to 1, and appended to the  intraout tensor.
                base = af::array(1,1,1,popN);
                //base(modifier-1,0,0,span) = 1;
                base = 1.0;
                intraout = af::join(0,intraout,base);
            }
            //intraout becomes the porduct of the matrix multiplication between the preious
            // intraout,the input, and the layer tensor
            intraout = matmul(tensor->tensor[u], intraout);
            if (!computeall)
            {
                Z[u] += intraout;
            }
            //applies the activation function to the output making the result the new output/intraout
            intraout = act(intraout,netstructure.layers[u].func);
            if (!computeall)
            {
                layerinput[1+u] += intraout;
            }

        }
        return intraout(seq(netstructure.outs),span,span,span);
    }
    //computes sequential data
    af::array seqCompute(const af::array data,af::dim4 outputdim,bool computeall = true)
    {
        af::array output = af::array(outputdim);
        af::array sequnits = data.dims(1);
        af::array interoutput = af::array(netstructure.outs);
        for (int i = 0; i < sequnits.scalar<int>();i++)
        {

        }
        return output;
    }

    // runs/computes the model once to get output then prints
    af::array run(int num)
    {
        af::array x = compute(testIn[num], false);
        af_print(x);
        return x;
    }

    //computes the error or d error of the model/s
    void computeError(bool gradient, int batchsize = 0, int batch = 0)
    {
        //sets the error tensor to 0 for all NNs
        error = 0;
        dError = 0;
        //sets the layerinputs to 0
        for (auto & i : layerinput)
        {
            i = 0;
        }
        for (auto & i : Z)
        {
            i = 0;
        }
        //the true output values
        af::array correct;
        af::array sum;
        af::array out;
        //the shift is the value needed to reach the batch set for this particular batch set
        int shift =  batchsize * (batch);
        //loops through the batch's set of examples
        //if batch size = 1 then it will loop through each batch normally
        for (int i = 0; i < batchsize; i++)
        {
            //gets output from the computation
            out = compute(trainIn[i+shift],!gradient);
            //takes correct output and tiles it to the size of the output from all the NNs
            correct = tile(trainOut[i+shift],1,1,1,popN);
            //sum is the difference between the correct output and all the provided outputs
            //added negative for gradint decent
            sum = -(correct - out);
            //if this is not a calc/gradient based training func then the output differences are squared
            if(gradient)
            {
                //if it is a gradient based training the non squared difference is summed and preserved in dError
                dError += sum * 2;
                layerinput[0] += trainIn[i+shift];
            }
            //
            sum = pow(sum,2);
            //here the sum is summed along the 0 dimension
            sum = af::sum(sum,0);
            //here the sum is added to the error
            error += sum;
        }
        //the error is multiplied by 2
        error = error * 2;
        //then divided by the numbers of examples it was created with
        error = error / batchsize;
    }


    // genetic algo specific
    //performs darwinian training of population to reach an optimal model
    void traindarwinian(float maxmute, float topFrac,int generations,bool debug = false, int batchsize = 1)
    {
        //examples at a time rather than everything at once per epoch
        //loops through consequtive sets of examples
        for (int i = 0; i < int(trainIn.size()/batchsize); i++)
        {

            for (int o = 0; o < generations; o++)
            {
                //calculates the error of each NN of that spesific batch
                computeError(false, batchsize, i);
                //selects and keeps the best fit NNs and removes the rest
                selection(topFrac);
                //prints the debug info if wanted
                if(debug){cout << "gen: " << (o+1) << " batch: " << (i+1) << endl; optimalError();}
                //clones the remaining NNs to form a new population
                reproduce(topFrac);
                //mutates the new population
                mutate(maxmute);
                //sets the previous optimal to current gen optimal
                tensor->setnetequal(optimal,0);
            }
        }
    }

    //mutates population of NNs
    void mutate(float maxmute)
    {
        //sets a new random seed to the random engine
        rengine.setSeed(randint());
        //mutagen tensor that contains random floats to be added to the tensor
        auto * mutagen = new NeuralTensor(netstructure,rengine,popN,maxmute);
        //sets the mutagen term for the most optimal NN to 0 to preserve it
        mutagen->setnetequal(0.0,0);
        //adds the mutagen to the tensor and all NNs
        tensor->add(mutagen);
        //sets the optimal NN to the 0st position in the tensor
        tensor->setnetequal(optimal,0);
    }

    //performs preservation of high performance NNs while discarding others
    void selection(float topFrac)
    {
        //indexing
        //creates an index array which is just some consecutive integers (1,2,3...)
        af::array index = af::array(1,1,1,popN);
        gfor(seq i, popN)
        {
            //sets the index array to consecutive integers (1,2,3...)
            index(span,span,span,i) = i;
        }
        //is an array with errors of the strongest/(best fit) of the NNs
        af::array strong;
        //sets the strong array to integers representing the strongest/(best fit) of the NNs
        //sorts the index array to an order representing the best fit NN to the least fit NN
        af::sort(strong,index, error,3, true);
        //sets the error to the str
        error = strong;
        //final selection
        //is the number of NNs that will be used to create the next generation
        int topnum = topFrac * popN;
        //takes the topnum of the index and removes the rest
        index = index(span,span,span,seq(topnum));
        //formats index to the size required
        index = moddims(index,topnum);
        //sets tensor to the best fit NNs only as defined by their errors and the top fraction
        tensor->tensor = tensor->extractNN(index);
        //sets the current gen's best fit NN to be preserved to the next gen
        optimal = new NeuralTensor(tensor->extractNN());
    }

    //perfroms replication of current NNs in population
    void reproduce(float topfrac)
    {
        //reproduce simply makes 1/topfrac copies of every NN in the tensor
        for (auto & i : tensor->tensor)
        {
            //looping through each layer to copy each then uses the tile function to copy them
            i = tile(i, 1, 1, 1, int(1 / topfrac));
        }
    }
};

class CortexRL
{
private:
    //initializing
    int popN;//number of NNs
    //random engine
    af::randomEngine rengine = af::randomEngine(af::randomEngineType::AF_RANDOM_ENGINE_PHILOX,randint());
    //main tensor with all NNs
    NeuralTensor * tensor;
    //the error of the NN/NNs
    af::array error;
    //derivative error
    af::array dError;
    //Z for gradient decent
    vector<af::array> Z;
    //the finished or most optimal NN
    NeuralTensor * optimal{};
    //momentum tensor terms
    NeuralTensor * mtensor;
    //sets neural net structure
    NetSpecs netstructure = NetSpecs(0);
    //the output of each layer, necessary for gradient decent
    vector<af::array> layerinput;
    //sets train input and output data
    vector<vector<af::array>> trainIn;
    vector<vector<af::array>> trainOut;

    static int randint()
    {
        std::random_device r;
        std::uniform_int_distribution<int> dist;
        return dist(r);
    }


};
*/