#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <fstream>

class CSVReader
{
    std::string fileName;
    std::string delimeter;
    vector<vector<float>> data;
public:
    explicit CSVReader(string filename, string delm = ",")
    {
        fileName = move(filename);
        delimeter = move(delm);
    }
    // Function to fetch data from a CSV File
    vector<vector<float>> saveOutput(int column)
    {
        vector<vector<float>> out;
        out.reserve(data.size());
        for (auto & i : data)
        {
            out.push_back(vector<float>{i[column-1]});
        }
        return out;
    }
    vector<vector<float>> saveInput(int c1,int c2)
    {
        vector<float>::const_iterator first;
        vector<float>::const_iterator last;

        vector<vector<float>> in;
        in.reserve(data.size());
        for (auto & i : data)
        {
            first = i.begin() + (c1-1);
            last = i.begin() + c2;
            in.emplace_back(first,last);
        }
        return in;
    }
    void importData()
    {
        ifstream file(fileName);
        vector<vector<float> > dataList;
        string line;
        while (std::getline(file, line))
        {
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            vector<float> v;
            v.reserve(vec.size());
            for (auto & i : vec)
            {
                v.push_back(std::stof(i));
            }
            dataList.push_back(v);
        }
        file.close();

        data = dataList;
    }
};