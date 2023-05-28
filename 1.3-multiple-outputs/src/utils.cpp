#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "NvInfer.h"


using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))){
        cout << "File " << fileName << " doesn't exists" << endl;
        return false;
    }else{
        return true;
    }
}

bool fileRead(const string &path, vector<unsigned char> &data, size_t &size){
    stringstream trtModelStream;
    ifstream cache(path);

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();

    vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    tmp.resize(size);

    /* 将trtModelStream中的stream通过read函数写入modelMem中*/
    trtModelStream.read((char*)data[0], size);
    return true;
}

vector<unsigned char> loadFile(const string &file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

string printDims(const nvinfer1::Dims dims){
    int n = 0;
    char buff[100];
    string result;

    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1) {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float* tensor, int size){
    int n = 0;
    char buff[100];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
        if (i != size - 1){
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}