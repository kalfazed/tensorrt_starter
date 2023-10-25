#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "NvInfer.h"
#include "model.hpp"


using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))){
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
    char buff[10000];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%.4lf", tensor[i]);
        if (i != size - 1){
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float* tensor, int size, int stride){
    int n = 0;
    char buff[10000];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ \n");

    for (int i = 0; i < size / stride; i ++) {
        for (int j = 0; j < stride; j ++){
            n += snprintf(buff + n, sizeof(buff) - n, "%.4lf", tensor[j + i * stride]);
            if (j != stride - 1){
                n += snprintf(buff + n, sizeof(buff) - n, ", ");
            } else {
                n += snprintf(buff + n, sizeof(buff) - n, "\n");
            }
        }

    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float* tensor, int size, int strideH, int strideW){
    int n = 0;
    char buff[10000];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ \n");

    int area = strideW * strideH;

    for (int i = 0; i < size; i += area) {
        for (int j = 0; j < area; j += strideW) {
            for (int k = 0; k < strideW; k++) {
                n += snprintf(buff + n, sizeof(buff) - n, "%.4lf", tensor[i + j + k]);
                if (k != strideW - 1){
                    n += snprintf(buff + n, sizeof(buff) - n, ", ");
                } else {
                    n += snprintf(buff + n, sizeof(buff) - n, "\n");
                }
            }
        }
        n += snprintf(buff + n, sizeof(buff) - n, "\n");
    }

    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float* tensor, int size, nvinfer1::Dims dim) {
    if (dim.nbDims == 2) {
        return printTensor(tensor, size, dim.d[1]);
    } else if (dim.nbDims == 3) {
        return printTensor(tensor, size, dim.d[1], dim.d[2]);
    } else if (dim.nbDims == 4) {
        return printTensor(tensor, size, dim.d[2], dim.d[3]);
    }
}

int getDimSize(nvinfer1::Dims dims) {
    int size = 1;
    for (int j = 0; j < dims.nbDims; j++) {
        size *= dims.d[j];
    }
    return size;
}

string printTensorShape(nvinfer1::ITensor* tensor){
    string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
        str += to_string(dims.d[j]);
        if (j != dims.nbDims - 1) {
            str += " x ";
        }
    }
    str += "]";
    return str;
}

string getEnginePath(string onnxPath, Model::precision prec){
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");

    int dir_l  = 0;
    int dir_r  = onnxPath.find("/");

    string enginePath;

    enginePath = onnxPath.substr(dir_l, dir_r);
    enginePath += "/engine";
    enginePath += onnxPath.substr(name_l, name_r - name_l);
    
    if (prec == Model::precision::FP16) {
        enginePath += "_fp16";
    } else if (prec == Model::precision::INT8) {
        enginePath += "_int8";
    } else {
        enginePath += "_fp32";
    }

    enginePath += ".engine";
    return enginePath;
}

string getFileType(string filePath){
    int pos = filePath.rfind(".");
    string suffix;
    suffix = filePath.substr(pos, filePath.length());
    return suffix;
}

string getPrecision(nvinfer1::DataType type) {
    switch(type) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        default:                          return "unknown";
    }
}
