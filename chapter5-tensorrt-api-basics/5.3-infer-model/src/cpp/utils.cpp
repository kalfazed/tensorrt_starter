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

    trtModelStream.seekg(0, ios::beg);

    /* 将trtModelStream中的stream通过read函数写入modelMem中*/
    trtModelStream.read((char*)&data[0], size);
    return true;
}

/* 常规的使用iostream读取文件的方法，可复用*/
vector<unsigned char> loadFile(const string &file){
    vector<uint8_t> data;

    /* 通过ifstream读取文件，并保存为unsigned char的vector*/
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()) return {};

    /* 设置数据流位置到末尾，并获取文件大小*/
    in.seekg(0, ios::end);
    size_t length = in.tellg();
    if (length <= 0) return {};

    /* 通过ifstream读取文件，并保存为unsigned char的vector*/
    in.seekg(0, ios::beg);
    data.resize(length);
    in.read((char*)&data[0], length); 
    /* 这里可能比较绕，&data[0]的类型是字符指针，指向data这个vector中的第一个元素 */

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

string getEnginePath(string onnxPath){
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");

    int dir_l  = 0;
    int dir_r  = onnxPath.find("/");

    string enginePath;
    enginePath = onnxPath.substr(dir_l, dir_r);
    enginePath += "/engine";
    enginePath += onnxPath.substr(name_l, name_r - name_l);
    enginePath += ".engine";
    return enginePath;
}

