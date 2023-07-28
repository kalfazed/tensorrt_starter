我们做实验的话，希望c++和python读取同样的数据
所以我们需要想办法让c++和python之间进行数据的共享

所以，流程上我们想做到:

python的tensor的数据最终可以导出为以下两种格式中其中一个
- npy格式: 一个多维的tensor给转换为一维的float数组，以及各个维度的大小
- npz格式: 多个多维的tensor按照字典的形式转换为多个一维的float数组，以及各个维度的大小

c++可以读取npy或者npz任意一种格式，并给保存到自己的数据类型中
```c++
struct Data {
  int* dims;
  float* data;
} npy

std::map<string, Data> npz
```

实验所需要的Device memory的大小可以根据ICudaEngine中的inputDim的大小和outputDim的大小来定
一般这个是在inference需要的，所以可以在反序列化Engine之后就申请内存
