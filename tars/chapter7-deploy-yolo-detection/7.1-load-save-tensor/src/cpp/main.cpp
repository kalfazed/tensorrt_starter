#include "trt_logger.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*
        从c++读取一个python下保存的npz file
    */
    // cnpy::npz_t npz_data = cnpy::npz_load("data/data_python.npz");
    // cnpy::NpyArray arr   = npz_data["data_python"];
    // for (int i = 0; i < arr.shape.size(); i ++){
    //     LOG("arr.shape[%d]: %d", i, arr.shape[i]);
    // }
    // LOG("Succeeded loading data from .npy/.npz!");
    // LOG("Tensor values:");
    // printTensorNPY(arr);


    /*
        在c++下保存一个python可以识别的npy/npz file
    */
    const int b = 3;
    const int c = 2;
    const int h = 4;
    const int w = 4;
    int size = b * c * h * w;

    float* data = (float*)malloc(size * sizeof(float));
    initTensor(data, size, 0, 1, 0);

    cnpy::npz_save("data/data_cpp.npz", "data_cpp", &data[0], {b, c, h, w}, "w");
    cnpy::npy_save("data/data_cpp.npy", &data[0], {b, c, h, w}, "w");

    LOG("Succeeded saving data as .npy/.npz!");
    LOG("Tensor values:");
    printTensorCXX(data, b, c, h, w);

    return 0;
}
