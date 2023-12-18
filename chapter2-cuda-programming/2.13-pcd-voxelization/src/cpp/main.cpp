#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "timer.hpp"
#include "preprocess.hpp"

using namespace std;

int main(){
    Timer timer;

    string data_path     = "data/291e7331922541cea98122b607d24831.bin";
    string output_prefix = "results/";
    string output_path   = "";

    preprocess::Params params;

    preprocess::PreprocessPCD voxelization;

    voxelization.alloc_resource();
    
    unsigned int length = 0;
    void* data_h = NULL;
    float* data_d = NULL;

    loadData(data_path.c_str(), &data_h, &length);
    size_t origin_points_num = length / (params.feature_num * sizeof(float));

    LOG("Before voxelization, the point counts is %d", origin_points_num);

    CUDA_CHECK(cudaMalloc((void**)&data_d, length));
    cudaMemcpy(data_d, data_h, length, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // voxeliazation的调用
    voxelization.generateVoxels((float*)data_d, origin_points_num, stream);
    
    float* voxel_feature;
    // free(voxel_feature);
    unsigned int* indices;
    vector<int> sparse_shape;

    unsigned int valid_points_num = voxelization.getOutput(&voxel_feature, &indices, sparse_shape);


    // int z_size = params.getGridZSize();
    // int y_size = params.getGridYSize();
    // int x_size = params.getGridXSize();

    // for (int i = 0; i < z_size; i ++){
    //     for (int j = 0; j < y_size; j++) {
    //         for (int k = 0; k < x_size; k ++){
    //             int offset = i * y_size * x_size + j + x_size + k;
    //             double x = voxel_feature[offset];
    //             double y = voxel_feature[offset + 1];
    //             double z = voxel_feature[offset + 2];
    //             double indensity = voxel_feature[offset + 3];
    //             double line = voxel_feature[offset + 4];
    //             
    //             if (indensity != 0)
    //                 LOG("The point clound info: (%3lf, %3lf, %3lf): indensity: %3f", x, y, z, indensity);
    //         }
    //     }
    // }

    free(voxel_feature);


    LOG("After voxelization, the point count is %d", valid_points_num);


    return 0;
}
