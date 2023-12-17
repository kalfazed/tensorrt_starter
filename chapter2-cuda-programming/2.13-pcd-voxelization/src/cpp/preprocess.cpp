#include <assert.h>
#include <iostream>
#include "utils.hpp"
#include "preprocess.hpp"
#include "kernel.hpp"

namespace preprocess{

PreprocessPCD::PreprocessPCD()
{}

PreprocessPCD::~PreprocessPCD()
{
    CUDA_CHECK(cudaFree(hash_table_));
    CUDA_CHECK(cudaFree(voxels_temp_));

    CUDA_CHECK(cudaFree(d_voxel_features_));
    CUDA_CHECK(cudaFree(d_voxel_num_));
    CUDA_CHECK(cudaFree(d_voxel_indices_));

    CUDA_CHECK(cudaFree(d_real_num_voxels_));
    CUDA_CHECK(cudaFreeHost(h_real_num_voxels_));
}

unsigned int PreprocessPCD::getOutput(
    float** h_voxel_features, unsigned int** d_voxel_indices, std::vector<int>& sparse_shape)
{
    // *d_voxel_features = d_voxel_features_;
    *d_voxel_indices = d_voxel_indices_;

    CUDA_CHECK(cudaMemcpy(h_voxel_features_, d_voxel_features_, voxel_features_size_, cudaMemcpyDeviceToHost));

    sparse_shape.clear();
    sparse_shape.push_back(params_.getGridZSize() + 1);
    sparse_shape.push_back(params_.getGridYSize());
    sparse_shape.push_back(params_.getGridXSize());

    return *h_real_num_voxels_;
}

int PreprocessPCD::alloc_resource(){
    hash_table_size_ = MAX_POINTS_NUM * 2 * 2 * sizeof(unsigned int);

    voxels_temp_size_ = params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(float);
    voxel_features_size_ = params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(float);
    voxel_features_real_size_ = params_.getGridZSize() * params_.getGridYSize() * params_.getGridXSize() * params_.feature_num;

    CUDA_CHECK(cudaMallocManaged((void **)&hash_table_, hash_table_size_));
    CUDA_CHECK(cudaMallocManaged((void **)&voxels_temp_, voxels_temp_size_));

    voxel_num_size_ = params_.max_voxels * sizeof(unsigned int);
    voxel_idxs_size_ = params_.max_voxels * 4 * sizeof(unsigned int);

    CUDA_CHECK(cudaMallocManaged((void **)&h_voxel_features_, voxel_features_size_));
    CUDA_CHECK(cudaMallocManaged((void **)&d_voxel_features_, voxel_features_size_));
    CUDA_CHECK(cudaMallocManaged((void **)&d_voxel_num_, voxel_num_size_));
    CUDA_CHECK(cudaMallocManaged((void **)&d_voxel_indices_, voxel_idxs_size_));
    CUDA_CHECK(cudaMalloc((void **)&d_real_num_voxels_, sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost((void **)&h_real_num_voxels_, sizeof(unsigned int)));
    
    CUDA_CHECK(cudaMemset(d_voxel_num_, 0, voxel_num_size_));
    CUDA_CHECK(cudaMemset(d_voxel_features_, 0, voxel_features_size_));
    CUDA_CHECK(cudaMemset(d_voxel_indices_, 0, voxel_idxs_size_));
    CUDA_CHECK(cudaMemset(d_real_num_voxels_, 0, sizeof(unsigned int)));

    return 0;
}

int PreprocessPCD::generateVoxels(const float *points, size_t points_size, cudaStream_t stream)
{
    // 清空device上的分配的内存上的数据
    CUDA_CHECK(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, stream));
    CUDA_CHECK(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, stream));

    CUDA_CHECK(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, stream));
    CUDA_CHECK(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // voxelization，将pcd里的数据一一对应到voxel中
    CUDA_CHECK(voxelizationLaunch(
        points, points_size,
        params_.min_x_range, params_.max_x_range,
        params_.min_y_range, params_.max_y_range,
        params_.min_z_range, params_.max_z_range,
        params_.pillar_x_size, params_.pillar_y_size, params_.pillar_z_size,
        params_.getGridYSize(), params_.getGridXSize(), params_.feature_num, params_.max_voxels,
        params_.max_points_per_voxel, hash_table_,
        d_voxel_num_, voxels_temp_, d_voxel_indices_,
        d_real_num_voxels_, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 特征提取, 将一个voxel里的所有的点找到具有代表性的点作为这个voxel的特征点
    CUDA_CHECK(featureExtractionLaunch(
        voxels_temp_, d_voxel_num_,
        *h_real_num_voxels_, params_.max_points_per_voxel, params_.feature_num,
        d_voxel_features_, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}


} // namespace preprocess
