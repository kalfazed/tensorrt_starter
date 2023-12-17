#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

const int THREADS_FOR_VOXEL = 256;

#include <cuda_runtime.h>

cudaError_t voxelizationLaunch(
    const float *points, size_t points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int feature_num,
	int max_voxels, int max_points_voxel,
    unsigned int *hash_table,
	unsigned int *num_points_per_voxel, float *voxel_features,
	unsigned int *voxel_indices, unsigned int *real_voxel_num,
    cudaStream_t stream = 0);

cudaError_t featureExtractionLaunch(
    float *voxels_temp_,
	unsigned int *num_points_per_voxel,
    const unsigned int real_voxel_num, int max_points_per_voxel,
	int feature_num, float *voxel_features, cudaStream_t stream_ = 0);

#endif // __KERNEL_HPP__
