#include "kernel.hpp"
#include <cuda.h>
#include <stdio.h>

__device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

__device__ inline void insertHashTable(
    const uint32_t key, uint32_t *value,
	const uint32_t hash_size, uint32_t *hash_table) 
{
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
     uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
     if (pre_key == empty_key) {
       hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);
       break;
     } else if (pre_key == key) {
       break;
     }
     slot = (slot + 1) % (hash_size / 2);
  }
}

__device__ inline uint32_t lookupHashTable(
    const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) 
{
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (cnt < 100 /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size / 2];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % (hash_size / 2);
    }
  }
  return empty_key;
}

__global__ void buildHashKernel(
    const float *points, size_t points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int feature_num,
	unsigned int *hash_table, unsigned int *real_voxel_num) 
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }
  
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
                            + voxel_idy * grid_x_size
                            + voxel_idx;
  insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);
}


/*
  使用CUDA进行pcd的体素化(voxelization)的加速
  其实我们可以通过voxelization之后把voxelization之后的pcd给保存下来用pcl_viewer打开来看
  这一部分的voxelization其实在PCL中也是有实现的。只不过会相比于这个CUDA版本的Voxelization速度会慢
  这个Voxelization的加速体现在
  - 使用hash table进行快速查找voxel的id以及对应的索引和内存空间
  - 使用CUDA的多线程计算，对大量的点进行并行处理。针对每一个点判断是否给放到voxel里面，以及放到哪一个voxel里面
  voxelization在PCL定位为filter中的downsampling的方式的一种，可以通过VoxelGrid Filter来使用PCL来进行处理
  
  TODO: PCL的VoxelGrid filter与voxelization CUDA kernel的速度对比
*/
__global__ void voxelizationKernel(
    const float *points, size_t points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
    int max_points_per_voxel,
	unsigned int *hash_table, unsigned int *num_points_per_voxel,
	float *voxels_temp, unsigned int *voxel_indices, unsigned int *real_voxel_num) 
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }
  
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  // 通过点的坐标x, y, z找到voxel的坐标x, y, z
  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
                            + voxel_idy * grid_x_size
                            + voxel_idx;

  // 通过hash_table，以voxel_offset为key，找到对应的voxel的id
  unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);
  if (voxel_id >= max_voxels) {
    return;
  }
  
  // 这个voxel里面的点的数量加1，由于有可能其他线程也会对这个voxel进行赋值。所以需要atomicAdd
  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);

  if (current_num < max_points_per_voxel) {
    // src(point cloud)和tar(voxel)上的点的offset
    unsigned int dst_offset = voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num;
    unsigned int src_offset = point_idx * feature_num;

    // 把point cloud里面的5个feature(x, y, z, intensity, ray_id)给放在voxel的数组里面
    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
      voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
    }

    // now only deal with batch_size = 1
    // since not sure what the input format will be if batch size > 1
    uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx};
    ((uint4 *)voxel_indices)[voxel_id] = idx;

  }
}

/*
  使用CUDA进行pcd的体素化(voxelization)的加速
  一个voxel里面有0~10个点，每一个点有5个feature(x, y, z, intensity, beam_id), 所有一个voxel会有0 ~ 50个fp32的数据
  初始化是设定voxel的最大数量是160,000个，如果每一个voxel都有这么多数据的话，存储空间会很大。所以需要进行压缩。
  这个kernel是对voxel的feature进行压缩的实现加速，主要负责
  - 通过两个for循环将所有的10个点的各个feature进行求均值，作为这个voxel的feature来保存
  - 将所有fp32的feature转换为fp16进行保存

  以这个项目中的bin文件为例，通过voxelization和特征压缩以后，
  239,991 points ->  85,179 points
  267,057 points -> 106,004 points

  可以大幅度的减少SCN和后续网络的input feature map
  建议尝试把这个阶段结束后的pcd可视化一下，看看效果
*/
__global__ void featureExtractionKernel(
    float *voxels_temp,
    unsigned int *num_points_per_voxel,
    int max_points_per_voxel, int feature_num, float *voxel_features) 
{
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel ?
	                                          max_points_per_voxel :  num_points_per_voxel[voxel_idx];
  int valid_points_num = num_points_per_voxel[voxel_idx];
  int offset = voxel_idx * max_points_per_voxel * feature_num;

  // 求voxel里面的5个feature的均值
  for (int feature_idx = 0; feature_idx< feature_num; ++feature_idx) {
    for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
      voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
    }
    voxels_temp[offset + feature_idx] /= valid_points_num;
  }

  // 为了让存储连续，将这几个值重新放置在新的空间, 作为后续的SCN的输入
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    int dst_offset = voxel_idx * feature_num;
    int src_offset = voxel_idx * feature_num * max_points_per_voxel;
    voxel_features[dst_offset + feature_idx] = voxels_temp[src_offset + feature_idx];
  }
}

cudaError_t featureExtractionLaunch(
    float *voxels_temp, unsigned int *num_points_per_voxel,
    const unsigned int real_voxel_num, int max_points_per_voxel, int feature_num,
	float *voxel_features, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((real_voxel_num + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  featureExtractionKernel<<<blocks, threads, 0, stream>>>
    (voxels_temp, num_points_per_voxel,
        max_points_per_voxel, feature_num, voxel_features);
  cudaError_t err = cudaGetLastError();
  return err;
}

/*
  这里面调用两个kernel
    1. 创建hash table的kernel
    2. 进行voxelization的kernel

    由于点云中临近的几个点会被放在同一个voxel中，那么在进行voxel的时候，寻找哪一个点对应哪一个voxel会比较耗时
    但是如果我们可以通过以hash table的方式，将这个voxel的offset和这个voxel的数据给保存起来，
    那么寻找这个voxel的过程就可以通过look up table(LUT)的方式找到这个voxel，会比较快。

    创建完voxel的offset的hash table以后，在进行voxelization的时候，直接通过LUT的方式寻找索引就好了
*/

cudaError_t voxelizationLaunch(
    const float *points, size_t points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
	int max_points_per_voxel,
	unsigned int *hash_table, unsigned int *num_points_per_voxel,
	float *voxel_features, unsigned int *voxel_indices,
	unsigned int *real_voxel_num, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((points_size+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  buildHashKernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size,
        grid_y_size, grid_x_size, feature_num, hash_table,
	real_voxel_num);
  voxelizationKernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size,
        grid_y_size, grid_x_size, feature_num, max_voxels,
        max_points_per_voxel, hash_table,
	num_points_per_voxel, voxel_features, voxel_indices, real_voxel_num);
  cudaError_t err = cudaGetLastError();
  return err;
}
