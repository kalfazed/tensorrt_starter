#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "timer.hpp"
#include <cmath>
#include <vector>

const unsigned int MAX_POINTS_NUM = 300000;

namespace preprocess{

/*
 * 针对NuScenes的点云的params。以ego为原点的坐标系
 *  1. x在(-45, 45)
 *  2. y在(-45, 45)
 *  3. z在(-5, 3)
 * voxel的大小是0.075 * 0.075 * 0.2，
 * 所以我们可以得到voxelization之后的特征图大小是(1200, 1200, 40)
 * 这个也是后续输入到SCN的input的大小
*/
class Params{
  public:
    const float voxel_size[2] = { 0.075, 0.075, };
    const float pc_range[2]   = { -54, -54, };

    const float min_x_range = -54;
    const float max_x_range = 54;
    const float min_y_range = -54;
    const float max_y_range = 54;
    const float min_z_range = -5.0;
    const float max_z_range = 3.0;

    // the size of a pillar
    const float pillar_x_size = 0.075;
    const float pillar_y_size = 0.075;
    const float pillar_z_size = 0.2;

    const int max_points_per_voxel = 10;

    const unsigned int max_voxels = 160000;
    const unsigned int feature_num = 5;

    Params() {};

    int getGridXSize() {
      return (int)std::round((max_x_range - min_x_range) / pillar_x_size);
    }
    int getGridYSize() {
      return (int)std::round((max_y_range - min_y_range) / pillar_y_size);
    }
    int getGridZSize() {
      return (int)std::round((max_z_range - min_z_range) / pillar_z_size);
    }
    
};

class PreprocessPCD {
  private:
    Params       params_;

    unsigned int *point2voxel_offset_;
    unsigned int *hash_table_;
    float        *voxels_temp_;

    unsigned int *d_real_num_voxels_;
    unsigned int *h_real_num_voxels_;
    float        *h_voxel_features_;
    float        *d_voxel_features_;
    unsigned int *d_voxel_num_;
    unsigned int *d_voxel_indices_;

    unsigned int hash_table_size_;
    unsigned int voxels_temp_size_;
    unsigned int voxel_features_size_;
    unsigned int voxel_features_real_size_;
    unsigned int voxel_idxs_size_;
    unsigned int voxel_num_size_;

  public:
    PreprocessPCD();
    ~PreprocessPCD();

    int alloc_resource();
    int generateVoxels(const float *points, size_t points_size, cudaStream_t stream);
    unsigned int getOutput(float** d_voxel_features, unsigned int** d_voxel_indices, std::vector<int>& sparse_shape);
};

}; //namespace preprocess

#endif //__PREPROCESS_HPP__
