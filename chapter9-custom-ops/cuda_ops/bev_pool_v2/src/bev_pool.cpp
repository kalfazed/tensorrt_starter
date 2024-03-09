#include <c10/cuda/CUDAGuard.h>
// #include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../utils/timer.hpp"

// CUDA function declarations
void bev_pool_v2_cuda(
  int c, int n_intervals,
  const float *depth, const float *feat,
  const int *ranks_depth, const int *ranks_feat,
  const int *ranks_bev, const int *interval_starts,
  const int *interval_lengths, float *out);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    out              : output features, FloatTensor[b, c, h_out, w_out]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals] 
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals] 
  Return:
*/
void bev_pool_v2(
  const at::Tensor _depth, 
  const at::Tensor _feat,
  at::Tensor _out, 
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts) 
{
  int c = _feat.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
  const float *depth = _depth.data_ptr<float>();
  const float *feat = _feat.data_ptr<float>();
  const int *ranks_depth = _ranks_depth.data_ptr<int>();
  const int *ranks_feat = _ranks_feat.data_ptr<int>();
  const int *ranks_bev = _ranks_bev.data_ptr<int>();

  const int *interval_lengths = _interval_lengths.data_ptr<int>();
  const int *interval_starts = _interval_starts.data_ptr<int>();

  float *out = _out.data_ptr<float>();

  bev_pool_v2_cuda(
    c, n_intervals, 
    depth, feat, ranks_depth, ranks_feat, ranks_bev,
    interval_starts, interval_lengths, out);
}


/*
  Function: pillar pooling (forward, cuda) in debug mode
*/
void bev_pool_v2_d(
  const at::Tensor _depth, 
  const at::Tensor _feat,
  at::Tensor _out, 
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts) 
{
  Timer timer;
  int c = _feat.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
  const float *depth = _depth.data_ptr<float>();
  const float *feat = _feat.data_ptr<float>();
  const int *ranks_depth = _ranks_depth.data_ptr<int>();
  const int *ranks_feat = _ranks_feat.data_ptr<int>();
  const int *ranks_bev = _ranks_bev.data_ptr<int>();

  const int *interval_lengths = _interval_lengths.data_ptr<int>();
  const int *interval_starts = _interval_starts.data_ptr<int>();

  float *out = _out.data_ptr<float>();

  timer.start_gpu();
  bev_pool_v2_cuda(
    c, n_intervals, 
    depth, feat, ranks_depth, ranks_feat, ranks_bev,
    interval_starts, interval_lengths, out);
  timer.stop_gpu("BEVPoolV2(CUDA)");
  timer.show();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_v2", &bev_pool_v2, "bev_pool_v2_forward");
  m.def("bev_pool_v2_d", &bev_pool_v2_d, "bev_pool_v2_forward(debug mode)");
}
