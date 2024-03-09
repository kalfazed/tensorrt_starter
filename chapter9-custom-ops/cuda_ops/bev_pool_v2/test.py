import torch
import logging
from cuda_ops.bev_pool_v2 import bev_pool_v2
from cuda_ops.utils.logger import Logger


def main():
    logger = Logger()
    logger.setLevel(logging.DEBUG)

    n = 6
    camera_h = int(256 / 8)
    camera_w = int(704 / 8)
    camera_c = 80
    depth_c = 118
    bev_h = 256
    bev_w = 1112
    num_points = 1583950
    num_intervals = 710377

    # initialization
    depth = torch.randn(1, n, depth_c, camera_h, camera_w, device='cuda')  # [1, 6, 118, 32, 88]
    feat = torch.randn(1, n, camera_h, camera_w, camera_c, device='cuda')  # [1, 6, 32, 88, 80]
    ranks_depth = torch.ones(num_points, device='cuda')  # [n, d, c, h, w] 1,583,950
    ranks_feat = torch.ones(num_points, device='cuda')  # 1,583,950
    ranks_bev = torch.ones(num_points, device='cuda')  # 1,583,950
    interval_lengths = torch.ones(num_intervals, device='cuda')  # 710,377
    interval_starts = torch.ones(num_intervals, device='cuda')  # 710,377

    # change data
    depth = depth.contiguous().float()
    feat = feat.contiguous().float()
    ranks_depth = ranks_depth.contiguous().int()
    ranks_feat = ranks_feat.contiguous().int()
    ranks_bev = ranks_bev.int()
    interval_lengths = interval_lengths.contiguous().int()
    interval_starts = interval_starts.contiguous().int()

    # set output
    bev_feat_shape = [1, 1, bev_h, bev_w, camera_c]  # 1, 1, 256, 1112, 80
    out = feat.new_zeros(bev_feat_shape, device='cuda')

    logger.printTensorInformation(out, "Before", n=100)

    # GPU bev_pool_v2
    logger.info("start bev pool")

    bev_pool_v2.bev_pool_v2_d(
        depth,
        feat,
        out,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_lengths,
        interval_starts)

    logger.info("finish bev pool")
    logger.printTensorInformation(out, "GPU", n=100)


if __name__ == '__main__':
    main()
