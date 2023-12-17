import numpy as np
import pcl.pcl_visualization
import os
 
currrrent_path = os.path.dirname(os.path.abspath(__file__))
lidar_path = os.path.join(currrrent_path, "../../data/291e7331922541cea98122b607d24831.bin")

points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
 
# 这里对第四列进行赋值，它代表颜色值，根据你自己的需要赋值即可；
points[:, 3] = 3329330
 
# PointCloud_PointXYZRGB 需要点云数据是N*4，分别表示x,y,z,RGB ,其中RGB 用一个整数表示颜色；
color_cloud = pcl.PointCloud_PointXYZRGB(points)
visual = pcl.pcl_visualization.CloudViewing()
visual.ShowColorCloud(color_cloud, b'cloud')
flag = True
while flag:
    flag != visual.WasStopped()
