import numpy as np

def read_pcd(pcd_path):
    lines = []
    num_points = None

    with open(pcd_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        x, y, z, i = list(map(float, line.split()))
        #这里没有把i放进去，也是为了后面 x, y, z 做矩阵变换的时候方面
        #但这个理解我选择保留， 因为可能把i放进去也不影响代码的简易程度
        points.append((np.array([x, y, z, i])))

    return points

def load_pcd_to_array(pcd_path):
    with open(pcd_path, "r") as f:
        while True:
            ln = f.readline().strip()
            if ln.startswidth('DATA'):
                break

            points = np.loadtxt(f)
            return points

def read_nuscene_test():
    pcd_path = "../../data/291e7331922541cea98122b607d24831.bin"
    data = np.frombuffer(open(pcd_path, "rb").read(), dtype=np.float32)
    data = data.reshape(-1, 5)[:, :4]

    print(data)
    print(data.shape)
    data.tofile("../../results/kitti-291e7331922541cea98122b607d24831.bin")

def read_pcd_test():
    pcd_path = "../../data/table_scene_lms400.pcd"
    data = load_pcd_to_array(pcd_path)
    print(data)
    print(data.shape)


if __name__ == "__main__":
    read_nuscene_test()
    # read_pcd_test()
