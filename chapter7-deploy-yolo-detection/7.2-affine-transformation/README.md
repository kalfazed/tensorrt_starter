Affine-transformation
===
这个小节是接着Chapter2的``2.10 bilinear-interploation``的扩展。
之前在Classification的推理的时候，我们只需要对图像做一次resize就好了。
但是在做yolo的推理的时候，我们首先需要将图片给resize到yolo model可以识别的大小(e.g. 640x640)。
之后我们得到在这个尺寸下的bbox。但是我们在绘图的时候，我们还需要将bbox还原成原图的大小。
所以需要某一种形式进行resize的正向/反向的变换。这个可以通过warp affine来实现。


其实我们之前所作的bilinear-interpolation的实现，非常贴近warp affine了
```c++
// bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
int src_y2 = src_y1 + 1;
int src_x2 = src_x1 + 1;

...

// bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);
```

warp affine的基本公式就是
```c++
// forward
forward_scale = min(tar_w / src_w, tar_h / src_h);
tar_x         = src_x * forward_scale + forward_shift_x;
tar_y         = src_y * forward_scale + forward_shift_y;

// reverse
reverse_scale   = 1 / forward_scale;
reverse_shift_x = -forward_shift / forward_scale_x;
reverse_shift_y = -forward_shift / forward_scale_y;
src_x           = tar_x * reverse_scale + reverse_shift_x;
src_y           = tar_y * reverse_scale + reverse_shift_y;
```

如果给规范化一下的话可以写成
```c++
// forward
tar_x  =  src_x * forward_scale + src_y * 0             + forward_shift_x;
tar_y  =  src_x * 0             + src_y * forward_scale + forward_shift_y;

// reverse
src_x  =  tar_x * reverse_scale + tar_y * 0             + reverse_shift_x;
src_y  =  tar_x * 0             + tar_y * reverse_scale + reverse_shift_y;
```


我们可以通过matrix的形式保存这些scale和shift，等需要的时候直接使用。
在yolo的preprocess中，由于我们需要把图片resize成letter box，所以我们可以把scale和shift写成
```c++
// 存储forward时需要的scale和shift
void calc_forward_matrix(TransInfo trans){
    forward[0] = forward_scale;
    forward[1] = 0;
    forward[2] = - forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
    forward[3] = 0;
    forward[4] = forward_scale;
    forward[5] = - forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
};

// 存储reverse时需要的scale和shift
void calc_reverse_matrix(TransInfo trans){
    reverse[0] = reverse_scale;
    reverse[1] = 0;
    reverse[2] = - reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
    reverse[3] = 0;
    reverse[4] = reverse_scale;
    reverse[5] = - reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
};

// 仿射变换的计算公式
__device__ void affine_transformation(
    float trans_matrix[6], 
    int src_x, int src_y, 
    float* tar_x, float* tar_y)
{
    *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
    *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}
```
