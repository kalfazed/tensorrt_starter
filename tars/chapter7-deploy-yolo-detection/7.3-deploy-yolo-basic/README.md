Deploy-yolo-basic
===
实现一个基本的yolov8的部署。整个代码是对``6.4小节``的扩展。
由于我们创建的推理框架是模块化的，所以我们在``6.4小节``的基础上需要扩展的东西其实只有这些。
- src/cpp/trt_detector.cpp
- src/cpp/trt_worker.cpp
- src/cpp/trt_model.cpp
- include/trt_detector.hpp
- inlcude/trt_worker.hpp

对于worker的修改，我们只需要修改一下构造函数，添加一个针对detection task的成员变量``m_detector``，
之后通过m_detector来做推理就可以了。

对于detector的实现，我们只需要关注preprocess和postprocess的实现就可以了。
preprocess部分我们已经实现好了一个生成letterbox的方法，我们可以使用warp affine。
postprocess是我们真正需要关注的。我们需要自己实现一个decode和nms的算法。
为了与pytorch对齐，我们需要先知道PyTorch是如何实现的

## 理解ultralytics的后处理实现
我们可以直接从官方clone下来yolov8。由于我们需要更改模型，让其能够生成适合我们做推理的结果，所以我们从源码安装
```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

之后我们就针对yolov8的权重export一个onnx就好了。官方提供了非常方便的接口，我们可以直接用
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')
```

但是由于onnx的输出格式是[n, feature, bbox]，不方便我们在C++做处理。我们希望对每一个bbox做处理，
所以希望每一个bbox内部的数据是内存上连续的，我们可以更改一下detect的head部分
```python
# ultralytics/ultralytics/nn/modules/head.py
class Detect(nn.Module):
    # ...
    def forward(self, x):
        # ...
        y = torch.cat((dbox, cls.sigmoid()), 1)
        y = y.transpose(1, 2)
        return y if self.export else (y, x)
```
其实只需要做这一步就可以了。但是如果要导出支持动态batch的onnx,需要改的地方会比较多，我们之后有机会讲。
这个小节我们只关注postprocess部分

## 基于pytorch的实现在C++实现后处理

### Postprocess -- yolov8的postprocess需要做的事情
1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
3. 把最终得到的bbox绘制到原图中

### Postprocess -- 1. decode
我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
1. 从每一个bbox中对应的ch中获取cx, cy, width, height
2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
3. 将cx, cy, width, height转换为x0, y0, x1, y1
4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
    

### Postprocess -- 2. NMS
1. 做一个IoU计算的lambda函数
2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold

### Postprocess -- 3. draw_bbox
1. 通过label获取name
2. 通过label获取color
3. cv::rectangle
4. cv::putText

理解了算法实现起来就很容易了。详细请看代码。

## int8 Calibration
基于classification章节实现的int8 calibrator,我们也可以用在yolov8上。
代码中采用的calibration数据及是coco2017,大家可以根据自己的情况修改。
然而，当我们在做int8推理的时候，我们会发现精度下降非常严重。主要是出现在无法检测到物体。
有关这个问题会在``7.4-quantization-analysis``讲解。