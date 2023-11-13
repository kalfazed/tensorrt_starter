Quantization-analysis
===
这个小节主要解决``7.3-deploy-yolo-basic``中出现的int8量化掉点严重的问题。
当我们发现int8量化掉点严重的时候，我们可以有几个参考的点
- 是否在input/output附近做了int8量化
- 如果是multi-task的话，是否所有的task都掉点严重
- calibration的数据集是不是选的不是很好
- calibration batch size是不是选择的不是很好
- calibrator是不是没有选择好
- 某些计算是否不应该做量化
- 使用polygraphy分析

其实，恢复精度有很大程度上需要依靠经验，但是比较好的出发点是从量化的基本原理去寻找答案，结合yolov8的模型架构，
我们就能顺理成章的猜到yolov8掉点严重的原因是因为``IInt8EntropyCalibrator2``。
这个小节带着大家引导一下思路，以及今后遇到类似的问题的时候应该如何去考虑。
