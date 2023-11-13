int8-calibration
===
这里主要介绍了如何用C++实现int8的calibrator。主要看
- src/cpp/trt_calibrator.cpp
- src/cpp/trt_model.cpp
- include/trt-calibrator.hpp

如果我们需要让我们的模型实现FP16或者INT8量化的话，我们需要在模型创建的时候在``config``里面设置。比如:
```c++
if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
    config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
} else if (builder->platformHasFastInt8() && m_params->prec == model::INT8) {
    config->setFlag(BuilderFlag::kINT8);
    config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
}
```

如果是FP16量化的话，其实这么两行就可以了，但是如果是INT8的话，我们需要自己设计一个calibrator(校准器)。
和设计``logger``与``Plugin``一样，我们在创建calibrator类的时候需要继承nvinfer1里的calibrator。NVIDIA官方提供了以下五种
- nvinfer1::IInt8EntropyCalibrator2
- nvinfer1::IInt8MinMaxCalibrator
- nvinfer1::IInt8EntropyCalibrator
- nvinfer1::IInt8LegacyCalibrator
- nvinfer1::IInt8Calibrator

不同的calibrator所能够实现的dynamic range是不一样的。具体有什么不一样可以回顾一下第四章节的内容。
我们在calibrator类中需要实现的函数只需要四个
```c++
int         getBatchSize() const noexcept override {return m_batchSize;};
bool        getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
const void* readCalibrationCache(std::size_t &length) noexcept override;
void        writeCalibrationCache (const void* ptr, std::size_t legth) noexcept override;
```

- ``getBatchSize``: 获取calibration的batch大小，需要注意的是不同的batch size会有不同的校准效果。
- ``getBatch``: calibration是获取一个batch的图像，之后H2D到GPU，在GPU做统计。这里需要注意的是，在``getBatch``获取的图像必须要和真正推理时所采用的预处理保持一直。不然dynamic range会不准
- ``readCalibrationCache``: 用来读取calibration table,也就是之前做calibration统计得到的各个layer输出tensor的dynamic range。实现这个函数可以让我们避免每次做int8推理的时候都需要做一次calibration
- ``writeCalibrationCache``: 将统计得到的dynamic range写入到calibration table中去


实现完了基本的calibrator之后，在build引擎的时候通过config指定calibrator就可以了。
```c++
shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(
    64, 
    "calibration/calibration_list_imagenet.txt", 
    "calibration/calibration_table.txt",
    3 * 224 * 224, 224, 224));
config->setInt8Calibrator(calibrator.get());
```

这里面的``calibration_list_imagenet.txt``使用的是``ImageNet2012``的test数据集的一部分。这里可以根据各自的情况去更改。需要注意的是，如果calibrator改变了，或者模型架构改变了，需要删除掉``calibration_table.txt``来重新计算dynamic range。否则会报错