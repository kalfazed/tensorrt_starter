trt-engine-explorer
===
``trt-engine-explorer``是NVIDIA官方提供的分析TensorRT优化后的推理引擎架构的工具包。可以在TensorRT官方的git repository中找到。链接在这里: 

[TensorRT tool: trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/release/8.6/tools/experimental/trt-engine-explorer)

参考官方提供的``README.md``把环境搭建起来之后，我们可以尝试分析一下我们的模型。为了方便，在``src/python``目录下放置了一些官方提供的python文件，可以使用。

比如说:
```python
# 通过TensorRT优化的推理引擎获取各个layer的信息，以json形式保存
python process_engine.py ../../result/resnet50/resnet50.engine ../../result/resnet50/resnet50.engine --profile-engine

# 通过保存得到的json信息，绘制出TensorRT优化后的网络模型架构，以SVG格式保存成图片
python draw_engine.py ../../result/resnet50/resnet50.engine.graph.json

# 通过jupyter在浏览器里打开SVG模式下的结构图
jupyter-notebook --ip=0.0.0.0 --no-browser
```

建议这里多花一下时间比较一下各个模型的``.onnx``架构和``.engine``架构的不同，以及同一个模型的不同精度的``.engine``的不同。
观察TensorRT优化后哪些层被融合了，哪些层是新添加的。感兴趣的可以考虑一下``reformatter``节点是什么。
这个``reformatter``非常重要，因为这个涉及到了TensorRT的各个layer所支持的Data layout。
如果想要把部署优化做到极致，理解Data layout是必须要做的。