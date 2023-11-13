Deploy-classification-advanced
===
基于6.1的代码，这里做了很多改动

首先我们希望让我们的推理框架能够有:
- 代码可复用性
- 可读性
- 安全性
- 可扩展性
- 可调试性

这里围绕着这些点简单展开一下：

## 代码可复用性

我们设计一个推理框架，我们会更多希望它能够支持很多task，但是这些task无论是classification，，
还是detection，还是segmentation, 还是pose estimation, 还是xxx, 我们都是会有一个
``前处理 -> DNN推理 -> 后处理 ``的这么一套流程。只不过不同的task的前/后处理会有所不同。
同时，我们在创建一个推理引擎的时候，都是``builder->network->config->parser->serialize->save file``。
做inference的时候，都是``load file->deserialize->engine->context->enqueue``这么一套，
我们在初始化的时候，都是要根据trt engine的信息做``分配host memory, 分配device memory``这些。
那么我们就会很自然的想到，我们是否可以设计一个类来实现这一套最基本的操作，之后不同的task来继承这个类，
去完成每一个task各自需要单独处理的内容。这个我们可以通过``C++工厂模式``的设计思路去搭建模型，实现封装。

## 可读性
我们希望我们的代码比较好的可读性，就意味着我们在设计的时候尽量通过接口来暴露或者隐蔽一些功能。
比如说，我们可以使用``worker``作为接口进行推理。在main中，我们只需要做到
``创建一个worker -> woker读取图片 -> worker做推理``就好了。同时，worker也只暴露这些接口。
在``worker``内部，我们可以让``worker``根据main函数传入的参数，启动多种不同的task。
比如
    - ``worker->classifier`` 负责做分类推理
    - ``worker->detector`` 负责做检测
    - ``worker->segmetor`` 负责做分割
``worker``的具体每一个task会根据task的内容做自己的实现

## 安全性
我们在设计框架的时候，需要做很多初始化，释放内存，以及针对错误调用时的处理。
当代码的规模仍然还是很小的时候，意识安全性其实并不是很复杂。但是当代码的规模很大，有很多个task需要我们实现的话
我们难免会遇到一下类似于
    - 忘记释放内存
    - 忘记对某一种调用做error handler
    - 没有分配内存，却释放了内存
    - ...
为了避免这种情况发生，可以使用unique pointer或者shared pointer这种智能指针帮助我们管理内存的释放。
以及使用``RAII设计机制``，将资源的申请封装在一个对象的生命周期内，方便管理。
``RAII``是Resource acquisition is initialization的缩写，中文译为“资源获取即初始化”。
比较常见的方法就是在一个类的构造函数的时候就把一系列初始化完成。

## 可扩展性
一个比较好的框架需要有很强的扩展性。这就意味着我们的设计需要尽量模块化。当有新的task出现的时候，
我们可以做到最小限度的代码更改。比如说
```c++
    worker(...);                                         //根据模型的种类(分类、检测、分割)来初始化一个模型
   
    worker->classifier(...);                             //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->classifier.load_image(...);                  //在这里我们读取图片，并分配pinned memory，分配device memory
    score = worker->classifier.infer_classifier(...);    //在这里我们进行预处理，推理，后处理的部分

    worker->detector(...);                               //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->detector.load_image(...);                    //在这里我们读取图片，并分配pinned memory，分配device memory
    bboxes = worker->detector.infer_detector(...);       //在这里我们进行预处理，推理，后处理的部分
    
    worker->segmenter(...);                              //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->segmenter.load_image(...);                   //在这里我们读取图片，并分配pinned memory，分配device memory
    mask = worker->segmenter.infer_segmentor(...);       //在这里我们进行预处理，推理，后处理的部分
    
    worker->drawBBox(...);                               //worker负责将bbox的信息绘制在原图上
    worker->drawMask(...);                               //worker负责将mask的信息融合在原图上
    worker->drawScore(...);                              //worker负责将score的信息绘制在原图上
```
这个案例下的worker的内容是很空的。因为目前只是一个单独的classification的任务。整体上的结构很simple。
但这么设计的目的是为了今后的扩展，比如说针对视频流的异步处理，多线程的处理，multi-stage model的处理，multi-task model的处理等等。因为比如说会出现这种情况：
```c++
    // 1st stage detection
    worker->detector(...);
    worker->detector.load_image(...);
    bboxes = worker->detector.infer_detector(...);

    // 2nd stage classification
    worker->classifier(...);
    worker->classifier.load_from_bbox(...);
    score = worker->classifier.infer_classifier(...);
```

## 可调试性
在设计框架的时候，我们希望能够为了让开发效率提高，在设计中比较关键的几个点设置debug信息，方便我们查看我们的模型是否设计的有问题
比如说，yolo在NMS处理之后，bbox还剩多少; ONNX模型在TensorRT优化以后，网络的架构是什么样子了; 模型各个layer所支持的输入的data layout
是NCHW还是NHWC等等。我们可以实现一个``logger``来方便我们管理这些。``logger``可以通过传入的不同参数显示不同级别的日志信息。
比如说，如果我们在main中声明我们想打印有关``VERBOSE``信息，我们可以打印在代码中所有以``LOGV()``显示的信息。
