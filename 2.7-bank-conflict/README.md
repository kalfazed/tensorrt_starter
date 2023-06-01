# 有关CUDA编程中的memory管理

---

## Cuda shared memory的相关部分

- 一个SM上有的memory的种类

  - Register
  - Shared memory
	- L1 cache
  - Read only区域？
  - Constant区域？

- 在SM之下的会有

	- L2 Cache (local memory / global memory)
  - DRAM

- Shared memory和L1 cache相比于L2 cache延迟大概低20~30倍，带宽大约是10倍

- shared memory的大小一般是48Kb

- shared memory是跟block绑定的。一个block内部的所有threads共享一块shared memory

- 根据在Shared memory中变量的申明方式，可以将变量分为

	- 静态共享变量 (__shared__ 之后固定大小)

		- 地址是不一样的
    - 可以是一维、二维、三维

  - 动态共享变量(extern __shared__之后不指定大小，大小在kernel函数启动的时候指定)

  	- 所有的地址都是一样的
    - 注意，动态声明只支持一位数组

- shared memory是一维的，和其他内存一样。如果生命的时候是二维的，在寻址的时候需要转换为一维的方式进行寻址。

- shared memory的存储形式是有32个存储体(bank)，分为对应一个warp中的32个线程。如果每一个线程访问的bank和其他不冲突，效率是最高的。如果冲突的话带宽利用率会降低。我们管这个叫做bank conflict

	- 这里需要注意的是：一个shared memory可以有32个bank，那么就说明每一个bank可以有多个地址。很多地址共用同一块bank
  - 这里所说的bank conflict不是指访问同一个地址，而是访问同一个bank中的不同地址
  - 如果访问的是同一个地址的话，其实是不会有冲突的。因为那个地址一更新，更新的值会广播到所有访问它的thread中。
  - 但比如说一个warp中的32个线程，对同一个bank中的N的地址同时访问了。那么这个访问就需要被序列化。因此会导致有n次内存事务来完成这对这个bank的访问。大大的降低了内存带宽。因为如果不冲突的话只需要1次内存事务就可以了。

- bank的宽度可以是4个字节(32bit，单精度浮点数)，也可以是8个字节(64bit，双精度浮点数)。比如说如果bank宽度是4个字节，由于shared memory是32个bank来划分的，所以每一个bank以128个字节为单位进行stride，来存储不同地址的数据(图)

- 那么如何防止bank conflict呢？

	- 调整数组的大小进行padding就okay
  - 加大bank的宽度。比如之前是4个字节，改成8个字节。将一个bank分成两个部分

- 什么时候容易发生bank conflict呢？

	- 矩阵转置





## CUDA中对内存的配置选项

- 我们可以配置bank的大小

- 我们可以配置shared memory和L1 cache的大小

	- 基本上来说SM上有64KB的片上内存。其中48KB是shared memory的，16KB是L1 cache的
  - 如果我们的核函数共享内存使用的多，那么我们就扩大shared memory
  - 如果我们的核函数寄存器使用的多，那么我们就扩大L1 cache



__syncthreads()是干什么的呢？是指在同一线程块内的所有线程的同步。起到一个barrier的作用


