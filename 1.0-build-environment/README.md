# 课程环境搭建设置

---

## Server环境搭建(without docker)

[安装OpenSSH server并启动](https://www.cyberciti.biz/faq/ubuntu-linux-install-openssh-server/)

[通过GPU显卡型号找最新的driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)

[寻找TensorRT release note](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#rel-8-5-1)

[安装cuda(这里是11.6的链接，根据情况选择版本)](https://developer.nvidia.com/cuda-11-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)

[安装cuDNN的流程](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)



## Server环境搭建(with docker)

[安装NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)



## Server环境搭建(with/without docker)

[安装opencv](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html)



## 提高开发效率的软件安装与环境设置

[安装fish shell](https://fishshell.com)

[安装fisher进行fish包的管理](https://github.com/jorgebucaran/fisher)

[使用z进行目录的高速跳转](https://github.com/jethrokuan/z)

[使用peco快速寻找history以及git版本](https://github.com/peco/peco)

[ssh远程linux是转发X11使用GUI](https://blog.csdn.net/weixin_44966641/article/details/120365459)

[安装fim图片浏览器从终端打开图片](https://howtoinstall.co/en/fim)

[安装tmux进行多window多session的管理](https://github.com/tmux/tmux/wiki)

[miniconda的安装进行python环境的隔离](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

[安装netron进行DNN网络架构图的分析](https://github.com/lutzroeder/netron)



## 使用neovim进行开发环境的设置

[neovim编辑器的安装, 推荐使用v0.8.0](https://neovim.io/)

[lunarvim编辑器安装, 推荐使用v1.2](https://www.lunarvim.org)

[使用lsp signature进行neovim补全信息的提示](https://github.com/ray-x/lsp_signature.nvim)

[设置lsp server进行编程语言的自动补全](https://github.com/neovim/nvim-lspconfig)

[安装trouble.nvim进行代码的diagnostics](https://github.com/folke/trouble.nvim)

[安装markdown preview进行远程实时编辑markdown](https://github.com/iamcco/markdown-preview.nvim)

[开启端口允许远程访问markdown preview](https://blog.csdn.net/haima1998/article/details/112741623)

[通过DAP在neovim中进行C/C++/Python的Debug](https://github.com/mfussenegger/nvim-dap)

[扩展neovim的DAP进行GUI上的Debug操作](https://github.com/rcarriga/nvim-dap-ui)

[可以参考的C/C++的DAP环境配置](https://github.com/LunarVim/starter.lvim/blob/c-ide/config.lua)

### neovim配置(其他)
这些配置也很重要但没有找到可以参考的资料, 可以看我的lvim/lua/下各个文件的配置文件的写法
- 设置lsp diagnostics的error/warning信息的提示方式
- 对python lsp server进行python错误显示的级别设定
- 设置.clangd进行C/C++, CUDA代码实现/声明/使用跳转

---
## 有关my_dot_files
这里面是我平时用的一些配置文件，包含lvim, fish, 以及tmux。如果感兴趣的话可以拿过去直接用
```
cp -r my_dot_files/fish/* ~/.config/fish/
cp -r my_dot_files/lvim/* ~/.config/lvim/
cp -r my_dot_files/tmux/* ~/
```
我在my_dot_files/fish/config.fish，以及my_dot_files/lvim/config.lua里面的开头我做了一些标注，
大家根据自己的情况去配置一下

## 有关dockerfile
请根据需求修改dockerfile中的前三行:
```
FROM nvcr.io/nvidia/tensorrt:21.12-py3   
ENV TZ=Asia/Tokyo
ARG user=trt-starter
```

我在scripts/里面写了两个脚本，大家可以方便使用:
```bash
# 请根据情况更改v1.0

# `build-docker.sh`是通过dockerfile创建docker image的脚本
bash scripts/build-docker.sh v1.0

# `run-docker.sh`是通过已经建立好的docker image创建container的脚本
# 大家根据情况可以自己添加port以及volumn
bash scripts/run-docker.sh v1.0
```


