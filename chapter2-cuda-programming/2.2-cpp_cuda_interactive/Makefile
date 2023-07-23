CONFIG        :=  ../../config/Makefile.config
CONFIG_LOCAL  :=  ./config/Makefile.config

include $(CONFIG)
include $(CONFIG_LOCAL)

BUILD_PATH    :=  build
SRC_PATH      :=  src
CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)

CXX_SRC       +=  $(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      :=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_DEPS      :=  $(CXX_SRC)
APP_DEPS      +=  $(KERNELS_SRC)
APP_DEPS      +=  $(wildcard $(SRC_PATH)/*.hpp)


CUCC          :=  $(CUDA_DIR)/bin/nvcc
CXXFLAGS      :=  -std=c++11 -fPIC
CUDAFLAGS     :=  -Xcompiler -fPIC 

INCS          :=  -I $(CUDA_DIR)/include \
                  -I $(SRC_PATH) 

LIBS          :=  -L "$(CUDA_DIR)/lib64" \

ifeq ($(DEBUG),1)
CUDAFLAGS     +=  -g -O0 -G
CXXFLAGS      +=  -g -O0
else
CUDAFLAGS     +=  -O3
CXXFLAGS      +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=  -w
CXXFLAGS      +=  -w
endif

all:
	$(MAKE) $(APP)

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CUCC) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
	@echo finished building $@. Have fun!!

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	rm -rf $(APP)
	rm -rf build

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CC) -o $@ -c $< $(CXXFLAGS) $(INCS)

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) $(ARCH) -o $@ -c $< $(CUDAFLAGS) $(INCS)

.PHONY: all update show clean 
