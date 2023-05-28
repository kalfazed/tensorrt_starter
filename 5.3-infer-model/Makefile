CONFIG 			:=	./config/Makefile.config

ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

ifeq ($(CUDA_VER),)
$(error "CUDA_VER variable is not set in Makefile.config")
endif

ifeq ($(PLATFORM),)
$(error "PLATFORM variable is not set in Makefile.config")
endif


BUILD_PATH		:=	build
SRC_PATH		:=	src
CUDA_DIR		:=	/usr/local/cuda-$(CUDA_VER)

CXX_SRC			+=	$(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC		:=	$(wildcard $(SRC_PATH)/*.cu)

APP_OBJS		:=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS		+=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_MKS			:=	$(APP_OBJS:.o=.mk)

APP_DEPS		:=	$(CXX_SRC)
APP_DEPS		+=	$(KERNELS_SRC)
APP_DEPS		+=	$(wildcard $(SRC_PATH)/*.h)



CUCC			:=	$(CUDA_DIR)/bin/nvcc
CUDAFLAGS		:=	-O3 -arch=compute_86 --shared -Xcompiler -fPIC \
					-I $(CUDA_DIR)/include \

INCS			:=	-I $(CUDA_DIR)/include \
					-I $(TENSORRT_INSTALL_DIR)/include \
					-I $(OPENCV_INSTALL_DIR) \
					-I /usr/include/libxml2/ \
					-I $(SRC_PATH) \

LIBS			:=	-L "$(TENSORRT_INSTALL_DIR)/lib" \
					-L "$(CUDA_DIR)/lib64" \
					-Wl,-rpath="$(TENSORRT_INSTALL_DIR)/lib" \
					-Wl,-rpath="$(CUDA_DIR)/lib64" \
					-lnvinfer -lnvinfer_plugin \
					-lcudart -lcublas -lcudnn\
					-lgflags -lpthread \
					`pkg-config --libs opencv` \
					-lboost_system -lboost_program_options -lboost_filesystem \
					-lyaml-cpp -lstdc++fs -ldl -std=gnu++11 -lnvparsers \

ifeq ($(DEBUG),1)
CXXFLAGS		:=	-g -O0 -std=c++11 -lstdc++fs -w -DOPENCV -fopenmp -lxml2 -lnvonnxparser
else
CXXFLAGS		:=	-O3 -std=c++11 -lstdc++fs -w -DOPENCV -fopenmp -lxml2 -lnvonnxparser
endif

ifeq ($(SHOW_WARNING),1)
CXXFLAGS		+=	-Wall -Wunused-function -Wunused-variable -Wfatal-errors
endif

ifeq ($(APP),trt-infer)
CXXFLAGS		+=	-D MAIN_
endif

all: dirs deps
	$(MAKE) $(APP)

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	$(CXX) $(APP_OBJS) -o $@ $(LIBS) $(CXXFLAGS) $(INCS)
	@echo finished building $@. Have fun!!

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

dirs:	
	@if [ ! -d "build" ]; then mkdir -p build; fi

install: $(APP)
	@echo Installing $(APP)
	cp -rv $(APP) /usr/bin/

clean:
	rm -rf $(APP)
	rm -rf build
	rm -rf models/*.engine

reinstall: $(APP)
	@echo "Re installing $(APP)"
	$(MAKE) clean
	$(MAKE) all
	$(MAKE) install

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp 
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)
$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@echo Compile Dependence CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)
$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)%.cu
	@echo Compile Dependence CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)

.PHONY: all deps install clean clean_models clean_detections reinstall
