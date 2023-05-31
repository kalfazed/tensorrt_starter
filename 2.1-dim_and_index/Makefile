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


BUILD_PATH    :=	build
SRC_PATH      :=	src
CUDA_DIR      :=	/usr/local/cuda-$(CUDA_VER)

CXX_SRC       +=	$(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC   :=	$(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      :=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS      +=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_MKS       :=	$(APP_OBJS:.o=.mk)

APP_DEPS      :=	$(CXX_SRC)
APP_DEPS      +=	$(KERNELS_SRC)
APP_DEPS      +=	$(wildcard $(SRC_PATH)/*.h)



CUCC          :=	$(CUDA_DIR)/bin/nvcc
CUDAFLAGS     :=	-O3 -arch=compute_86 --shared -Xcompiler -fPIC \
									-I $(CUDA_DIR)/include \

INCS          :=	-I $(CUDA_DIR)/include \
									-I $(SRC_PATH) 

LIBS          := -L "$(CUDA_DIR)/lib64" \
								 -lcudart -lcublas -lcudnn

ifeq ($(DEBUG),1)
CUDAFLAGS     +=	-g -O0
else
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=	-Wall -Wunused-function -Wunused-variable -Wfatal-errors
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
