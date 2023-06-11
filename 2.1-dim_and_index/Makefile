CONFIG            :=  ../config/Makefile.config
CONFIG_LOCAL      :=  ./config/Makefile.config

include $(CONFIG)
include $(CONFIG_LOCAL)


BUILD_PATH    :=	build
SRC_PATH      :=	src
CUDA_DIR      :=	/usr/local/cuda-$(CUDA_VER)

KERNELS_SRC   :=	$(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      +=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_MKS       :=	$(APP_OBJS:.o=.mk)

APP_DEPS      +=	$(KERNELS_SRC)



CUCC          :=	$(CUDA_DIR)/bin/nvcc
CUDAFLAGS     :=	-O3 --shared -Xcompiler -fPIC \
									-I $(CUDA_DIR)/include \

INCS          :=	-I $(CUDA_DIR)/include \
									-I $(SRC_PATH) 

LIBS          := -L "$(CUDA_DIR)/lib64" \

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
