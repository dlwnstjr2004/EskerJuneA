# ~/sample/python/uff_ssd/
mkdir build
cd build
sudo cmake .. -DNVINFER_LIB=/home/wyl/software/pc/TensorRT-6.0.1.8/lib/libnvinfer.so

# uff_ssd/plugin/FlattenConcat.cpp
# change "NvInferPlugin.h"
#include "<YOUR_TensorRT(Version)_PATH/include/NvInferPlugin.h>"
sudo make
