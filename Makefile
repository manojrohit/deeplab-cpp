# Compiler and flags
CXX := g++

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
LIB_DIR := /usr/local/lib
TENSORFLOW_INC_DIR := /home/manoj/.local/lib/python3.10/site-packages/tensorflow/include
OPENCV_INC_DIR := /usr/local/include/opencv4

# Libraries
TENSORFLOW_LIBS := -ltensorflow_cc -ltensorflow_framework
OPENCV_LIBS := -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn
YAML_LIBS := -lyaml-cpp

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Target executable
TARGET := deeplab_inference

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(LIB_DIR) $(TENSORFLOW_LIBS) $(OPENCV_LIBS) $(YAML_LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -I$(TENSORFLOW_INC_DIR) -I$(OPENCV_INC_DIR) -c -o $@ $<

clean:
	@rm -rf $(BUILD_DIR) $(TARGET)