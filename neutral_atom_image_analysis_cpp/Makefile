CC=c++
INCLUDE=-I../extern/pybind11/include -Iinclude -I../extern/eigen $(shell python3-config --includes)
CXXFLAGS=-O3 -Wall -shared -std=c++17 -fPIC -fopenmp $(INCLUDE)

TARGET_DIR=../neutral_atom_image_analysis/
SRC_DIR	:= src
OBJ_DIR	:= obj
BIN_DIR	:= bin

SOURCES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp)
OBJECTS	:= $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: module
fresh:
	-rm $(OBJECTS)
	-rm $(BIN_DIR)/neutral_atom_image_analysis_cpp$(shell python3-config --extension-suffix)
	make module

module: 
	make $(BIN_DIR)/neutral_atom_image_analysis_cpp
$(BIN_DIR)/neutral_atom_image_analysis_cpp: $(OBJECTS) | $(BIN_DIR) $(TARGET_DIR)
	$(CC) $^ -o $@$(shell python3-config --extension-suffix) $(CXXFLAGS)
	cp $@$(shell python3-config --extension-suffix) $(TARGET_DIR)$(notdir $@$(shell python3-config --extension-suffix))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(CXXFLAGS)
$(BIN_DIR) $(OBJ_DIR) $(TARGET_DIR):
	mkdir -p $@