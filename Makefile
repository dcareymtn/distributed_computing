# Define required macros here
# CXX ?= g++
CXX = mpiCC
CU = nvcc

#PATHS
CPP_SRC_PATH=cpp/src
CPP_MAIN_PATH=cpp/main
CU_SRC_PATH=cu/src
CU_MAIN_PATH=cu/main
BUILD_PATH = build/lib
BUILD_CU_PATH = build/libcu
BUILD_MAIN_PATH = build/main
BUILD_MAIN_CU_PATH = build/cu_main
BIN_PATH=build/bin
BIN_CU_PATH=buildcu/bin
INC_PATH=include

# Extensions 
CPP_SRC_EXT =cpp
CU_SRC_EXT =cu

# Code lists
# Find all source files in the source directory, sorted by most recently modified
SOURCES = $(shell find $(CPP_SRC_PATH) -name '*.$(CPP_SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2)
CU_SOURCES = $(shell find $(CU_SRC_PATH) -name '*.$(CU_SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2) 

MAINSOURCES = $(shell find $(CPP_MAIN_PATH) -name '*.$(CPP_SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2)
CU_MAINSOURCES = $(shell find $(CU_MAIN_PATH) -name '*.$(CU_SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2)

# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
LIBOBJECTS=$(SOURCES:$(CPP_SRC_PATH)/%.$(CPP_SRC_EXT)=$(BUILD_PATH)/%.o)
CU_LIBOBJECTS=$(CU_SOURCES:$(CU_SRC_PATH)/%.$(CU_SRC_EXT)=$(BUILD_CU_PATH)/%.o)

MAINOBJECTS=$(MAINSOURCES:$(CPP_MAIN_PATH)/%.$(CPP_SRC_EXT)=$(BUILD_MAIN_PATH)/%.o)
CU_MAINOBJECTS=$(CU_MAINSOURCES:$(CU_MAIN_PATH)/%.$(CU_SRC_EXT)=$(BUILD_MAIN_CU_PATH)/%.o)

CU_BINOBJECTS=$(CU_MAINSOURCES:$(CU_MAIN_PATH)/%.$(CU_SRC_EXT)=$(BIN_CU_PATH)/%)
CPP_BINOBJECTS=$(MAINSOURCES:$(CPP_MAIN_PATH)/%.$(CPP_SRC_EXT)=$(BIN_PATH)/%)

BINOBJECTS=$(CPP_BINOBJECTS) $(CU_BINOBJECTS)


OBJECTS=$(LIBOBJECTS) $(MAINOBJECTS) $(CU_LIBOBJECTS) $(CU_MAINOBJECTS)

# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# flags "
#CFLAGS = -std=c++11 -Wall -Wextra -g
#CFLAGS = -std=c++11 -g -fopenmp
CFLAGS = -std=c++0x -g -fopenmp
#CFLAGS = -g -fopenmp
INCLUDES = -I inc/ -I /usr/local/include -I $(INC_PATH)

# Space-separated pkg-config libraries used by this project
LIBS = -lcuda -lcudart

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(CFLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating Directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(dir $(BINOBJECTS))
	@mkdir -p $(BIN_PATH)	
	@mkdir -p $(INC_PATH)

.PHONY: test
test: $(OBJECTS)
	@echo "$(OBJECTS)"
	@echo "$(BINOBJECTS)" 


.PHONY: clean
clean:
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)
	@$(RM) -r $(BUILD_MAIN_PATH)

# Checks the executable and symlinks to the output
.PHONY: all
all: $(BINOBJECTS)
	@echo "$(BINOBJECTS)"

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
$(BUILD_PATH)/%.o: $(CPP_SRC_PATH)/%.$(CPP_SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(LIBS) $(INCLUDES) -MP -MMD -c $< -o $@

# Source file rules for the main objects
$(BUILD_MAIN_PATH)/%.o: $(CPP_MAIN_PATH)/%.$(CPP_SRC_EXT)
	@echo "Compiling Main: $< -> $@"
	$(CXX) $(CXXFLAGS) $(LIBS) $(INCLUDES) -MP -MMD -c $< -o $@

$(BUILD_CU_PATH)/%.o: $(CU_SRC_PATH)/%.$(CU_SRC_EXT)
	@echo "Compiling Cuda Main: $< -> $@"
	$(CU) $(INCLUDES) $(LIBS) -c $< -o $@

$(BUILD_MAIN_CU_PATH)/%.o: $(CU_MAIN_PATH)/%.$(CU_SRC_EXT)
	@echo "Compiling Cuda Object: $< -> $@"
	$(CU) $(INCLUDES) $(LIBS) -c $< -o $@

# Rule to make the main binary files
$(BIN_PATH)/%: dirs $(OBJECTS) $(CPP_MAIN_PATH)/%.$(CPP_SRC_EXT) 
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBOBJECTS) $(CU_LIBOBJECTS) $(subst $(BIN_PATH),$(BUILD_MAIN_PATH),$@.o) -o $@

$(BIN_CU_PATH)/%: dirs $(OBJECTS) $(CU_MAIN_PATH)/%.$(CU_SRC_EXT)
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBOBJECTS) $(CU_LIBOBJECTS) $(subst $(BIN_CU_PATH),$(BUILD_MAIN_CU_PATH),$@.o) -o $@
