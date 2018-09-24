# Define required macros here
CXX ?= g++

#PATHS
SRC_PATH=cpp/src
MAIN_PATH=cpp/main
BUILD_PATH = build/lib
BUILD_MAIN_PATH = build/main
BIN_PATH=build/bin
INC_PATH=include

# executable
BIN_NAME = homework1 homework2
# Extension
SRC_EXT = cpp

# Code lists
# Find all source files in the source directory, sorted by most recently modified
SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2)
MAINSOURCES = $(shell find $(MAIN_PATH) -name '*.$(SRC_EXT)' -printf '%T@ %p\n' | sort -k 1nr | cut -d ' ' -f 2)

# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
LIBOBJECTS=$(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
MAINOBJECTS=$(MAINSOURCES:$(MAIN_PATH)/%.$(SRC_EXT)=$(BUILD_MAIN_PATH)/%.o)

BINOBJECTS=$(MAINSOURCES:$(MAIN_PATH)/%.$(SRC_EXT)=$(BIN_PATH)/%)

#OBJECTS = $(LIBOBJECTS)
OBJECTS=$(LIBOBJECTS) $(MAINOBJECTS)

# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# flags "
#CFLAGS = -std=c++11 -Wall -Wextra -g
CFLAGS = -std=c++11 -g -fopenmp
INCLUDES = -I inc/ -I /usr/local/include -I $(INC_PATH)

# Space-separated pkg-config libraries used by this project
LIBS = 

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

.PHONY: clean
clean:
	@echo "Deleting $(BIN_NAME) symlink"
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
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# Source file rules for the main objects
$(BUILD_MAIN_PATH)/%.o: $(MAIN_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# Rule to make the main binary files
$(BIN_PATH)/%: dirs $(OBJECTS) $(MAIN_PATH)/%.$(SRC_EXT) 
	$(CXX) $(CXXFLAGS) $(LIBOBJECTS) $(subst $(BIN_PATH),$(BUILD_MAIN_PATH),$@.o) -o $@
