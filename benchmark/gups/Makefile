CXX = g++
SRC = gups.cpp
OBJ_DIR = obj
BIN_DIR = bin
TARGET = $(BIN_DIR)/gups
LDFLAGS = -lnuma -g

OBJS = $(addprefix $(OBJ_DIR)/, $(SRC:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/* $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run help