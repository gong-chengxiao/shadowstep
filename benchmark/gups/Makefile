CXX = g++
LDFLAGS = -lnuma
SRC = gups.cpp
OBJ_DIR = obj
BIN_DIR = bin
TARGET = $(BIN_DIR)/gups

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
	./$(TARGET) --wss 96 --hot 32 --threads 32 --mem 4 --cpu 0 --updates 10000000 --objsize 8

.PHONY: all clean run help