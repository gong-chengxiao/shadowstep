CXX = g++
LDFLAGS = -lnuma
SRC = gups.cpp
TARGET = gups

OBJS = $(SRC:.cpp=.o)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: all
	./$(TARGET) --wss 96 --hot 32 --threads 32 --mem 4 --cpu 0 --updates 10000000 --objsize 8

.PHONY: all clean run help