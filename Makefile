CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -mavx2 -fPIC -Wall -Wextra
all: libaci.so
libaci.so: src/aci.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@
clean:
	rm -f libaci.so
