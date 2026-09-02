# exp17 build.  ONE flag set for every compared algorithm (acceptance
# standard 1); -ffast-math is forbidden (would break inf/eps semantics) and
# -ffp-contract=off pins the FP operation order to Python's (bitwise
# crosscheck on dyadic instances).
CXX      := clang++
CXXFLAGS := -std=c++20 -O2 -DNDEBUG -ffp-contract=off -Wall -Wextra

DBGFLAGS := -std=c++20 -O0 -g -fsanitize=address,undefined -ffp-contract=off -Wall -Wextra

SRC  := main.cpp bcn_core.hpp generator.hpp io.hpp

all: build/bcn

build/bcn: $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

build/bcn_debug: $(SRC)
	@mkdir -p build
	$(CXX) $(DBGFLAGS) -o $@ main.cpp

debug: build/bcn_debug

clean:
	rm -rf build

.PHONY: all debug clean
