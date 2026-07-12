NVCC  = nvcc
ARCH ?= -arch=sm_86
FLAGS = -O3 $(ARCH) -std=c++17 -lineinfo
MODEL ?= yolo11n
BIN   = yolo11cuda

all: $(BIN)

$(BIN): engine/engine.cu third_party/stb_image.h
	$(NVCC) $(FLAGS) -o $(BIN) engine/engine.cu

# download weights, export graph + reference dumps for $(MODEL)  (yolo11n / yolo11s / yolo11m ...)
export:
	python3 export/export_yolo11.py $(MODEL)

# per-op numeric verification + end-to-end detection check
test: $(BIN)
	./$(BIN) dump build/$(MODEL) > /dev/null
	python3 test/compare.py build/$(MODEL)
	./$(BIN) detect build/$(MODEL)

bench: $(BIN)
	./$(BIN) bench build/$(MODEL) 300
	./$(BIN) pipeline build/$(MODEL) 300

clean:
	rm -f $(BIN) engine_bin

.PHONY: all export test bench clean
