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
	python3 export/export_yolo11.py $(MODEL) $(if $(IMGSZ),--imgsz $(IMGSZ))

# per-op numeric verification + end-to-end detection check
test: $(BIN)
	./$(BIN) dump build/$(MODEL) > /dev/null
	python3 test/compare.py build/$(MODEL)
	./$(BIN) detect build/$(MODEL)

bench: $(BIN)
	./$(BIN) bench build/$(MODEL) 300
	./$(BIN) pipeline build/$(MODEL) 300

clean:
	rm -f $(BIN) engine_bin yolo11serve engine_lib.o server/yolo.pb.* server/yolo.grpc.pb.*

# ---- batch-labeling gRPC server (needs: libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev) ----
server/yolo.grpc.pb.cc server/yolo.pb.cc: proto/yolo.proto
	protoc -Iproto --cpp_out=server --grpc_out=server \
	  --plugin=protoc-gen-grpc=$$(which grpc_cpp_plugin) proto/yolo.proto

engine_lib.o: engine/engine.cu engine/yolo11.h third_party/stb_image.h
	$(NVCC) $(FLAGS) -DYOLO11_LIB -c engine/engine.cu -o engine_lib.o

serve: yolo11serve
yolo11serve: server/serve.cpp server/yolo.pb.cc server/yolo.grpc.pb.cc engine_lib.o
	g++ -O2 -std=c++17 -I/usr/local/cuda/include -Iserver \
	  server/serve.cpp server/yolo.pb.cc server/yolo.grpc.pb.cc engine_lib.o \
	  -L/usr/local/cuda/lib64 -lcudart -lnvjpeg -lgrpc++ -lgrpc -lgpr -labsl_synchronization -lprotobuf -lpthread \
	  -o yolo11serve

.PHONY: all export test bench serve clean
