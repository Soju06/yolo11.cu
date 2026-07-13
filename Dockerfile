# yoloserve container. Build:  docker build -t yoloserve .
# Run:  docker run --gpus all -p 50051:50051 -v $(pwd)/build:/app/build yoloserve
# (export the model on the host first: make export MODEL=yolo11n)
ARG CUDA_TAG=13.0.0-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}
ARG ARCH=-arch=sm_86

RUN apt-get update && apt-get install -y --no-install-recommends \
    make g++ libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY engine engine/
COPY server server/
COPY proto proto/
COPY third_party third_party/
RUN make serve ARCH="${ARCH}"

EXPOSE 50051
ENTRYPOINT ["./yoloserve"]
CMD ["--dir", "build/yolo11n", "--max-batch", "16", "--target-ms", "50"]
