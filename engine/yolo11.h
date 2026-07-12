// Embedding API for the yolo11.cu engine (used by the serving binary).
#pragma once
#include <cuda_runtime.h>

extern "C" {

typedef struct { float x1, y1, x2, y2, score; int cls; } YoloDet;

// Load model dir, fuse, autotune, build+calibrate CUDA graphs for B = 1..max_batch.
void* yolo_create(const char* model_dir, int max_batch);
// Calibrated GPU latency (ms) of a full batch-B forward (net + decode + NMS + D2H).
float yolo_exec_ms(void* h, int B);
cudaStream_t yolo_stream(void* h);
// Letterbox+normalize a device BGR u8 HWC image into batch slot `slot` (async on engine stream).
// Detect-only: classify models need resize-shortest+center-crop, not letterbox — do not use
// this (or the detect getters below) on a classify model dir.
void yolo_preprocess(void* h, const unsigned char* dev_bgr, int height, int width, int slot);
// Launch the batch-B graph and synchronize. Results are in pinned host memory afterwards.
void yolo_run(void* h, int B);
// Async variant for pipelining (launch only / wait), same engine stream.
void yolo_run_async(void* h, int B);
void yolo_sync(void* h);
// Fetch detections for a slot, mapped back to original image coordinates. Returns count.
int yolo_get(void* h, int slot, YoloDet* out, int cap);

}  // extern "C"
