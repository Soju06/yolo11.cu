// Embedding API for the yolo.cu engine (used by the serving binary).
#pragma once
#include <cuda_runtime.h>

extern "C" {

typedef struct { float x1, y1, x2, y2, score; int cls; } YoloDet;
typedef struct { float x, y, w, h, angle, score; int cls; } YoloObbDet;

// Load model dir, fuse, autotune, build+calibrate CUDA graphs for B = 1..max_batch.
void* yolo_create(const char* model_dir, int max_batch);
// Task of the loaded model: 0 = detect, 1 = obb, 2 = classify, 3 = segment. Check this
// before fetching: yolo_get is valid only on detect/segment handles and yolo_get_obb only
// on obb handles — the wrong getter reads the result buffer at the wrong record stride and
// returns garbage; classify results are not exposed through this API at all.
int yolo_task(void* h);
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
// Classify-model preprocessing (resize-shortest + center-crop; use instead of
// yolo_preprocess on classify handles).
void yolo_preprocess_cls(void* h, const unsigned char* dev_bgr, int height, int width, int slot);
// Top-k classes for a slot (classify handles). Returns entries written.
int yolo_get_cls(void* h, int slot, int* ids, float* probs, int k);
// Letterbox geometry of a slot (scale/top/left set by yolo_preprocess) — map segment masks
// (letterbox space, proto resolution) back to original coords with it.
void yolo_slot_geom(void* h, int slot, float* scale, int* top, int* left);
// OBB variant: rotated boxes (center xy, wh, angle in radians), original image coords
// (angle unchanged, no clipping — matches ultralytics scale_boxes(xywh=True)).
int yolo_get_obb(void* h, int slot, YoloObbDet* out, int cap);
// Segment mask dimensions (proto resolution, i.e. imgsz/4; 0/0 for non-segment models).
void yolo_mask_dim(void* h, int* ph, int* pw);
// Binary u8 {0,1} mask of detection `det` of slot `slot` (out: ph*pw bytes; call after
// yolo_run/yolo_sync). Masks are box-cropped, thresholded at proto resolution; mask pixel
// (mx,my) covers letterbox pixels [4mx,4mx+4)x[4my,4my+4) — mapping to original image
// coords (subtract pad, divide by scale) is the caller's job. Returns 0, or -1 if det is
// out of range or the model has no masks.
int yolo_get_mask(void* h, int slot, int det, unsigned char* out);

}  // extern "C"
