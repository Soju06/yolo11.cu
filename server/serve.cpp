// yoloserve — gRPC batch-labeling server on top of the yolo.cu engine.
//
// Scheduling: deadline-aware dynamic batching. Every request gets a deadline
// (arrival + target latency). The batcher keeps admitting requests into the
// pending batch for as long as the OLDEST request could still meet its
// deadline if we waited and then ran a full max-batch — using the engine's
// calibrated per-batch-size latency model plus a decode-time EMA. The moment
// waiting longer would risk the oldest deadline (or the batch is full), it
// fires with everything queued. This maximizes batch size — and therefore
// throughput — subject to the latency SLO.
#include <grpcpp/grpcpp.h>
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "yolo.pb.h"
#include "yolo.grpc.pb.h"
#include "../engine/yolo.h"
#include "../third_party/stb_image.h"   // implementation lives in the engine object

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Req {
  int model = 0;
  const std::string* data;
  Clock::time_point arrival, deadline;
  yolo::DetectResponse* resp;
  std::promise<bool> done;   // false = decode failure
};

struct Model {
  void* eng = nullptr;
  std::string name;          // dir basename, request routing key
  int task = 0;              // 0 detect, 1 obb, 2 classify, 3 segment
  const char* taskName = "detect";
  int maxB = 16;
  int maskH = 0, maskW = 0;  // segment proto resolution
  std::vector<unsigned char> maskBuf;
  std::deque<Req*> q;        // per-model queue (batches never mix models)
};

struct Server {
  std::vector<Model> models;
  int maxB = 16;             // max over models (scratch sizing)
  double targetMs, safetyMs = 1.0;
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<long> served{0};
  double decodeEmaMs = 2.0;   // per-image decode estimate, updated online

  // nvJPEG + per-slot device scratch
  nvjpegHandle_t nvj;
  std::vector<unsigned char*> scratch;
  std::vector<size_t> scratchSz;

  // decoder pool: per-thread nvJPEG state + CUDA stream; parallel CPU huffman across cores.
  struct DecodeJob {
    std::vector<Req*>* batch;
    std::vector<int>*ok; std::vector<int>* hh; std::vector<int>* ww;
    std::atomic<int> next{0}, done{0};
    int B = 0;
  };
  DecodeJob* job = nullptr;
  std::mutex dmu; std::condition_variable dcv, dcvDone;
  int nDecoders = 8;

  int slotBase = 0;   // current scratch set (0 or maxB)

  void decodeOne(nvjpegJpegState_t state, cudaStream_t st, int i) {
    int slot = slotBase + i;
    auto& bytes = *(*job->batch)[i]->data;
    const auto* p = (const unsigned char*)bytes.data();
    int& w = (*job->ww)[i]; int& h = (*job->hh)[i];
    (*job->ok)[i] = 0;
    if (bytes.size() > 3 && p[0] == 0xFF && p[1] == 0xD8) {   // jpeg -> GPU decode
      int nc; nvjpegChromaSubsampling_t ss; int ws[NVJPEG_MAX_COMPONENT], hs[NVJPEG_MAX_COMPONENT];
      if (nvjpegGetImageInfo(nvj, p, bytes.size(), &nc, &ss, ws, hs) != NVJPEG_STATUS_SUCCESS) return;
      w = ws[0]; h = hs[0];
      ensure(slot, (size_t)w * h * 3);
      nvjpegImage_t out{};
      out.channel[0] = scratch[slot];
      out.pitch[0] = w * 3;
      if (nvjpegDecode(nvj, state, p, bytes.size(), NVJPEG_OUTPUT_BGRI, &out, st) != NVJPEG_STATUS_SUCCESS)
        return;
      CK(cudaStreamSynchronize(st));
      (*job->ok)[i] = 1;
    } else {                                                  // png etc -> stb on CPU
      int comp;
      unsigned char* img = stbi_load_from_memory(p, (int)bytes.size(), &w, &h, &comp, 3);
      if (!img) return;
      for (size_t k = 0; k < (size_t)w * h; k++) std::swap(img[k * 3], img[k * 3 + 2]);
      ensure(slot, (size_t)w * h * 3);
      CK(cudaMemcpyAsync(scratch[slot], img, (size_t)w * h * 3, cudaMemcpyHostToDevice, st));
      CK(cudaStreamSynchronize(st));
      stbi_image_free(img);
      (*job->ok)[i] = 1;
    }
  }

  void decoderThread() {
    nvjpegJpegState_t state; nvjpegJpegStateCreate(nvj, &state);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    for (;;) {
      DecodeJob* j;
      {
        std::unique_lock<std::mutex> lk(dmu);
        dcv.wait(lk, [&] { return job && job->next.load() < job->B; });
        j = job;
      }
      int i;
      while ((i = j->next.fetch_add(1)) < j->B) {
        decodeOne(state, st, i);
        if (j->done.fetch_add(1) + 1 == j->B) { std::lock_guard<std::mutex> lk(dmu); dcvDone.notify_all(); }
      }
    }
  }

  void decodeBatch(std::vector<Req*>& batch, std::vector<int>& ok,
                   std::vector<int>& hh, std::vector<int>& ww) {
    DecodeJob j;
    j.batch = &batch; j.ok = &ok; j.hh = &hh; j.ww = &ww; j.B = (int)batch.size();
    {
      std::lock_guard<std::mutex> lk(dmu);
      job = &j;
    }
    dcv.notify_all();
    {
      std::unique_lock<std::mutex> lk(dmu);
      dcvDone.wait(lk, [&] { return j.done.load() == j.B; });
      job = nullptr;
    }
  }
  void ensure(int slot, size_t n) {
    if (scratchSz[slot] >= n) return;
    if (scratch[slot]) CK(cudaFree(scratch[slot]));
    CK(cudaMalloc(&scratch[slot], n));
    scratchSz[slot] = n;
  }

  // ---- the batcher + 2-stage pipeline: decode batch k+1 concurrently with GPU batch k ----
  struct Inflight {
    std::vector<Req*> batch;
    std::vector<int> ok, hh, ww;
    Clock::time_point tLaunch;
    int B = 0, model = 0;
  };

  // pick the most urgent model queue: fire when a queue is full or its oldest request's
  // slack (given that model's calibrated exec time) runs out; otherwise wait for more work.
  int collect(std::vector<Req*>& batch) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { for (auto& m : models) if (!m.q.empty()) return true; return false; });
    int pick;
    for (;;) {
      pick = -1;
      double minSlack = 1e30;
      bool fire = false;
      for (int mi = 0; mi < (int)models.size(); mi++) {
        Model& m = models[mi];
        if (m.q.empty()) continue;
        double slack = targetMs - ms_since(m.q.front()->arrival, Clock::now())
                       - yolo_exec_ms(m.eng, m.maxB) - decodeEmaMs - safetyMs;
        bool ready = (int)m.q.size() >= m.maxB || slack <= 0;
        if ((ready && !fire) || (ready == fire && slack < minSlack)) {
          pick = mi; minSlack = slack; fire = fire || ready;
        }
      }
      if (fire) break;
      cv.wait_for(lk, std::chrono::microseconds(std::min((long)(minSlack * 1000), 200L)));
    }
    Model& m = models[pick];
    int B = std::min((int)m.q.size(), m.maxB);
    batch.assign(m.q.begin(), m.q.begin() + B);
    m.q.erase(m.q.begin(), m.q.begin() + B);
    return pick;
  }

  // encode a binary u8 mask as alternating 0/1 run lengths (row-major, starts with a 0-run)
  static void rleEncode(const unsigned char* m, size_t n, yolo::Mask* out) {
    unsigned char cur = 0;
    uint32_t run = 0;
    for (size_t i = 0; i < n; i++) {
      if (m[i] == cur) { run++; continue; }
      out->add_rle(run);
      cur = m[i]; run = 1;
    }
    out->add_rle(run);
  }

  void respond(Inflight& f, double inferMs) {
    Model& m = models[f.model];
    void* eng = m.eng;
    int task = m.task;
    YoloDet dets[300];
    YoloObbDet rdets[300];
    for (int i = 0; i < f.B; i++) {
      Req* r = f.batch[i];
      r->resp->set_task(m.taskName);
      if (f.ok[i]) {
        if (task == 2) {                                     // classify: top-5
          int ids[5]; float probs[5];
          int n = yolo_get_cls(eng, i, ids, probs, 5);
          for (int j = 0; j < n; j++) {
            auto* c = r->resp->add_classes();
            c->set_id(ids[j]); c->set_prob(probs[j]);
          }
        } else if (task == 1) {                              // obb: rotated boxes
          int n = yolo_get_obb(eng, i, rdets, 300);
          for (int j = 0; j < n; j++) {
            auto* b = r->resp->add_rboxes();
            b->set_cx(rdets[j].x); b->set_cy(rdets[j].y);
            b->set_w(rdets[j].w); b->set_h(rdets[j].h);
            b->set_angle(rdets[j].angle);
            b->set_score(rdets[j].score); b->set_cls(rdets[j].cls);
          }
        } else {                                             // detect / segment: boxes
          int n = yolo_get(eng, i, dets, 300);
          for (int j = 0; j < n; j++) {
            auto* b = r->resp->add_boxes();
            b->set_x1(dets[j].x1); b->set_y1(dets[j].y1);
            b->set_x2(dets[j].x2); b->set_y2(dets[j].y2);
            b->set_score(dets[j].score); b->set_cls(dets[j].cls);
          }
          if (task == 3) {                                   // segment: RLE masks + geometry
            r->resp->set_mask_h(m.maskH); r->resp->set_mask_w(m.maskW);
            float sc; int top, left;
            yolo_slot_geom(eng, i, &sc, &top, &left);
            r->resp->set_lb_scale(sc); r->resp->set_lb_top(top); r->resp->set_lb_left(left);
            for (int j = 0; j < n; j++) {
              if (yolo_get_mask(eng, i, j, m.maskBuf.data()) == 0)
                rleEncode(m.maskBuf.data(), (size_t)m.maskH * m.maskW, r->resp->add_masks());
              else
                r->resp->add_masks();
            }
          }
        }
      }
      r->resp->set_queue_ms((float)ms_since(r->arrival, f.tLaunch));
      r->resp->set_infer_ms((float)inferMs);
      r->resp->set_batch(f.B);
      r->done.set_value(f.ok[i]);
    }
    long n = served += f.B;
    if (n % 5000 < f.B)
      printf("[serve] %ld imgs, B=%d gpu %.2f ms (%.0f img/s in-batch)\n",
             n, f.B, inferMs, f.B * 1000.0 / inferMs);
  }

  void worker() {
    Inflight cur, prev;
    bool havePrev = false;
    for (;;) {
      cur.model = collect(cur.batch);
      cur.B = (int)cur.batch.size();
      cur.ok.assign(cur.B, 0); cur.hh.assign(cur.B, 0); cur.ww.assign(cur.B, 0);
      auto d0 = Clock::now();
      decodeBatch(cur.batch, cur.ok, cur.hh, cur.ww);     // decoder pool || GPU running prev
      decodeEmaMs = 0.9 * decodeEmaMs + 0.1 * ms_since(d0, Clock::now());
      if (havePrev) {                                     // prev finished while we decoded
        yolo_sync(models[prev.model].eng);
        respond(prev, ms_since(prev.tLaunch, Clock::now()));
      }
      for (int i = 0; i < cur.B; i++)                     // engine stream: serializes after prev graph
        if (cur.ok[i]) {
          Model& m = models[cur.model];
          if (m.task == 2) yolo_preprocess_cls(m.eng, scratch[slotBase + i], cur.hh[i], cur.ww[i], i);
          else             yolo_preprocess(m.eng, scratch[slotBase + i], cur.hh[i], cur.ww[i], i);
        }
      cur.tLaunch = Clock::now();
      yolo_run_async(models[cur.model].eng, cur.B);
      prev = std::move(cur);
      havePrev = true;
      slotBase = maxB - slotBase;                         // ping-pong scratch set
      // opportunistic drain: if the queue is empty, finish prev now instead of waiting for traffic
      {
        std::unique_lock<std::mutex> lk(mu);
        bool empty = true;
        for (auto& m : models) empty &= m.q.empty();
        if (empty) {
          lk.unlock();
          yolo_sync(models[prev.model].eng);
          respond(prev, ms_since(prev.tLaunch, Clock::now()));
          havePrev = false;
        }
      }
    }
  }
};

class YoloService final : public yolo::Yolo::Service {
 public:
  explicit YoloService(Server* s) : s_(s) {}
  grpc::Status Detect(grpc::ServerContext*, const yolo::DetectRequest* req,
                      yolo::DetectResponse* resp) override {
    Req r;
    if (!req->model().empty()) {
      r.model = -1;
      for (int mi = 0; mi < (int)s_->models.size(); mi++)
        if (s_->models[mi].name == req->model()) { r.model = mi; break; }
      if (r.model < 0)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "unknown model: " + req->model());
    }
    r.data = &req->image();
    r.arrival = Clock::now();
    r.deadline = r.arrival + std::chrono::microseconds((long)(s_->targetMs * 1000));
    r.resp = resp;
    auto fut = r.done.get_future();
    {
      std::lock_guard<std::mutex> lk(s_->mu);
      s_->models[r.model].q.push_back(&r);
    }
    s_->cv.notify_all();
    bool ok = fut.get();
    return ok ? grpc::Status::OK
              : grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "image decode failed");
  }
 private:
  Server* s_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string addr = "0.0.0.0:50051";
  std::vector<std::string> dirs;
  int maxB = 16;
  double targetMs = 50.0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--dir" && i + 1 < argc) dirs.push_back(argv[++i]);
    else if (a == "--addr" && i + 1 < argc) addr = argv[++i];
    else if (a == "--max-batch" && i + 1 < argc) maxB = atoi(argv[++i]);
    else if (a == "--target-ms" && i + 1 < argc) targetMs = atof(argv[++i]);
    else {
      printf("usage: yoloserve [--dir build/yolo11n[:maxB]]... [--addr 0.0.0.0:50051] "
             "[--max-batch 16] [--target-ms 50]\n"
             "multiple --dir flags serve several models from one process; requests route by\n"
             "the DetectRequest.model field (dir basename), default = first --dir\n");
      return 1;
    }
  }
  if (dirs.empty()) dirs.push_back("build/yolo11n");

  Server s;
  s.targetMs = targetMs;
  s.maxB = 1;
  for (auto spec : dirs) {
    Model m;
    m.maxB = maxB;
    size_t colon = spec.rfind(':');
    if (colon != std::string::npos && colon > spec.rfind('/')) {   // per-model :maxB suffix
      m.maxB = atoi(spec.c_str() + colon + 1);
      spec = spec.substr(0, colon);
    }
    size_t slash = spec.find_last_of('/');
    m.name = slash == std::string::npos ? spec : spec.substr(slash + 1);
    m.eng = yolo_create(spec.c_str(), m.maxB);
    m.task = yolo_task(m.eng);
    m.taskName = m.task == 1 ? "obb" : m.task == 2 ? "classify" : m.task == 3 ? "segment" : "detect";
    if (m.task == 3) {
      yolo_mask_dim(m.eng, &m.maskH, &m.maskW);
      m.maskBuf.resize((size_t)m.maskH * m.maskW);
    }
    printf("model '%s': task=%s max-batch=%d\n", m.name.c_str(), m.taskName, m.maxB);
    s.maxB = std::max(s.maxB, m.maxB);
    s.models.push_back(std::move(m));
  }
  if (nvjpegCreateSimple(&s.nvj) != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "nvjpeg init failed\n"); return 1;
  }
  s.scratch.assign(2 * s.maxB, nullptr);   // ping-pong slot sets for the decode/infer pipeline
  s.scratchSz.assign(2 * s.maxB, 0);
  s.nDecoders = std::min(8u, std::max(2u, std::thread::hardware_concurrency() - 2));
  for (int i = 0; i < s.nDecoders; i++) std::thread([&] { s.decoderThread(); }).detach();
  std::thread worker([&] { s.worker(); });

  YoloService svc(&s);
  grpc::ServerBuilder b;
  b.AddListeningPort(addr, grpc::InsecureServerCredentials());
  b.SetMaxReceiveMessageSize(64 << 20);
  b.RegisterService(&svc);
  auto server = b.BuildAndStart();
  printf("yoloserve listening on %s (%zu models, target=%.0f ms)\n",
         addr.c_str(), s.models.size(), targetMs);
  server->Wait();
  worker.join();
  return 0;
}
