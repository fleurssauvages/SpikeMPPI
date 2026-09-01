#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct Job {
  const double* initial_state = nullptr;
  const double* controls = nullptr;
  const double* nominal = nullptr;
  const double* ctrl_scale = nullptr;
  double* positions = nullptr;
  double* costs = nullptr;
  double* progress = nullptr;
  bool* failed = nullptr;

  int n = 0;
  int h = 0;
  int nu = 0;
  int substeps = 1;
  int root_qadr = 0;
  int task_qadr = 0;

  double origin_x = 0.0;
  double origin_y = 0.0;
  double rot00 = 1.0;
  double rot01 = 0.0;
  double rot10 = 0.0;
  double rot11 = 1.0;
  double canonical_start_x = 0.0;
  double canonical_start_y = 0.0;
  double radius = 1.0;
  double left_arc_x = 0.0;
  double right_arc_x = 1.0;
  double center_y = 0.0;
  double straight_length = 1.0;
  double track_length = 1.0;
  double allowed_sq = 1.0;
  double min_height = 0.0;
  double min_up = -1.0;
  double upright_weight = 0.0;
  double control_deviation_weight = 0.0;
  double box_progress_weight = 1.0;
  double robot_progress_weight = 0.0;
  double robot_box_approach_weight = 0.0;
  double box_max_lift = 0.12;
  double box_min_up = 0.75;
  double current_s = 0.0;
  double current_root_s = 0.0;
  double initial_task_root_distance = 0.0;
  double initial_task_height = 0.0;
  double start_time = 0.0;
};

inline double PositiveMod(double x, double m) {
  double y = std::fmod(x, m);
  return y < 0.0 ? y + m : y;
}

inline void ProjectStadium(const Job& j, double xw, double yw,
                           double* best_s, double* best_d2) {
  const double dx = xw - j.origin_x;
  const double dy = yw - j.origin_y;
  const double px = dx * j.rot00 + dy * j.rot10 + j.canonical_start_x;
  const double py = dx * j.rot01 + dy * j.rot11 + j.canonical_start_y;

  // Bottom straight.
  const double bx = std::min(std::max(px, j.left_arc_x), j.right_arc_x);
  double ex = px - bx;
  double ey = py - (j.center_y - j.radius);
  double d2 = ex * ex + ey * ey;
  double s = bx - j.left_arc_x;

  // Right semicircle.
  constexpr double kPi = 3.141592653589793238462643383279502884;
  double theta = std::atan2(py - j.center_y, px - j.right_arc_x);
  const double right_lo = -0.5 * kPi;
  const double right_hi = 0.5 * kPi;
  theta = std::min(std::max(theta, right_lo), right_hi);
  double cx = j.right_arc_x + j.radius * std::cos(theta);
  double cy = j.center_y + j.radius * std::sin(theta);
  ex = px - cx;
  ey = py - cy;
  double candidate_d2 = ex * ex + ey * ey;
  if (candidate_d2 < d2) {
    d2 = candidate_d2;
    s = j.straight_length + j.radius * (theta + 0.5 * kPi);
  }

  // Top straight.
  const double tx = std::min(std::max(px, j.left_arc_x), j.right_arc_x);
  ex = px - tx;
  ey = py - (j.center_y + j.radius);
  candidate_d2 = ex * ex + ey * ey;
  if (candidate_d2 < d2) {
    d2 = candidate_d2;
    s = j.straight_length + kPi * j.radius + (j.right_arc_x - tx);
  }

  // Left semicircle.
  theta = std::atan2(py - j.center_y, px - j.left_arc_x);
  if (theta < 0.5 * kPi) {
    theta += 2.0 * kPi;
  }
  const double left_lo = 0.5 * kPi;
  const double left_hi = 1.5 * kPi;
  theta = std::min(std::max(theta, left_lo), left_hi);
  cx = j.left_arc_x + j.radius * std::cos(theta);
  cy = j.center_y + j.radius * std::sin(theta);
  ex = px - cx;
  ey = py - cy;
  candidate_d2 = ex * ex + ey * ey;
  if (candidate_d2 < d2) {
    d2 = candidate_d2;
    s = 2.0 * j.straight_length + kPi * j.radius
        + j.radius * (theta - 0.5 * kPi);
  }

  *best_s = PositiveMod(s, j.track_length);
  *best_d2 = d2;
}

}  // namespace

class FusedRolloutEvaluator {
 public:
  FusedRolloutEvaluator(const std::string& model_path, int nthread, int root_qpos_adr,
                        int task_qpos_adr, int chunk_size)
      : nthread_(std::max(1, nthread)),
        root_qadr_(root_qpos_adr),
        task_qadr_(task_qpos_adr),
        chunk_size_(std::max(1, chunk_size)) {
    model_ = mj_loadModel(model_path.c_str(), nullptr);
    if (!model_) {
      throw std::runtime_error("mj_loadModel failed for fused rollout model: " + model_path);
    }
    if (root_qadr_ < 0 || root_qadr_ + 7 > model_->nq ||
        task_qadr_ < 0 || task_qadr_ + 7 > model_->nq) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("invalid robot/task free-joint qpos address for fused rollout evaluator");
    }

    nstate_ = mj_stateSize(model_, mjSTATE_FULLPHYSICS);
    data_.reserve(nthread_);
    for (int i = 0; i < nthread_; ++i) {
      mjData* d = mj_makeData(model_);
      if (!d) {
        CleanupData();
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("mj_makeData failed for fused rollout worker");
      }
      // These user-input arrays are not part of FULLPHYSICS.  They are fixed
      // at zero for this evaluator and mj_step never mutates them, so initialize
      // them once instead of clearing O(nv + nbody) memory for every rollout.
      mju_zero(d->qfrc_applied, model_->nv);
      mju_zero(d->xfrc_applied, 6 * model_->nbody);
      data_.push_back(d);
    }

    // Worker 0 is the calling thread. Keep nthread-1 persistent background
    // workers to avoid thread creation/destruction on every 20 ms control tick.
    for (int tid = 1; tid < nthread_; ++tid) {
      workers_.emplace_back([this, tid]() { WorkerLoop(tid); });
    }
  }

  ~FusedRolloutEvaluator() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    work_cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    CleanupData();
    if (model_) {
      mj_deleteModel(model_);
      model_ = nullptr;
    }
  }

  FusedRolloutEvaluator(const FusedRolloutEvaluator&) = delete;
  FusedRolloutEvaluator& operator=(const FusedRolloutEvaluator&) = delete;

  py::tuple Evaluate(
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_state,
      py::array_t<double, py::array::c_style | py::array::forcecast> controls,
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal,
      py::array_t<double, py::array::c_style | py::array::forcecast> ctrl_scale,
      py::array_t<double, py::array::c_style | py::array::forcecast> params,
      int control_substeps) {
    const auto state_info = initial_state.request();
    const auto ctrl_info = controls.request();
    const auto nominal_info = nominal.request();
    const auto scale_info = ctrl_scale.request();
    const auto param_info = params.request();

    if (state_info.ndim != 1 || state_info.shape[0] != nstate_) {
      throw std::runtime_error("fused initial_state has incorrect FULLPHYSICS size");
    }
    if (ctrl_info.ndim != 3) {
      throw std::runtime_error("fused controls must have shape [N,H,nu]");
    }
    const int n = static_cast<int>(ctrl_info.shape[0]);
    const int h = static_cast<int>(ctrl_info.shape[1]);
    const int nu = static_cast<int>(ctrl_info.shape[2]);
    if (n <= 0 || h <= 0 || nu != model_->nu) {
      throw std::runtime_error("fused controls dimensions do not match MuJoCo model");
    }
    if (nominal_info.ndim != 2 || nominal_info.shape[0] != h ||
        nominal_info.shape[1] != nu) {
      throw std::runtime_error("fused nominal controls must have shape [H,nu]");
    }
    if (scale_info.ndim != 1 || scale_info.shape[0] != nu) {
      throw std::runtime_error("fused ctrl_scale must have shape [nu]");
    }
    if (param_info.ndim != 1 || param_info.shape[0] != 29) {
      throw std::runtime_error("fused evaluator expected 29 track/cost parameters");
    }

    py::array_t<double> positions({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(h),
                                   static_cast<py::ssize_t>(2)});
    py::array_t<double> costs({static_cast<py::ssize_t>(n)});
    py::array_t<double> progress({static_cast<py::ssize_t>(n)});
    py::array_t<bool> failed({static_cast<py::ssize_t>(n)});

    const double* p = static_cast<const double*>(param_info.ptr);
    Job job;
    job.initial_state = static_cast<const double*>(state_info.ptr);
    job.controls = static_cast<const double*>(ctrl_info.ptr);
    job.nominal = static_cast<const double*>(nominal_info.ptr);
    job.ctrl_scale = static_cast<const double*>(scale_info.ptr);
    job.positions = positions.mutable_data();
    job.costs = costs.mutable_data();
    job.progress = progress.mutable_data();
    job.failed = failed.mutable_data();
    job.n = n;
    job.h = h;
    job.nu = nu;
    job.substeps = std::max(1, control_substeps);
    job.root_qadr = root_qadr_;
    job.task_qadr = task_qadr_;
    job.origin_x = p[0];
    job.origin_y = p[1];
    job.rot00 = p[2];
    job.rot01 = p[3];
    job.rot10 = p[4];
    job.rot11 = p[5];
    job.canonical_start_x = p[6];
    job.canonical_start_y = p[7];
    job.radius = p[8];
    job.left_arc_x = p[9];
    job.right_arc_x = p[10];
    job.center_y = p[11];
    job.straight_length = p[12];
    job.track_length = p[13];
    job.allowed_sq = p[14];
    job.min_height = p[15];
    job.min_up = p[16];
    job.upright_weight = p[17];
    job.control_deviation_weight = p[18];
    job.box_progress_weight = p[19];
    job.robot_progress_weight = p[20];
    job.robot_box_approach_weight = p[21];
    job.box_max_lift = p[22];
    job.box_min_up = p[23];
    job.current_s = p[24];
    job.current_root_s = p[25];
    job.initial_task_root_distance = p[26];
    job.initial_task_height = p[27];
    job.start_time = p[28];

    {
      py::gil_scoped_release release;
      RunJob(job);
    }

    return py::make_tuple(std::move(positions), std::move(costs),
                          std::move(progress), std::move(failed));
  }

  int nthread() const { return nthread_; }
  int nstate() const { return nstate_; }
  int chunk_size() const { return chunk_size_; }

 private:
  void CleanupData() {
    for (mjData* d : data_) {
      if (d) {
        mj_deleteData(d);
      }
    }
    data_.clear();
  }

  void RunJob(const Job& job) {
    if (nthread_ == 1) {
      current_job_ = &job;
      next_.store(0, std::memory_order_relaxed);
      ProcessAvailable(0);
      current_job_ = nullptr;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_job_ = &job;
      next_.store(0, std::memory_order_relaxed);
      finished_workers_ = 0;
      ++generation_;
    }
    work_cv_.notify_all();

    // Calling thread participates as worker 0.
    ProcessAvailable(0);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this]() {
        return finished_workers_ == static_cast<int>(workers_.size());
      });
      current_job_ = nullptr;
    }
  }

  void WorkerLoop(int tid) {
    std::uint64_t seen_generation = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this, &seen_generation]() {
          return stop_ || generation_ != seen_generation;
        });
        if (stop_) {
          return;
        }
        seen_generation = generation_;
      }

      ProcessAvailable(tid);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++finished_workers_;
        if (finished_workers_ == static_cast<int>(workers_.size())) {
          done_cv_.notify_one();
        }
      }
    }
  }

  void ProcessAvailable(int tid) {
    const Job* job = current_job_;
    if (!job) {
      return;
    }
    for (;;) {
      const int begin = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
      if (begin >= job->n) {
        break;
      }
      const int end = std::min(job->n, begin + chunk_size_);
      for (int i = begin; i < end; ++i) {
        EvaluateOne(*job, i, data_[tid]);
      }
    }
  }

  void EvaluateOne(const Job& j, int i, mjData* d) const {
    // Mirror mujoco.rollout's per-trajectory FULLPHYSICS + zero-warmstart
    // semantics without mj_resetData. Applied forces, mocap inputs and equality
    // activation are immutable inside this evaluator and were initialized by
    // mj_makeData, so repeating those O(nbody) resets made obstacle-rich models
    // slower even before the first collision query.
    mj_setState(model_, d, j.initial_state, mjSTATE_FULLPHYSICS);
    mju_zero(d->qacc_warmstart, model_->nv);
    for (int w = 0; w < mjNWARNING; ++w) {
      d->warning[w].number = 0;
    }

    const double* controls_i = j.controls + static_cast<std::size_t>(i) * j.h * j.nu;
    double* positions_i = j.positions + static_cast<std::size_t>(i) * j.h * 2;

    double s_prev = j.current_s;
    double root_s_prev = j.current_root_s;
    double cumulative = 0.0;
    double root_cumulative = 0.0;
    double prefix_sum = 0.0;
    double root_prefix_sum = 0.0;
    double approach_prefix_sum = 0.0;
    double upright_cost = 0.0;
    double control_cost = 0.0;
    double prev_time = j.start_time;
    bool failed = false;
    int last_t = -1;

    const double half_track = 0.5 * j.track_length;
    for (int t = 0; t < j.h; ++t) {
      const double* u = controls_i + static_cast<std::size_t>(t) * j.nu;
      for (int k = 0; k < j.nu; ++k) {
        d->ctrl[k] = static_cast<mjtNum>(u[k]);
      }
      // Match stock mujoco.rollout warning semantics.  It stops stepping after
      // a MuJoCo warning and repeats the last state for the rest of the rollout.
      // Here we do not materialize those repeated states, but the unchanged-time
      // failure check below produces the same infinite-cost outcome.
      bool warning_stalled = false;
      for (int sub = 0; sub < j.substeps; ++sub) {
        for (int w = 0; w < mjNWARNING; ++w) {
          if (d->warning[w].number) {
            warning_stalled = true;
            break;
          }
        }
        if (warning_stalled) {
          break;
        }
        mj_step(model_, d);
      }

      // Progress belongs to the task body (robot for racing, box for pushing).
      // Stability constraints always belong to the locomotion robot root.
      const double xw = d->qpos[j.task_qadr + 0];
      const double yw = d->qpos[j.task_qadr + 1];
      const double z = d->qpos[j.root_qadr + 2];
      const double qx = d->qpos[j.root_qadr + 4];
      const double qy = d->qpos[j.root_qadr + 5];
      const double up = 1.0 - 2.0 * (qx * qx + qy * qy);
      positions_i[2 * t + 0] = xw;
      positions_i[2 * t + 1] = yw;
      last_t = t;

      double best_s = 0.0;
      double best_d2 = 0.0;
      ProjectStadium(j, xw, yw, &best_s, &best_d2);

      // In box-pushing mode the task body and locomotion root are distinct.
      // The box drives progress, but both box and robot must remain on-road.
      double root_d2 = best_d2;
      double root_s = best_s;
      if (j.task_qadr != j.root_qadr) {
        ProjectStadium(j, d->qpos[j.root_qadr + 0], d->qpos[j.root_qadr + 1],
                       &root_s, &root_d2);
      }

      if (j.task_qadr != j.root_qadr) {
        const double task_z = d->qpos[j.task_qadr + 2];
        const double task_qx = d->qpos[j.task_qadr + 4];
        const double task_qy = d->qpos[j.task_qadr + 5];
        const double task_up = 1.0 - 2.0 * (task_qx * task_qx + task_qy * task_qy);
        if (task_z > j.initial_task_height + j.box_max_lift ||
            task_up < j.box_min_up) {
          failed = true;
          break;
        }
      }

      if (best_d2 > j.allowed_sq || root_d2 > j.allowed_sq ||
          z < j.min_height || up < j.min_up ||
          d->time <= prev_time + 1e-15) {
        failed = true;
        break;
      }

      double ds = best_s - s_prev;
      if (ds > half_track) {
        ds -= j.track_length;
      } else if (ds < -half_track) {
        ds += j.track_length;
      }
      cumulative += ds;
      prefix_sum += cumulative;
      s_prev = best_s;

      if (j.task_qadr != j.root_qadr) {
        double root_ds = root_s - root_s_prev;
        if (root_ds > half_track) {
          root_ds -= j.track_length;
        } else if (root_ds < -half_track) {
          root_ds += j.track_length;
        }
        root_cumulative += root_ds;
        const double coupled = std::min(root_cumulative, std::max(cumulative, 0.0));
        root_prefix_sum += coupled;
        root_s_prev = root_s;

        const double dx_rb = d->qpos[j.task_qadr + 0] - d->qpos[j.root_qadr + 0];
        const double dy_rb = d->qpos[j.task_qadr + 1] - d->qpos[j.root_qadr + 1];
        const double dist_rb = std::sqrt(dx_rb * dx_rb + dy_rb * dy_rb);
        approach_prefix_sum += j.initial_task_root_distance - dist_rb;
      }

      prev_time = d->time;

      double du_sq_sum = 0.0;
      const double* unom = j.nominal + static_cast<std::size_t>(t) * j.nu;
      for (int k = 0; k < j.nu; ++k) {
        const double scaled = (u[k] - unom[k]) / j.ctrl_scale[k];
        du_sq_sum += scaled * scaled;
      }
      control_cost += j.control_deviation_weight * (du_sq_sum / std::max(1, j.nu));
      const double err_up = 1.0 - up;
      upright_cost += j.upright_weight * err_up * err_up;
    }

    // Only best_rollout is used for visualization/diagnostics. For a failed
    // rollout, avoid wasting configured-integrator work after failure and hold its last XY.
    if (failed && last_t >= 0) {
      const double x = positions_i[2 * last_t + 0];
      const double y = positions_i[2 * last_t + 1];
      for (int t = last_t + 1; t < j.h; ++t) {
        positions_i[2 * t + 0] = x;
        positions_i[2 * t + 1] = y;
      }
    }

    j.progress[i] = cumulative;
    j.failed[i] = failed;
    if (failed) {
      j.costs[i] = std::numeric_limits<double>::infinity();
    } else {
      const double inv_h = 1.0 / std::max(1, j.h);
      double progress_cost = -j.box_progress_weight * prefix_sum * inv_h;
      if (j.task_qadr != j.root_qadr) {
        progress_cost -= j.robot_progress_weight * root_prefix_sum * inv_h;
        progress_cost -= j.robot_box_approach_weight * approach_prefix_sum * inv_h;
      }
      j.costs[i] = progress_cost + upright_cost + control_cost;
    }
  }

  mjModel* model_ = nullptr;
  std::vector<mjData*> data_;
  int nthread_ = 1;
  int root_qadr_ = 0;
  int task_qadr_ = 0;
  int chunk_size_ = 1;
  int nstate_ = 0;

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;
  std::uint64_t generation_ = 0;
  int finished_workers_ = 0;
  const Job* current_job_ = nullptr;
  std::atomic<int> next_{0};
};

PYBIND11_MODULE(_fused_mujoco, m) {
  m.doc() = "Fused MuJoCo rollout + stadium cost evaluator";
  py::class_<FusedRolloutEvaluator>(m, "FusedRolloutEvaluator")
      .def(py::init<const std::string&, int, int, int, int>(),
           py::arg("model_path"), py::arg("nthread"),
           py::arg("root_qpos_adr"), py::arg("task_qpos_adr"),
           py::arg("chunk_size") = 1)
      .def("evaluate", &FusedRolloutEvaluator::Evaluate,
           py::arg("initial_state"), py::arg("controls"),
           py::arg("nominal_controls"), py::arg("ctrl_scale"),
           py::arg("params"), py::arg("control_substeps"))
      .def_property_readonly("nthread", &FusedRolloutEvaluator::nthread)
      .def_property_readonly("nstate", &FusedRolloutEvaluator::nstate)
      .def_property_readonly("chunk_size", &FusedRolloutEvaluator::chunk_size);
}
