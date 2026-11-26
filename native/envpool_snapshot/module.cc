// Optional headless binding over EnvPool 1.2.7's real CartPole and worker pool.
// Upstream source/licenses are retained by the pinned build; see README.md.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "envpool/classic_control/cartpole.h"

namespace py = pybind11;
constexpr std::size_t kRngWords = std::mt19937::state_size + 1;
constexpr char kRevision[] = "9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3";

// Rendering is intentionally unavailable, not a dummy rendered observation.
namespace classic_control::rendering {
void RenderCartPole(double, double, int, int, std::uint8_t*) {
  throw std::runtime_error("This checkpoint binding supports headless CartPole only");
}
}  // namespace classic_control::rendering

struct SlotSnapshot {
  std::array<double, 4> physics;
  std::array<std::uint32_t, kRngWords> rng;
  // seed, base step, CartPole step, done, force_reset
  std::array<std::int64_t, 5> counters;
};

class SnapshotEnv : public classic_control::CartPoleEnv {
 public:
  using CartPoleEnv::CartPoleEnv;

  SlotSnapshot Snapshot() const {
    SlotSnapshot ret{{x_, x_dot_, theta_, theta_dot_}, {},
                     {seed_, SnapshotStepCount(), elapsed_step_, done_, force_reset_}};
    std::stringstream stream;
    stream.imbue(std::locale::classic());
    stream << gen_;
    for (auto& word : ret.rng) {
      if (!(stream >> word)) throw std::runtime_error("Unsupported MT19937 export format");
    }
    stream >> std::ws;
    if (!stream.eof()) throw std::runtime_error("Unsupported MT19937 export size");
    return ret;
  }

  std::mt19937 Validate(const SlotSnapshot& value) const {
    for (double element : value.physics) {
      if (!std::isfinite(element)) throw std::invalid_argument("Non-finite snapshot physics");
    }
    const auto& c = value.counters;
    if (c[0] < 0 || c[0] > std::numeric_limits<int>::max() || c[1] < 0 ||
        c[1] > max_episode_steps_ || c[1] != c[2] || c[3] < 0 || c[3] > 1 ||
        c[4] < 0 || c[4] > 1 || (c[1] == max_episode_steps_ && !c[3])) {
      throw std::invalid_argument("Invalid snapshot counters/flags");
    }
    bool nonzero = false;
    for (std::size_t i = 0; i < kRngWords - 1; ++i) nonzero |= value.rng[i] != 0;
    if (!nonzero || value.rng.back() > std::mt19937::state_size) {
      throw std::invalid_argument("Invalid MT19937 snapshot state/index");
    }
    std::stringstream stream;
    stream.imbue(std::locale::classic());
    for (auto word : value.rng) stream << word << ' ';
    std::mt19937 generator;
    if (!(stream >> generator)) throw std::invalid_argument("Invalid MT19937 snapshot");
    return generator;
  }

  void Restore(const SlotSnapshot& value, const std::mt19937& generator) {
    x_ = value.physics[0]; x_dot_ = value.physics[1];
    theta_ = value.physics[2]; theta_dot_ = value.physics[3];
    seed_ = static_cast<int>(value.counters[0]);
    RestoreStepCount(static_cast<int>(value.counters[1]));
    elapsed_step_ = static_cast<int>(value.counters[2]);
    done_ = value.counters[3]; force_reset_ = value.counters[4];
    gen_ = generator;
  }
};

class CorePool : public AsyncEnvPool<SnapshotEnv> {
 public:
  using AsyncEnvPool::AsyncEnvPool;

  void Quiescent() {
    if (!is_sync_ || stepping_env_num_ != 0 || action_buffer_queue_->SizeApprox() != 0) {
      throw std::runtime_error("Snapshot requires a consumed synchronous batch");
    }
    // Recv already saw every Done notification; await the final worker unwind.
    while (running_env_num_ != 0) std::this_thread::yield();
  }

  std::vector<SlotSnapshot> Snapshot() {
    Quiescent();
    std::vector<SlotSnapshot> result;
    result.reserve(num_envs_);
    for (auto& env : envs_) result.push_back(env->Snapshot());
    return result;
  }

  void Restore(const std::vector<SlotSnapshot>& values) {
    Quiescent();
    std::vector<std::mt19937> generators;
    generators.reserve(num_envs_);
    for (std::size_t i = 0; i < num_envs_; ++i) generators.push_back(envs_[i]->Validate(values[i]));
    for (std::size_t i = 0; i < num_envs_; ++i) envs_[i]->Restore(values[i], generators[i]);
  }
};

class NativePool {
  std::unique_ptr<CorePool> pool_;
  std::mutex mutex_;
  bool initialized_{false};
  std::array<std::uint32_t, 4> configuration_;

  void Running() const {
    if (!pool_ || !initialized_) throw std::runtime_error("Initialize a running pool before snapshot/step");
  }

 public:
  NativePool(int num_envs, int num_threads, int episode_steps, int seed) {
    if (num_envs < 1 || num_threads < 1 || episode_steps < 1 ||
        episode_steps == std::numeric_limits<int>::max() || seed < 0 ||
        seed > std::numeric_limits<int>::max() - num_envs + 1) {
      throw std::invalid_argument("Invalid slot/thread/episode counts or seed");
    }
    configuration_ = {1, static_cast<std::uint32_t>(num_envs),
                       static_cast<std::uint32_t>(num_threads),
                       static_cast<std::uint32_t>(episode_steps)};
    classic_control::CartPoleEnvSpec spec;
    spec.config["num_envs"_] = num_envs;
    spec.config["batch_size"_] = num_envs;
    spec.config["num_threads"_] = num_threads;
    spec.config["max_episode_steps"_] = episode_steps;
    spec.config["seed"_] = seed;
    pool_ = std::make_unique<CorePool>(spec);
  }

  std::vector<Array> Reset() {
    std::lock_guard lock(mutex_);
    if (!pool_ || initialized_) throw std::runtime_error("Initialize this pool exactly once");
    Array ids(Spec<int>({static_cast<int>(configuration_[1])}));
    auto* data = static_cast<int*>(ids.Data());
    for (std::size_t i = 0; i < configuration_[1]; ++i) data[i] = i;
    pool_->Reset(ids);
    auto result = pool_->Recv();
    pool_->Quiescent();
    initialized_ = true;
    return result;
  }

  std::vector<Array> Step(const std::vector<int>& action) {
    std::lock_guard lock(mutex_);
    Running();
    if (action.size() != configuration_[1]) throw std::invalid_argument("Action shape mismatch");
    std::vector<Array> arrays;
    for (int field = 0; field < 3; ++field) {
      arrays.emplace_back(Spec<int>({static_cast<int>(action.size())}));
      auto* data = static_cast<int*>(arrays.back().Data());
      for (std::size_t i = 0; i < action.size(); ++i) {
        if (action[i] < 0 || action[i] > 1) throw std::invalid_argument("Invalid CartPole action");
        data[i] = field == 2 ? action[i] : static_cast<int>(i);
      }
    }
    pool_->Send(arrays);
    auto result = pool_->Recv();
    pool_->Quiescent();
    return result;
  }

  std::vector<SlotSnapshot> Snapshot() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) throw std::runtime_error("Snapshot refused while a batch is in flight");
    Running();
    return pool_->Snapshot();
  }

  void Restore(const std::array<std::uint32_t, 4>& configuration,
               const std::vector<SlotSnapshot>& values) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) throw std::runtime_error("Restore refused while a batch is in flight");
    Running();
    if (configuration != configuration_ || values.size() != configuration_[1]) {
      throw std::invalid_argument("Snapshot version/configuration mismatch");
    }
    pool_->Restore(values);
  }

  void Close() {
    std::lock_guard lock(mutex_);
    pool_.reset();
  }
  const auto& Configuration() const { return configuration_; }
};

template <typename T>
py::array_t<T> Copy(const Array& value) {
  py::array_t<T> result(value.Shape());
  std::memcpy(result.mutable_data(), value.Data(), value.size * sizeof(T));
  return result;
}

py::tuple Transition(const std::vector<Array>& arrays, bool reset) {
  CorePool::State state(arrays);
  py::dict info;
  info["env_id"] = Copy<int>(state["info:env_id"_]);
  info["elapsed_step"] = Copy<int>(state["elapsed_step"_]);
  auto obs = Copy<float>(state["obs"_]);
  if (reset) return py::make_tuple(obs, info);
  auto done = Copy<bool>(state["done"_]);
  auto trunc = Copy<bool>(state["trunc"_]);
  py::array_t<bool> terminated(done.size());
  for (py::ssize_t i = 0; i < done.size(); ++i) terminated.mutable_data()[i] = done.data()[i] && !trunc.data()[i];
  return py::make_tuple(obs, Copy<float>(state["reward"_]), terminated, trunc, info);
}

template <typename T>
py::array_t<T> Require(const py::array& value, const std::vector<py::ssize_t>& shape) {
  if (!value.dtype().is(py::dtype::of<T>()) || !(value.flags() & py::array::c_style) ||
      reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T) != 0 ||
      value.ndim() != static_cast<py::ssize_t>(shape.size())) {
    throw std::invalid_argument("Snapshot/action shape, dtype or layout mismatch");
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (value.shape(i) != shape[i]) throw std::invalid_argument("Snapshot/action shape mismatch");
  }
  return py::reinterpret_borrow<py::array_t<T>>(value);
}

PYBIND11_MODULE(perpetua_envpool_snapshot, m) {
  m.attr("codec") = "envpool-cartpole-mt19937-v1";
  m.attr("upstream_revision") = kRevision;
  py::class_<NativePool>(m, "CartPolePool")
    .def(py::init<int, int, int, int>(), py::arg("num_envs"), py::arg("num_threads"), py::arg("episode_steps"), py::arg("seed"))
    .def("reset", [](NativePool& self) {
      std::vector<Array> result;
      { py::gil_scoped_release release; result = self.Reset(); }
      return Transition(result, true);
    })
    .def("step", [](NativePool& self, const py::array& action) {
      auto arr = Require<int>(action, {self.Configuration()[1]});
      std::vector<int> values(arr.data(), arr.data() + arr.size());
      std::vector<Array> result;
      { py::gil_scoped_release release; result = self.Step(values); }
      return Transition(result, false);
    })
    .def("snapshot", [](NativePool& self) {
      std::vector<SlotSnapshot> values;
      { py::gil_scoped_release release; values = self.Snapshot(); }
      const auto n = static_cast<py::ssize_t>(values.size());
      py::array_t<double> physics({n, py::ssize_t(4)});
      py::array_t<std::uint32_t> rng({n, static_cast<py::ssize_t>(kRngWords)});
      py::array_t<std::int64_t> counters({n, py::ssize_t(5)});
      py::array_t<std::uint32_t> configuration(4);
      std::copy(self.Configuration().begin(), self.Configuration().end(), configuration.mutable_data());
      for (std::size_t i = 0; i < values.size(); ++i) {
        std::copy(values[i].physics.begin(), values[i].physics.end(), physics.mutable_data() + i * 4);
        std::copy(values[i].rng.begin(), values[i].rng.end(), rng.mutable_data() + i * kRngWords);
        std::copy(values[i].counters.begin(), values[i].counters.end(), counters.mutable_data() + i * 5);
      }
      return py::make_tuple(configuration, physics, rng, counters);
    })
    .def("restore", [](NativePool& self, const py::array& config, const py::array& physics,
                       const py::array& rng, const py::array& counters) {
      auto conf = Require<std::uint32_t>(config, {4});
      const auto n = static_cast<py::ssize_t>(self.Configuration()[1]);
      auto p = Require<double>(physics, {n, 4});
      auto r = Require<std::uint32_t>(rng, {n, static_cast<py::ssize_t>(kRngWords)});
      auto c = Require<std::int64_t>(counters, {n, 5});
      std::array<std::uint32_t, 4> configuration;
      std::copy_n(conf.data(), 4, configuration.data());
      std::vector<SlotSnapshot> values(n);
      for (py::ssize_t i = 0; i < n; ++i) {
        std::copy_n(p.data() + i * 4, 4, values[i].physics.data());
        std::copy_n(r.data() + i * kRngWords, kRngWords, values[i].rng.data());
        std::copy_n(c.data() + i * 5, 5, values[i].counters.data());
      }
      py::gil_scoped_release release;
      self.Restore(configuration, values);
    })
    .def("close", &NativePool::Close, py::call_guard<py::gil_scoped_release>());
}
