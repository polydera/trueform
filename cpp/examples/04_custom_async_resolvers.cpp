#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/core/async/completion.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/csg/async/outer_shell.hpp>
#include <trueform/cpp/geometry/make_box_mesh.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// A resolver state can adapt worker completion to an application's own model.
// This one invokes callbacks on the worker thread and returns a future<void>
// that only represents completion of those callbacks.
template <typename T, typename OnValue, typename OnError> class callback_state {
  std::promise<void> _done;
  OnValue _on_value;
  OnError _on_error;

public:
  callback_state(OnValue on_value, OnError on_error)
      : _on_value(std::move(on_value)), _on_error(std::move(on_error)) {}

  auto result() & -> std::future<void> { return _done.get_future(); }

  template <typename U> auto set_value(U &&value) -> void {
    _on_value(std::forward<U>(value));
    _done.set_value();
  }

  auto set_exception(std::exception_ptr exception) noexcept -> void {
    try {
      _on_error(exception);
    } catch (...) {
    }
    try {
      _done.set_exception(std::move(exception));
    } catch (...) {
    }
  }
};

template <typename OnValue, typename OnError> class callback_resolver {
  OnValue _on_value;
  OnError _on_error;

public:
  callback_resolver(OnValue on_value, OnError on_error)
      : _on_value(std::move(on_value)), _on_error(std::move(on_error)) {}

  template <typename T> using state_type = callback_state<T, OnValue, OnError>;

  template <typename T> auto make_state() && -> std::shared_ptr<state_type<T>> {
    return std::make_shared<state_type<T>>(std::move(_on_value),
                                           std::move(_on_error));
  }
};

template <typename OnValue, typename OnError>
auto callbacks(OnValue on_value, OnError on_error) {
  return callback_resolver<std::decay_t<OnValue>, std::decay_t<OnError>>(
      std::move(on_value), std::move(on_error));
}

// A completion vendor is how an integrator adapts this runtime to a host: it
// runs on the worker thread the instant an operation finishes, and its only
// job is to publish the completion or to post it where a host runtime can.
// This one defers publication to a thread of its own, which is the shape a
// UI-thread integration takes.
class deferring_vendor {
  struct state {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<tf::cpp::async::completion> pending;
  };
  std::shared_ptr<state> _state = std::make_shared<state>();

public:
  auto callback() const {
    auto held = _state;
    return [held](tf::cpp::async::completion finished) {
      {
        std::lock_guard<std::mutex> lock(held->mutex);
        held->pending.push_back(std::move(finished));
      }
      held->changed.notify_all();
    };
  }

  auto publish_one(std::chrono::milliseconds timeout) -> bool {
    std::unique_lock<std::mutex> lock(_state->mutex);
    if (!_state->changed.wait_for(lock, timeout,
                                  [&] { return !_state->pending.empty(); }))
      return false;
    auto finished = std::move(_state->pending.back());
    _state->pending.pop_back();
    lock.unlock();
    finished.publish();
    return true;
  }
};

} // namespace

int main() {
  using real = double;
  using index = tf::cpp::default_index_t;
  using result_type = tf::polygons_buffer<index, real, 3, 3>;

  std::atomic<std::size_t> callback_faces{0};
  std::atomic<bool> callback_failed{false};
  const auto callback_storage = tf::cpp::make_box_mesh(2.0, 3.0, 4.0);
  tf::cpp::cache<index, real> callback_cache;
  const tf::cpp::mesh<index, real> callback_source{
      callback_storage.faces(), callback_storage.points(), callback_cache};

  auto callback_done = tf::cpp::async::outer_shell(
      callbacks(
          [&](result_type result) {
            callback_faces.store(result.size(), std::memory_order_release);
          },
          [&](std::exception_ptr) {
            callback_failed.store(true, std::memory_order_release);
          }),
      callback_source);
  static_assert(std::is_same_v<decltype(callback_done), std::future<void>>);
  callback_done.get();

  // The completion vendor is the runtime's one seam. It belongs to whoever
  // integrates this library into a language and is never visible to that
  // language's users, who simply wait on the future submit returned.
  deferring_vendor vendor;
  tf::cpp::async::set_completion_vendor(vendor.callback());

  const auto vendor_storage = tf::cpp::make_box_mesh(1.0, 1.0, 1.0);
  tf::cpp::cache<index, real> vendor_cache;
  const tf::cpp::mesh<index, real> vendor_source{
      vendor_storage.faces(), vendor_storage.points(), vendor_cache};
  auto vendor_pending = tf::cpp::async::outer_shell(vendor_source);
  static_assert(
      std::is_same_v<decltype(vendor_pending), std::future<result_type>>);

  // Nothing is waiting on that future until the vendor publishes it.
  const auto published = vendor.publish_one(std::chrono::seconds(5));
  tf::cpp::async::set_completion_vendor(nullptr);
  if (!published)
    return 1;

  const auto vendor_shell = vendor_pending.get();
  std::cout << "callback resolver: "
            << callback_faces.load(std::memory_order_acquire)
            << " output faces\ndeferring vendor: " << vendor_shell.size()
            << " output faces\n";

  return !callback_failed.load(std::memory_order_acquire) &&
                 callback_faces.load(std::memory_order_acquire) > 0 &&
                 vendor_shell.size() > 0
             ? 0
             : 1;
}
