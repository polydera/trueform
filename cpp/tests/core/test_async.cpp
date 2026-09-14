/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include "trueform/cpp/core/async/completion.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

class worker_failure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct int_callable {
  auto operator()() const -> int { return 1; }
};

template <typename Predicate>
auto wait_until(Predicate &&predicate,
                std::chrono::seconds timeout = std::chrono::seconds(5))
    -> bool {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct protocol_observation {
  std::atomic<std::size_t> result_calls{0};
  std::atomic<std::size_t> value_calls{0};
  std::atomic<std::size_t> exception_calls{0};
  std::atomic<bool> state_destroyed{false};
};

class immovable_state {
  std::promise<int> _promise;
  std::shared_ptr<protocol_observation> _observation;

public:
  explicit immovable_state(std::shared_ptr<protocol_observation> observation)
      : _observation(std::move(observation)) {}
  immovable_state(const immovable_state &) = delete;
  auto operator=(const immovable_state &) -> immovable_state & = delete;
  immovable_state(immovable_state &&) = delete;
  auto operator=(immovable_state &&) -> immovable_state & = delete;
  ~immovable_state() {
    _observation->state_destroyed.store(true, std::memory_order_release);
  }

  auto result() & -> std::future<int> {
    _observation->result_calls.fetch_add(1, std::memory_order_relaxed);
    return _promise.get_future();
  }

  auto set_value(int value) -> void {
    _observation->value_calls.fetch_add(1, std::memory_order_relaxed);
    _promise.set_value(value);
  }

  auto set_exception(std::exception_ptr exception) noexcept -> void {
    _observation->exception_calls.fetch_add(1, std::memory_order_relaxed);
    try {
      _promise.set_exception(std::move(exception));
    } catch (...) {
    }
  }
};

class temporary_move_only_resolver {
  std::unique_ptr<int> _token = std::make_unique<int>(1);
  std::shared_ptr<std::atomic<bool>> _resolver_destroyed;
  std::shared_ptr<protocol_observation> _observation;

public:
  template <typename T> using state_type = immovable_state;

  temporary_move_only_resolver(
      std::shared_ptr<std::atomic<bool>> resolver_destroyed,
      std::shared_ptr<protocol_observation> observation)
      : _resolver_destroyed(std::move(resolver_destroyed)),
        _observation(std::move(observation)) {}
  temporary_move_only_resolver(const temporary_move_only_resolver &) = delete;
  auto operator=(const temporary_move_only_resolver &)
      -> temporary_move_only_resolver & = delete;
  temporary_move_only_resolver(temporary_move_only_resolver &&) = default;
  auto operator=(temporary_move_only_resolver &&)
      -> temporary_move_only_resolver & = default;
  ~temporary_move_only_resolver() {
    if (_resolver_destroyed)
      _resolver_destroyed->store(true, std::memory_order_release);
  }

  template <typename T> auto make_state() && -> std::shared_ptr<state_type<T>> {
    static_assert(std::is_same<T, int>::value, "test resolver supports int");
    return std::make_shared<state_type<T>>(_observation);
  }
};

class noncopyable_lvalue_resolver {
  std::shared_ptr<protocol_observation> _observation;

public:
  template <typename T> using state_type = immovable_state;

  explicit noncopyable_lvalue_resolver(
      std::shared_ptr<protocol_observation> observation)
      : _observation(std::move(observation)) {}
  noncopyable_lvalue_resolver(const noncopyable_lvalue_resolver &) = delete;
  auto operator=(const noncopyable_lvalue_resolver &)
      -> noncopyable_lvalue_resolver & = delete;

  template <typename T> auto make_state() & -> std::shared_ptr<state_type<T>> {
    static_assert(std::is_same<T, int>::value, "test resolver supports int");
    return std::make_shared<state_type<T>>(_observation);
  }
};

class throwing_move_callable {
public:
  throwing_move_callable() = default;
  throwing_move_callable(const throwing_move_callable &) = delete;
  throwing_move_callable(throwing_move_callable &&) {
    throw worker_failure("callable move failed");
  }

  auto operator()() -> int { return 1; }
};

class throwing_result {
public:
  throwing_result() = default;
  throwing_result(const throwing_result &) = default;
  throwing_result(throwing_result &&) {
    throw worker_failure("set value failed");
  }
};

static_assert(std::is_same<decltype(tf::cpp::async::submit(int_callable{})),
                           std::future<int>>::value,
              "default async submission returns the exact future type");
static_assert(
    std::is_same<
        tf::cpp::async::resolver_state_t<temporary_move_only_resolver, int>,
        immovable_state>::value,
    "resolver state aliases use Resolver::state_type<T>");
static_assert(
    std::is_same<
        tf::cpp::async::resolver_result_t<temporary_move_only_resolver, int>,
        std::future<int>>::value,
    "resolver result aliases use an lvalue state result method");

} // namespace

TEST_CASE("future resolver supports values waiting and move-only work",
          "[cpp][core][async]") {
  auto value = tf::cpp::async::submit([] { return std::string("complete"); });
  CHECK(value.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  value.wait();
  CHECK(value.get() == "complete");

  auto move_only =
      tf::cpp::async::submit([input = std::make_unique<int>(42)]() mutable {
        return std::move(input);
      });
  static_assert(std::is_same<decltype(move_only),
                             std::future<std::unique_ptr<int>>>::value,
                "move-only results preserve their exact future type");
  REQUIRE(move_only.get() != nullptr);
}

TEST_CASE("future resolver supports void and propagates worker exceptions",
          "[cpp][core][async]") {
  std::atomic<bool> invoked{false};
  auto void_result = tf::cpp::async::submit(
      [&invoked] { invoked.store(true, std::memory_order_release); });
  static_assert(std::is_same<decltype(void_result), std::future<void>>::value,
                "void work returns future<void>");
  CHECK_NOTHROW(void_result.get());
  CHECK(invoked.load(std::memory_order_acquire));

  auto failure = tf::cpp::async::submit(
      []() -> int { throw worker_failure("worker failed"); });
  try {
    static_cast<void>(failure.get());
    FAIL("worker exception was not propagated");
  } catch (const worker_failure &error) {
    CHECK(std::string(error.what()) == "worker failed");
  }
}

TEST_CASE("post-handle callable move failures resolve every state type",
          "[cpp][core][async]") {
  auto native = tf::cpp::async::submit(throwing_move_callable{});
  CHECK_THROWS_AS(native.get(), worker_failure);

  auto observation = std::make_shared<protocol_observation>();
  noncopyable_lvalue_resolver resolver(observation);
  auto custom = tf::cpp::async::submit<int>(resolver, throwing_move_callable{});
  CHECK_THROWS_AS(custom.get(), worker_failure);
  CHECK(observation->result_calls.load(std::memory_order_relaxed) == 1);
  CHECK(observation->value_calls.load(std::memory_order_relaxed) == 0);
  CHECK(observation->exception_calls.load(std::memory_order_relaxed) == 1);
}

TEST_CASE("throwing set_value paths terminally publish their exception",
          "[cpp][core][async]") {
  auto native = tf::cpp::async::submit([]() -> throwing_result { return {}; });
  CHECK_THROWS_AS(native.get(), worker_failure);
}

TEST_CASE("future state duplicate exceptions are nonthrowing and first-winner",
          "[cpp][core][async]") {
  auto state = tf::cpp::async::future_resolver{}.make_state<int>();
  static_assert(noexcept(state->set_exception(std::exception_ptr{})),
                "future set_exception is noexcept");
  auto result = state->result();
  state->set_exception(
      std::make_exception_ptr(worker_failure("first future failure")));
  CHECK_NOTHROW(state->set_exception(
      std::make_exception_ptr(worker_failure("second future failure"))));
  try {
    static_cast<void>(result.get());
    FAIL("future did not preserve its first failure");
  } catch (const worker_failure &error) {
    CHECK(std::string(error.what()) == "first future failure");
  }
}

TEST_CASE("temporary move-only resolvers hand worker-owned state to submit",
          "[cpp][core][async]") {
  auto resolver_destroyed = std::make_shared<std::atomic<bool>>(false);
  auto observation = std::make_shared<protocol_observation>();
  std::promise<void> release;
  auto release_future = release.get_future().share();

  auto result = tf::cpp::async::submit<int>(
      temporary_move_only_resolver(resolver_destroyed, observation),
      [release_future] {
        release_future.wait();
        return 17;
      });

  CHECK(resolver_destroyed->load(std::memory_order_acquire));
  CHECK(observation->result_calls.load(std::memory_order_relaxed) == 1);
  CHECK_FALSE(observation->state_destroyed.load(std::memory_order_acquire));
  release.set_value();
  CHECK(result.get() == 17);
  CHECK(wait_until([observation] {
    return observation->state_destroyed.load(std::memory_order_acquire);
  }));
}

TEST_CASE("noncopyable lvalue resolvers support lvalue-only immovable states",
          "[cpp][core][async]") {
  auto observation = std::make_shared<protocol_observation>();
  noncopyable_lvalue_resolver resolver(observation);

  auto result = tf::cpp::async::submit<int>(resolver, [] { return 29; });

  CHECK(result.get() == 29);
  CHECK(observation->result_calls.load(std::memory_order_relaxed) == 1);
  CHECK(observation->value_calls.load(std::memory_order_relaxed) == 1);
  CHECK(observation->exception_calls.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("a submitted future carries its value and its failure",
          "[cpp][core][async][future]") {
  auto value = tf::cpp::async::submit([] { return 23; });
  REQUIRE(value.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  CHECK(value.get() == 23);

  auto failure = tf::cpp::async::submit(
      []() -> int { throw worker_failure("submitted failure"); });
  CHECK_THROWS_AS(failure.get(), worker_failure);

  auto nothing =
      tf::cpp::async::submit<void>(tf::cpp::async::future_resolver{}, [] {});
  static_assert(std::is_same_v<decltype(nothing), std::future<void>>);
  CHECK_NOTHROW(nothing.get());
}

TEST_CASE("a future polls without blocking and delivers exactly once",
          "[cpp][core][async][future]") {
  auto pending = tf::cpp::async::submit([] { return 7; });
  CHECK(wait_until([&] {
    return pending.wait_for(std::chrono::seconds(0)) ==
           std::future_status::ready;
  }));
  CHECK(pending.valid());
  CHECK(pending.get() == 7);
  // the shared state is handed over once; the future knows it has nothing left
  CHECK_FALSE(pending.valid());
}

TEST_CASE("a dropped future neither leaks nor blocks its worker",
          "[cpp][core][async][future]") {
  auto observation = std::make_shared<protocol_observation>();
  {
    auto dropped = tf::cpp::async::submit([observation] {
      observation->value_calls.fetch_add(1, std::memory_order_release);
      return 1;
    });
  }
  CHECK(wait_until([&] {
    return observation->value_calls.load(std::memory_order_acquire) == 1;
  }));

  // the work still runs to completion, and dropping its future is not an error
  auto after = tf::cpp::async::submit([] { return 2; });
  CHECK(after.get() == 2);
}

TEST_CASE("the completion vendor sees every finished operation once",
          "[cpp][core][async][vendor]") {
  auto seen = std::make_shared<std::atomic<std::size_t>>(0);
  tf::cpp::async::set_completion_vendor(
      [seen](tf::cpp::async::completion finished) {
        seen->fetch_add(1, std::memory_order_release);
        finished.publish();
      });

  auto value = tf::cpp::async::submit([] { return 5; });
  CHECK(value.get() == 5);
  auto failure = tf::cpp::async::submit(
      []() -> int { throw worker_failure("vendored failure"); });
  CHECK_THROWS_AS(failure.get(), worker_failure);

  CHECK(seen->load(std::memory_order_acquire) == 2);
  tf::cpp::async::set_completion_vendor(nullptr);
}

TEST_CASE("a vendor may publish a completion from another thread",
          "[cpp][core][async][vendor]") {
  auto held = std::make_shared<std::vector<tf::cpp::async::completion>>();
  auto mutex = std::make_shared<std::mutex>();
  tf::cpp::async::set_completion_vendor(
      [held, mutex](tf::cpp::async::completion finished) {
        std::lock_guard<std::mutex> lock(*mutex);
        held->push_back(std::move(finished));
      });

  auto pending = tf::cpp::async::submit([] { return 11; });
  CHECK(wait_until([&] {
    std::lock_guard<std::mutex> lock(*mutex);
    return held->size() == 1;
  }));
  // the vendor held the completion, so nothing is waiting on it yet
  CHECK(pending.wait_for(std::chrono::seconds(0)) ==
        std::future_status::timeout);

  std::thread publisher([held, mutex] {
    std::lock_guard<std::mutex> lock(*mutex);
    (*held)[0].publish();
    // publishing twice is a no-op, so a vendor never tracks whether it did
    (*held)[0].publish();
  });
  publisher.join();
  CHECK(pending.get() == 11);

  tf::cpp::async::set_completion_vendor(nullptr);
  held->clear();
}

TEST_CASE(
    "dropping a vendor's held work breaks its futures rather than hanging",
    "[cpp][core][async][vendor]") {
  auto held = std::make_shared<std::vector<tf::cpp::async::completion>>();
  auto mutex = std::make_shared<std::mutex>();
  tf::cpp::async::set_completion_vendor(
      [held, mutex](tf::cpp::async::completion finished) {
        std::lock_guard<std::mutex> lock(*mutex);
        held->push_back(std::move(finished));
      });

  std::vector<std::future<int>> pending;
  for (int i = 0; i != 4; ++i)
    pending.push_back(tf::cpp::async::submit([i] { return i; }));
  CHECK(wait_until([&] {
    std::lock_guard<std::mutex> lock(*mutex);
    return held->size() == pending.size();
  }));

  // a vendor that goes away without publishing takes the outcome with it, so
  // its waiters learn the work is gone instead of waiting on it forever
  tf::cpp::async::set_completion_vendor(nullptr);
  {
    std::lock_guard<std::mutex> lock(*mutex);
    held->clear();
  }
  for (auto &future : pending)
    CHECK_THROWS_AS(future.get(), std::future_error);

  CHECK(tf::cpp::async::submit([] { return 3; }).get() == 3);
}

TEST_CASE("active workers can submit nested work resolved through futures",
          "[cpp][core][async]") {
  std::promise<void> outer_started;
  auto outer_started_future = outer_started.get_future();
  std::promise<void> release_outer;
  auto release_outer_future = release_outer.get_future().share();
  auto outer = tf::cpp::async::submit(
      [started = std::move(outer_started), release_outer_future]() mutable {
        started.set_value();
        release_outer_future.wait();
        return tf::cpp::async::submit([] { return 41; });
      });
  REQUIRE(outer_started_future.wait_for(std::chrono::seconds(5)) ==
          std::future_status::ready);

  release_outer.set_value();
  REQUIRE(outer.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  auto nested = outer.get();
  REQUIRE(nested.wait_for(std::chrono::seconds(5)) ==
          std::future_status::ready);
  CHECK(nested.get() == 41);
}
