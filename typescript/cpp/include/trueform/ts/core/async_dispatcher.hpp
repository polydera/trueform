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
#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "tbb/concurrent_hash_map.h"
#include "tbb/task_group.h"

#include <emscripten/val.h>

namespace tf {
namespace ts {

/// @brief Per-operation async context allocated on the WASM heap.
///
/// The `status` field is watched by JS via `Atomics.waitAsync`.
/// Its address is stable in SharedArrayBuffer memory.
/// The `converter` marshals `std::any` → `emscripten::val` at retrieve time,
/// so one generic retrieve serves all return types.
struct async_context {
  alignas(4) int32_t status{0}; // 0=pending, 1=done, -1=error
  std::any result;
  std::function<emscripten::val(std::any &&)> converter;
};

/// @brief Generic async dispatch system for running TBB-parallel operations
/// without blocking the main thread.
///
/// Each async operation allocates an `async_context` on the WASM heap.
/// JS watches the status field via `Atomics.waitAsync` (microtask resolution).
/// Results are stored as `std::any` in a `tbb::concurrent_hash_map`.
///
/// Two embind calls per operation:
///   1. `dispatch_*(args)` — allocates context, queues TBB work, returns slot
///   2. `retrieve(slot)` — returns result as emscripten::val, frees context
class async_dispatcher {
  tbb::concurrent_hash_map<uintptr_t, std::unique_ptr<async_context>> _tasks;
  tbb::task_group _group;

public:
  /// Allocates an async_context, dispatches `fn` to TBB, returns the address
  /// of the status field (for JS `Atomics.waitAsync`).
  ///
  /// The return type of `fn` is deduced at compile time. A converter is stored
  /// in the context that knows how to cast `std::any` → `emscripten::val`
  /// for that specific type.
  template <typename F> auto dispatch(F &&fn) -> uintptr_t {
    using R = std::invoke_result_t<std::decay_t<F>>;

    auto ctx = std::make_unique<async_context>();
    ctx->converter = [](std::any &&r) -> emscripten::val {
      return emscripten::val(std::any_cast<R>(std::move(r)));
    };

    auto *raw = ctx.get();
    auto ptr = reinterpret_cast<uintptr_t>(&raw->status);

    typename decltype(_tasks)::accessor acc;
    _tasks.insert(acc, ptr);
    acc->second = std::move(ctx);
    acc.release();

    _group.run([raw, f = std::forward<F>(fn)]() {
      try {
        raw->result = f();
        __atomic_store_n(&raw->status, 1, __ATOMIC_RELEASE);
      } catch (...) {
        __atomic_store_n(&raw->status, -1, __ATOMIC_RELEASE);
      }
      __builtin_wasm_memory_atomic_notify(&raw->status, 1);
    });

    return ptr;
  }

  /// Returns the result as `emscripten::val` and frees the context.
  /// Only call after status == 1 (JS side checks via Atomics.load).
  /// Uses the stored converter — no per-type retrieve functions needed.
  auto retrieve(uintptr_t ptr) -> emscripten::val {
    typename decltype(_tasks)::accessor acc;
    _tasks.find(acc, ptr);
    auto val = acc->second->converter(std::move(acc->second->result));
    _tasks.erase(acc); // unique_ptr deletes context
    return val;
  }

  /// Waits for all pending tasks and cleans up unretrieved entries.
  ~async_dispatcher() { _group.wait(); }
};

inline async_dispatcher dispatcher;

} // namespace ts
} // namespace tf
