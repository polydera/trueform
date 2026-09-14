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

#include <atomic>

#ifndef NDEBUG
#include <cassert>
#endif

namespace tf::cpp::detail {

/// @brief The courtesy that turns a broken cache contract into a named assert.
///
/// A cache is not thread safe: before entering concurrent regions a caller
/// either knows its lazy use is safe, or it prebuilt what it needs with the
/// build verbs. Two threads filling one cache is that contract broken, and
/// without this it is a silent data race — so under `!NDEBUG` the second filler
/// says which contract it broke. A release build takes the flag and never reads
/// it.
///
/// It is engaged where a cache WRITES, past the freshness check and never
/// around one: reads of filled state are free and unlimited, whatever their
/// number. A fill states its dependencies before it writes, so no fill is
/// engaged inside another.
///
/// The slot is unconditional so a debug and a release translation unit agree on
/// what a cache looks like.
class fill_guard {
public:
  explicit fill_guard(std::atomic<bool> &filling) : _filling(&filling) {
#ifndef NDEBUG
    assert(!_filling->exchange(true, std::memory_order_acq_rel) &&
           "trueform cache: two threads filled one cache. A cache is filled "
           "before it is shared, or prebuilt with the build_* verbs, or "
           "synchronized by the caller");
#endif
  }
  fill_guard(const fill_guard &) = delete;
  auto operator=(const fill_guard &) -> fill_guard & = delete;
  ~fill_guard() {
#ifndef NDEBUG
    _filling->store(false, std::memory_order_release);
#else
    static_cast<void>(_filling);
#endif
  }

private:
  std::atomic<bool> *_filling;
};

} // namespace tf::cpp::detail
