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

#include "trueform/core/curves.hpp"
#include "trueform/core/curves_buffer.hpp"
#include "trueform/core/paths.hpp"
#include "trueform/core/points.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"

namespace tf {
namespace ts {

/// @brief WASM-resident curves wrapper.
///
/// Holds path indices (offset-blocked) and point coordinates as wasm types.
/// Each accessor returns a copy with shared ownership.
class wasm_curves {
  wasm_offset_blocked_buffer<int, int> _paths;
  wasm_ndarray<float> _points;

public:
  wasm_curves() = default;

  auto paths() -> wasm_offset_blocked_buffer<int, int> { return _paths; }
  auto points() -> wasm_ndarray<float> { return _points; }
  auto size() const -> int { return _paths.size(); }

  auto destroy() -> void {
    _paths.destroy();
    _points.destroy();
  }

  auto is_valid() const -> bool { return _paths.is_valid(); }

  // -- Internal: trueform range views over raw data --

  auto paths_range() const { return tf::make_paths(_paths.make_range()); }

  auto points_range() const {
    return tf::make_points<3>(_points.make_range());
  }

  auto curves_range() const {
    return tf::make_curves(paths_range(), points_range());
  }

  // -- Construction from JS-side ndarrays --

  static auto create(wasm_offset_blocked_buffer<int, int> paths,
                     wasm_ndarray<float> points) -> wasm_curves {
    wasm_curves wc;
    wc._paths = std::move(paths);
    wc._points = std::move(points);
    return wc;
  }

  // -- Construction from C++ curves_buffer --

  static auto from_curves_buffer(tf::curves_buffer<int, float, 3> &&cb)
      -> wasm_curves {
    auto num_pts = cb.points_buffer().size();
    wasm_curves wc;
    wc._paths = wasm_offset_blocked_buffer<int, int>::from_buffer(
        std::move(cb.paths_buffer()));
    wc._points = wasm_ndarray<float>::from_buffer(
        std::move(cb.points_buffer().data_buffer()),
        {static_cast<int>(num_pts), 3});
    return wc;
  }
};

} // namespace ts
} // namespace tf
