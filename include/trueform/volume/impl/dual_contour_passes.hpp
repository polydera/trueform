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
#include <chrono>

namespace tf {
namespace volume_detail {

/// @brief The passes a dual-contouring build runs, in the order it runs them.
enum class dc_pass {
  classify,       ///< the shared x-edge classification
  count_rows,     ///< per cell row: its trim, its vertices, its quads
  row_prefix,     ///< the serial scan that gives every row its bases
  fill_active,    ///< the compact record of every cell that carries surface
  marked_faces,   ///< the faces whose two arcs would collapse into one edge
  arc_prefix,     ///< the arc identities, the touched quads, the output prefix
  allocate,       ///< the output points and faces
  patch_vertices, ///< one vertex per surface component of every active cell
  arc_vertices,   ///< the two vertices of every marked face
  emit,           ///< the triangles, and the fan centre of every touched quad
  refine          ///< the feature gate and the refit it admits
};

constexpr int k_dc_pass_count = static_cast<int>(dc_pass::refine) + 1;

/// @brief What a pass is called, for a consumer that reports the table.
inline auto dc_pass_name(dc_pass pass) -> const char * {
  static const char *const names[k_dc_pass_count] = {
      "classify",       "count_rows",   "row_prefix",     "fill_active",
      "marked_faces",   "arc_prefix",   "allocate",       "patch_vertices",
      "arc_vertices",   "emit",         "refine"};
  return names[static_cast<int>(pass)];
}

/// @brief The lap clock a build carries when a consumer asked for the pass
/// table: @ref mark closes the pass that just ran.
///
/// A build that did not ask holds no table and reads no clock, so the
/// attribution costs a build exactly what it requested and nothing else.
class dc_pass_clock {
public:
  dc_pass_clock(bool on, double *times) : _times(on ? times : nullptr) {
    if (_times == nullptr)
      return;
    for (int i = 0; i < k_dc_pass_count; ++i)
      _times[i] = 0.0;
    _last = std::chrono::steady_clock::now();
  }

  auto mark(dc_pass pass) -> void {
    if (_times == nullptr)
      return;
    const auto now = std::chrono::steady_clock::now();
    _times[static_cast<int>(pass)] =
        std::chrono::duration<double, std::milli>(now - _last).count();
    _last = now;
  }

private:
  double *_times;
  std::chrono::steady_clock::time_point _last{};
};

} // namespace volume_detail
} // namespace tf
