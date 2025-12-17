/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/faces.hpp"
#include <algorithm>

namespace tf {
template <typename Policy>
auto reverse_winding(tf::faces<Policy> &faces) -> void {
  tf::parallel_apply(faces, [](auto &&face) {
    std::reverse(face.begin(), face.end());
  });
}

template <typename Policy>
auto reverse_winding(tf::faces<Policy> &&faces) -> void {
  reverse_winding(faces);
}
} // namespace tf
