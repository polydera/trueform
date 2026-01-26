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
#include "./vertex_source.hpp"
#include <utility>

namespace tf::loop {
template <typename Index> struct vertex {
  static constexpr Index original_tag = -1;
  Index id;
  Index intersection_index;
  vertex_source source;

  friend auto operator==(const vertex &v0, const vertex &v1) -> bool {
    return std::make_pair(v0.source, v0.id) ==
           std::make_pair(v1.source, v1.id);
  }

  friend auto operator!=(const vertex &v0, const vertex &v1) -> bool {
    return !(v0 == v1);
  }

  friend auto operator<(const vertex &v0, const vertex &v1) -> bool {
    return std::make_pair(v0.source, v0.id) <
           std::make_pair(v1.source, v1.id);
  }
};
} // namespace tf::loop
