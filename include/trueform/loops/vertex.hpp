/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./vertex_source.hpp"
#include <utility>

namespace tf::loop {
template <typename Index> struct vertex {
  Index id;
  vertex_source source;

  friend auto operator==(const vertex &v0, const vertex &v1) -> bool {
    return std::make_pair(v0.source, v0.id) == std::make_pair(v1.source, v1.id);
  }

  friend auto operator!=(const vertex &v0, const vertex &v1) -> bool {
    return !(v0 == v1);
  }
};
} // namespace tf::loop
