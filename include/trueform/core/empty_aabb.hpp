/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb.hpp"
#include <limits>

namespace tf {
template <typename T, std::size_t Dims> auto make_empty_aabb() {
  tf::aabb<T, Dims> out;
  for (std::size_t i = 0; i < Dims; ++i) {
    out.min[i] = std::numeric_limits<T>::max();
    out.max[i] = std::numeric_limits<T>::lowest();
  }
  return out;
}
} // namespace tf
