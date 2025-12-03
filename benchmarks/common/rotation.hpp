/**
 * Benchmark rotation utilities
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>

namespace benchmark {

template <typename T, typename PointPolicy>
auto make_rotation(tf::deg<T> angle, int axis,
                   const tf::point_like<3, PointPolicy> &pivot)
    -> tf::transformation<T, 3> {
  switch (axis) {
  case 0:
    return tf::make_rotation(angle, tf::axis<0>, pivot);
  case 1:
    return tf::make_rotation(angle, tf::axis<1>, pivot);
  default:
    return tf::make_rotation(angle, tf::axis<2>, pivot);
  }
}

} // namespace benchmark
