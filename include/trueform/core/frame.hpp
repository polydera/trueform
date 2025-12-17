/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./frame_like.hpp"
#include "./linalg/safe_frame.hpp"

namespace tf {
template <typename T, std::size_t Dims>
using frame = tf::frame_like<Dims, tf::linalg::safe_frame<T, Dims>>;

template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_frame(
    const tf::transformation_like<Dims, Policy0> &_transformation,
    const tf::transformation_like<Dims, Policy1> &_inv_transformation) {
  return tf::frame<tf::coordinate_type<Policy0, Policy1>, Dims>{
      _transformation, _inv_transformation};
}

template <std::size_t Dims, typename Policy>
auto make_frame(const tf::transformation_like<Dims, Policy> &_transformation) {
  return tf::frame<tf::coordinate_type<Policy>, Dims>{_transformation};
}
} // namespace tf
