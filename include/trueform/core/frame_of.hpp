/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./points.hpp"
#include "./policy/frame.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {
template <typename Policy>
auto frame_of(const tf::points<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

template <typename Policy>
auto frame_of(const tf::polygons<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

template <typename Policy>
auto frame_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

template <typename Policy>
auto frame_of(const tf::unit_vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

template <typename Policy>
auto frame_of(const tf::segments<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

} // namespace tf
