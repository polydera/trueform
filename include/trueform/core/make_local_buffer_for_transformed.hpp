/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./local_buffer.hpp"
#include "./none.hpp"
#include "./polygon.hpp"
#include "./static_size.hpp"
#include "./transformed.hpp"

namespace tf::core {

/// Default: returns none (no buffer needed)
template <typename T, typename Transform>
auto make_local_buffer_for_transformed(const T &, const Transform &) {
  return tf::none;
}

/// Specialization for polygon: returns local_buffer for dynamic, none for
/// static
template <std::size_t Dims, typename Policy, typename Transform>
auto make_local_buffer_for_transformed(const tf::polygon<Dims, Policy> &poly,
                                       const Transform &transform) {
  if constexpr (tf::static_size_v<tf::polygon<Dims, Policy>> ==
                tf::dynamic_size) {
    using point_t = std::decay_t<decltype(tf::transformed(poly[0], transform))>;
    return tf::local_buffer<point_t>{};
  } else {
    return tf::none;
  }
}

} // namespace tf::core
