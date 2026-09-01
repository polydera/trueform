/*
 * Copyright (c) 2026 XLAB
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

#include "./bits/twos_complement.hpp"
#include "./meta.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// @ingroup exact
/// @brief Quantise a parameter so one step along this edge is one ulp of
///        the real type the lattice was converted from.
///
/// A split has to stay on its edge, so the only tolerance available to it
/// is measured along that edge. Expressing that tolerance as a fixed
/// fraction gives it a different physical size on every edge, and the two
/// parents of one crossing then disagree about how far apart two positions
/// must be to count as two. Deriving the step from the edge's own length
/// instead makes the tolerance one distance everywhere while leaving the
/// point exactly on the chord.
///
/// An edge shorter than one ulp has no interior at this resolution: the
/// step covers it whole, so every position on it rounds to an endpoint and
/// the degenerate edge collapses rather than acquiring a vertex.
template <typename Int>
auto edge_parameter_step_bits(const tf::exact::pt3<Int> &a,
                              const tf::exact::pt3<Int> &b) -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  T1 extent(0);
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    T1 delta = T1(b[coordinate]) - T1(a[coordinate]);
    if (delta < T1(0))
      delta = -delta;
    if (extent < delta)
      extent = delta;
  }
  int extent_bits = 0;
  while (extent > T1(0)) {
    extent >>= 1;
    ++extent_bits;
  }
  const int steps_bits = extent_bits - tf::exact::meta<Int>::conversion_ulp_bits;
  const int step_bits = tf::exact::meta<Int>::param_bits -
                        (steps_bits > 0 ? steps_bits : 0);
  return step_bits < 0 ? 0 : step_bits;
}

/// @ingroup exact
/// @brief `parameter` rounded to the nearest multiple of `2^step_bits`.
///
/// The one producer of a split's rounding. A caller that snaps a run of
/// positions on one edge asks @ref tf::exact::edge_parameter_step_bits for
/// the step once and states it here, since the step is the edge's.
template <typename Int>
auto snap_parameter_to_step(typename tf::exact::meta<Int>::param_type parameter,
                            int step_bits) ->
    typename tf::exact::meta<Int>::param_type {
  using param_t = typename tf::exact::meta<Int>::param_type;
  if (step_bits <= 0)
    return parameter;
  const unsigned bits = unsigned(step_bits);
  const param_t half = param_t(1) << (bits - 1u);
  return tf::exact::bits::align_down(parameter + half, bits);
}

/// @ingroup exact
/// @brief `parameter` rounded to the nearest step of @ref
///        tf::exact::edge_parameter_step_bits.
template <typename Int>
auto snap_parameter_to_edge(typename tf::exact::meta<Int>::param_type parameter,
                            const tf::exact::pt3<Int> &a,
                            const tf::exact::pt3<Int> &b) ->
    typename tf::exact::meta<Int>::param_type {
  return tf::exact::snap_parameter_to_step<Int>(
      parameter, tf::exact::edge_parameter_step_bits<Int>(a, b));
}

} // namespace tf::exact
