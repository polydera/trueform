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

#include "./meta.hpp"

namespace tf::exact {

/// @ingroup exact
/// @brief A position measured on a piece of an edge, expressed on the whole
///        edge instead.
///
/// `local` runs over `[0, denominator]` along the piece; `t0` and `t1` are
/// where the piece begins and ends on its parent, in the parent's own
/// orientation, so `t1 < t0` when the piece runs backwards along it.
///
/// The single producer of that rebase. A split is only shared when every
/// carrier of an edge agrees where on it the split lies, and a position
/// stated in a piece's coordinates means nothing to a carrier holding the
/// whole edge — so this conversion is on the path of every reported split,
/// from the resolution inside a triangulation and from the refiner, and the
/// two must not round differently. It rounds half away from zero, the same
/// rule @ref tf::exact::dyadic_blend applies when it turns the rebased
/// parameter back into a coordinate.
///
/// Returns `-1` for a `local` that is already the not-a-parameter sentinel,
/// or a `denominator` that cannot define one, so an unplaceable position
/// stays unplaceable through the rebase instead of becoming a position on
/// the parent.
template <typename Int>
auto rebase_parameter(typename tf::exact::meta<Int>::param_type local,
                      typename tf::exact::meta<Int>::param_type t0,
                      typename tf::exact::meta<Int>::param_type t1,
                      typename tf::exact::meta<Int>::T2 denominator) ->
    typename tf::exact::meta<Int>::param_type {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (local < param_t(0) || denominator <= T2(0))
    return param_t(-1);
  const T2 offset = (T2(t1) - T2(t0)) * T2(local);
  const T2 half = denominator / T2(2);
  return param_t(T2(t0) +
                 (offset < T2(0) ? offset - half : offset + half) / denominator);
}

} // namespace tf::exact
