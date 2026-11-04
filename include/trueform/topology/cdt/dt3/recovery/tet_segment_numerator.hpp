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
#include "../../../../exact/meta.hpp"
#include "./tet_segment_point.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The numerator over 2^bits at which the segment from site `a` to site `b`
/// places the site `p`, or `-1` where no split of that frame lands on it:
/// read along the axis the segment spans most and confirmed on every axis
/// by @ref tet_segment_point, all three read in the decision's `sites`
/// (@ref tet_decision_sites).
template <typename Sites>
auto tet_segment_numerator(const Sites &sites, typename Sites::index_type a,
                           typename Sites::index_type b,
                           typename Sites::index_type p, int bits)
    -> std::int64_t {
  using T2 = typename tf::exact::meta<typename Sites::int_type>::T2;
  const auto &pa = sites[std::size_t(a)].pt;
  const auto &pb = sites[std::size_t(b)].pt;
  const auto &pp = sites[std::size_t(p)].pt;
  std::size_t axis = 0;
  for (std::size_t k = 1; k < 3; ++k) {
    const T2 span = T2(pb[k]) - T2(pa[k]), widest = T2(pb[axis]) - T2(pa[axis]);
    if ((span < 0 ? -span : span) > (widest < 0 ? -widest : widest))
      axis = k;
  }
  const T2 span = T2(pb[axis]) - T2(pa[axis]);
  const T2 along = (T2(pp[axis]) - T2(pa[axis])) * (T2(1) << bits);
  if (along % span != T2(0))
    return -1;
  const T2 numerator = along / span;
  if (numerator <= T2(0) || numerator >= (T2(1) << bits))
    return -1;
  const auto at = tet_segment_point<T2>(pa, pb, std::int64_t(numerator), bits);
  for (std::size_t k = 0; k < 3; ++k)
    if (at[k] != T2(pp[k]) * (T2(1) << bits))
      return -1;
  return std::int64_t(numerator);
}

} // namespace tf::topology::cdt::dt3
