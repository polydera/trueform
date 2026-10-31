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
#include "../../exact/orient3d.hpp"
#include "../../exact/plane_support.hpp"
#include "./hole_lattice.hpp"
#include "./hole_rim.hpp"

namespace tf::fill {

/// Whether a prepared rim lies exactly in one plane: the plane its own sites
/// support, every site at orientation zero against it on the input's lattice.
/// A rim whose sites span no plane is not planar.
template <typename Index, typename Int, typename RealT>
auto hole_rim_is_planar(const tf::fill::hole_rim<Index, Int, RealT> &rim)
    -> bool {
  tf::exact::plane_support<Int> support;
  for (const auto &site : rim.sites) {
    support.offer(tf::fill::hole_unscaled_point<Int>(site.pt));
    if (support.size == 3)
      break;
  }
  if (support.size != 3)
    return false;
  const tf::exact::orient3d_plane<Int> plane{support.point[0], support.normal};
  for (const auto &site : rim.sites)
    if (tf::exact::orient3d_plane_sign(
            plane, tf::fill::hole_unscaled_point<Int>(site.pt)) != 0)
      return false;
  return true;
}

} // namespace tf::fill
