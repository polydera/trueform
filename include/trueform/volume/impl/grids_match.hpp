/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/coordinate_dims.hpp"
#include "../volume.hpp"
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

// Do two grids name the same samples? Equal dims, and spacing and origin
// within an absolute 1e-6 — which is the tolerance the answer can afford in
// only one direction: a false negative costs a resample, correct but slower,
// while a false positive is a sub-1e-6 misregistration on a voxel-wise
// combine, which no topology reads. A match is combined voxel-wise with no
// interpolation.
template <typename Real, typename Policy0, typename Policy1>
inline auto grids_match(const tf::volume<Policy0> &a,
                        const tf::volume<Policy1> &b) -> bool {
  const Real eps = static_cast<Real>(1e-6);
  if (a.dims() != b.dims())
    return false;
  for (std::size_t i = 0; i < tf::coordinate_dims_v<Policy0>; ++i) {
    if (std::abs(static_cast<Real>(a.spacing()[i]) -
                 static_cast<Real>(b.spacing()[i])) > eps)
      return false;
    if (std::abs(static_cast<Real>(a.origin()[i]) -
                 static_cast<Real>(b.origin()[i])) > eps)
      return false;
  }
  return true;
}

} // namespace volume_detail
} // namespace tf
