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
#include "./field_value.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

// Multilinear interpolation of the 2^Dims samples around a grid coordinate
// already clamped to [0, dims - 1]: one weight per corner, the product over
// the axes of that corner's fraction. Trilinear at three axes, bilinear at two.
template <typename Real, typename Policy>
inline auto multilinear_at(const tf::volume<Policy> &vol,
                           const std::array<Real, coordinate_dims_v<Policy>> &g)
    -> Real {
  constexpr std::size_t Dims = coordinate_dims_v<Policy>;
  std::array<int, Dims> lo{}, hi{};
  std::array<Real, Dims> f{};
  for (std::size_t d = 0; d < Dims; ++d) {
    lo[d] = static_cast<int>(std::floor(g[d]));
    hi[d] = std::min(lo[d] + 1, vol.dims()[d] - 1);
    f[d] = g[d] - static_cast<Real>(lo[d]);
  }

  Real acc = Real{0};
  for (unsigned corner = 0; corner < (1u << Dims); ++corner) {
    Real weight = Real{1};
    std::array<int, Dims> at{};
    for (std::size_t d = 0; d < Dims; ++d) {
      const bool up = ((corner >> d) & 1u) != 0;
      weight *= up ? f[d] : Real{1} - f[d];
      at[d] = up ? hi[d] : lo[d];
    }
    if (weight != Real{0})
      acc += weight * field_value<Real>(vol(at));
  }
  return acc;
}

// Multilinear sample of a scalar field at a local-space point, with
// clamp-to-edge addressing. Outside the grid the nearest boundary sample is
// returned; for a well-formed SDF (whose surface sits strictly inside its own
// domain) that boundary value is a positive "far outside" distance, so
// resampling onto a larger common grid never introduces a spurious zero
// crossing.
template <typename Real, typename Policy>
inline auto
sample_multilinear(const tf::volume<Policy> &vol,
                   const tf::point<Real, coordinate_dims_v<Policy>> &p)
    -> Real {
  constexpr std::size_t Dims = coordinate_dims_v<Policy>;
  std::array<Real, Dims> g{};
  for (std::size_t d = 0; d < Dims; ++d) {
    const Real at = (p[d] - static_cast<Real>(vol.origin()[d])) /
                    static_cast<Real>(vol.spacing()[d]);
    g[d] = std::min(std::max(at, Real{0}),
                    static_cast<Real>(vol.dims()[d] - 1));
  }
  return multilinear_at(vol, g);
}

// Sentinel variant: a sample point outside the grid box returns the
// caller-provided `outside` value instead of the clamped boundary sample. A
// tiny index-space tolerance absorbs floating-point roundoff so points sitting
// exactly on the boundary are still treated as inside (and clamped).
template <typename Real, typename Policy>
inline auto
sample_multilinear(const tf::volume<Policy> &vol,
                   const tf::point<Real, coordinate_dims_v<Policy>> &p,
                   Real outside) -> Real {
  constexpr std::size_t Dims = coordinate_dims_v<Policy>;
  const Real eps = static_cast<Real>(1e-4);
  std::array<Real, Dims> g{};
  for (std::size_t d = 0; d < Dims; ++d) {
    const Real hi = static_cast<Real>(vol.dims()[d] - 1);
    g[d] = (p[d] - static_cast<Real>(vol.origin()[d])) /
           static_cast<Real>(vol.spacing()[d]);
    if (g[d] < -eps || g[d] > hi + eps)
      return outside;
    g[d] = std::min(std::max(g[d], Real{0}), hi);
  }
  return multilinear_at(vol, g);
}

} // namespace volume_detail
} // namespace tf
