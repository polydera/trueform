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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/checked.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../volume.hpp"
#include "./field_value.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>

namespace tf {
namespace volume_detail {

/// @brief The far-outside sentinel of a field: strictly above every sample,
/// by a margin of one (largest) grid step, so no isovalue at or below the
/// field maximum is ever crossed against it.
template <typename Real, typename Policy>
inline auto far_outside_value(const tf::volume<Policy> &vol) -> Real {
  Real max_sample = Real{0};
  if (vol.voxel_count() != 0)
    max_sample = tf::reduce(
        vol,
        [](Real acc, const auto &sample) {
          return std::max(acc, field_value<Real>(sample));
        },
        std::numeric_limits<Real>::lowest(), tf::checked);
  Real max_spacing = static_cast<Real>(vol.spacing()[0]);
  for (std::size_t d = 1; d < tf::coordinate_dims_v<Policy>; ++d)
    max_spacing = std::max(max_spacing, static_cast<Real>(vol.spacing()[d]));
  return max_sample + std::max(max_spacing, Real{1});
}

} // namespace volume_detail
} // namespace tf
