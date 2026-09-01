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

#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"

#include <array>
#include <cstddef>

namespace tf::iso {

/// The pieces a scalar field cut the faces into, numbered densely across the
/// cut faces: block position is the region id, and a corner is a FLAT identity
/// in `[originals | field points | minted]`. A region's band is the label the
/// cut pass wrote into the partition it feeds.
///
/// `refused` names, ascending, the cut faces the constrained build declined —
/// a face that holds no piece here and none anywhere else. The chord split
/// cannot refuse (it declines to the constrained build instead), and a
/// constraint set the field states on a face it crosses is recoverable, so the
/// list speaks only on degenerate input and is otherwise empty.
template <typename Index, typename RealT, std::size_t Dims>
struct iso_cut_regions {
  tf::offset_block_buffer<Index, std::array<Index, 3>> triangles;
  tf::buffer<Index> faces;
  tf::buffer<tf::point<RealT, Dims>> minted_points;
  tf::buffer<Index> refused;
};

} // namespace tf::iso
