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

#include "./make_support_plane.hpp"

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/range.hpp"
#include "../../../core/views/indirect_range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"
#include "../placement_tables.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// The exact plane every face of the flat world stands on, in the flat face
/// numbering the tables already address.
///
/// THE ZERO QUADRUPLE IS THE FACE THAT NAMES NO PLANE — the same convention
/// the tables' own grid names carry — so a face with no support triple needs
/// no mask beside it.
template <typename Index, typename Int, typename RealType>
auto name_face_planes(const placement_tables<Index, Int, RealType> &tables,
                      const tf::blocked_buffer<Index, 3> &corners,
                      tf::buffer<tf::exact::canonical_plane<Int>> &plane)
    -> void {
  plane.allocate(corners.size());
  tf::parallel_for_each(
      tf::make_sequence_range(corners.size()),
      [&tables, &corners, &plane](std::size_t f) {
        plane[f] = tf::exact::canonical_plane<Int>{};
        make_support_plane<Int>(
            tf::make_indirect_range(corners[f], tf::make_range(tables.points)),
            plane[f]);
      },
      tf::checked);
}

} // namespace tf::exact::door::pool
