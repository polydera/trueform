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

#include "../../../core/algorithm/generate_offset_blocks.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../../core/point.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../placement_tables.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::exact::door::pool {

/// `R(n)`: every original nonfeature vertex a name carries, deduplicated,
/// one contiguous block per name.
///
/// A name is INDIVISIBLE — one distant occurrence refuses it globally — so
/// this set is what a certificate is asked about.
///
/// The block the primitive is filling IS the scratch: the corners are
/// appended, ordered and deduplicated in place, so a name of any population
/// costs one append and no allocation of its own.
template <typename Index, typename Int, typename RealType>
auto gather_name_supports(const placement_tables<Index, Int, RealType> &tables,
                          const tf::blocked_buffer<Index, 3> &corners,
                          const tf::offset_block_buffer<int, Index> &name_faces,
                          const tf::buffer<char> &feature,
                          tf::buffer<int> &support_offsets,
                          tf::buffer<tf::point<Int, 3>> &support) -> void {
  support.clear();
  support_offsets.allocate(name_faces.size() + 1);
  support_offsets[0] = 0;
  tf::generate_offset_blocks(
      tf::make_sequence_range(name_faces.size()), support_offsets, support,
      [&tables, &corners, &name_faces,
       &feature](std::size_t n, tf::buffer<tf::point<Int, 3>> &into) {
        const auto base = into.size();
        for (const auto face : name_faces[n])
          for (const auto v : corners[std::size_t(face)])
            if (!feature[std::size_t(v)])
              into.push_back(tables.points[std::size_t(v)]);
        std::sort(into.begin() + base, into.end());
        into.erase_till_end(std::unique(into.begin() + base, into.end()));
      },
      tf::checked);
}

} // namespace tf::exact::door::pool
