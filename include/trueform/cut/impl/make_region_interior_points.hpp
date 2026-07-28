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
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/topo_id.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief The interior point a persisted Steiner id names.
template <typename Index, typename Int>
auto region_interior_point_of(Index id, Index n_intersection_points,
                              const tf::buffer<tf::point<Int, 3>> &extra_points)
    -> tf::cut::detail::region_interior_point<Index, Int> {
  return {{tf::intersect::graph::vertex_source::created, id,
           {0, tf::topo_type::face}},
          extra_points[std::size_t(id - n_intersection_points)]};
}

/// @ingroup cut
/// @brief A region's interior points, read off the persisted Steiner table.
///
/// An empty table means no region has interior points, so every region maps
/// the empty range rather than indexing a table that is not there.
template <typename Index, typename Int>
auto make_region_interior_points(
    Index region, const tf::offset_block_buffer<Index, Index> &steiners,
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points) {
  const Index *begin = nullptr;
  const Index *end = nullptr;
  if (steiners.size() != 0) {
    begin = steiners[std::size_t(region)].begin();
    end = steiners[std::size_t(region)].end();
  }
  return tf::make_mapped_range(
      tf::make_range(begin, end),
      [n_intersection_points, &extra_points](Index id)
          -> tf::cut::detail::region_interior_point<Index, Int> {
        return tf::cut::region_interior_point_of(id, n_intersection_points,
                                                 extra_points);
      });
}

} // namespace tf::cut
