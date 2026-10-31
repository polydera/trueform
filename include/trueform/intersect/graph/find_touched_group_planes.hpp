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
#include "./plane_tables.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::intersect::graph {

/// The planes holding an instance of a `touched` group of the standing
/// `tables`, ascending: a touched group's constraint changed on every one of
/// them, so they are the dirty set of the restatement that touched it.
template <typename Index, typename Int, typename PlaneOfFace>
auto find_touched_group_planes(const plane_tables<Index, Int> &tables,
                               const tf::buffer<Index> &touched,
                               const PlaneOfFace &plane_of_face)
    -> tf::buffer<Index> {
  const auto offsets = tables.def_offsets();
  const auto defs = tables.edge_defs();
  tf::buffer<Index> planes;
  for (const auto canon : touched)
    for (auto row = offsets[std::size_t(canon)];
         row < offsets[std::size_t(canon) + 1]; ++row)
      planes.push_back(plane_of_face(defs[std::size_t(row)].face));
  std::sort(planes.begin(), planes.end());
  planes.erase_till_end(std::unique(planes.begin(), planes.end()));
  return planes;
}

} // namespace tf::intersect::graph
