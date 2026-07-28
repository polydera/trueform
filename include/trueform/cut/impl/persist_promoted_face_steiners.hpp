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
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "./resolve_region_vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index>
auto persist_promoted_face_steiners(
    Index n_tags, Index n_intersection_points, Index interior_base,
    const tf::buffer<tf::intersect::graph::face_descriptor<Index>>
        &promoted,
    const tf::buffer<Index> &interior_counts,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    tf::buffer<std::array<Index, 3>> &promoted_steiners) -> void {
  Index offset = 0;
  for (std::size_t face = 0; face < interior_counts.size(); ++face) {
    const auto descriptor = promoted[face];
    for (Index steiner = 0; steiner < interior_counts[face]; ++steiner) {
      const Index id =
          n_intersection_points + interior_base + offset + steiner;
      const tf::intersect::graph::vertex<Index> vertex{
          tf::intersect::graph::vertex_source::created,
          id,
          {0, tf::topo_type::face}};
      if (tf::cut::resolve_region_vertex_key(
              n_tags, Index(0), vertex, merges) ==
          tf::cut::detail::region_vertex_key(
              n_tags, Index(0), vertex))
        promoted_steiners.push_back(
            {descriptor.tag, descriptor.object, id});
    }
    offset += interior_counts[face];
  }
  if (merges.size() != 0) {
    std::size_t write = 0;
    for (std::size_t index = 0; index < promoted_steiners.size();
         ++index) {
      const tf::intersect::graph::vertex<Index> vertex{
          tf::intersect::graph::vertex_source::created,
          promoted_steiners[index][2],
          {0, tf::topo_type::face}};
      if (tf::cut::resolve_region_vertex_key(
              n_tags, Index(0), vertex, merges) ==
          tf::cut::detail::region_vertex_key(
              n_tags, Index(0), vertex))
        promoted_steiners[write++] = promoted_steiners[index];
    }
    promoted_steiners.erase_till_end(
        promoted_steiners.begin() + std::ptrdiff_t(write));
  }
  std::sort(promoted_steiners.begin(), promoted_steiners.end());
}

} // namespace tf::cut
