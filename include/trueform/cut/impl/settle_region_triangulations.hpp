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
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./conform_region_welds.hpp"
#include "./consolidate_region_merges.hpp"
#include "./recover_region_triangulations.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief Bring an emitted triangulation to rest: recover what emitted
///        nothing, fold the welds it produced into the authority, and carry
///        them to the triangles that were emitted before they existed.
///
/// The stock and refined builds reach this point by different routes and
/// must leave it in the same state, so they share the step rather than each
/// stating it. Two copies of it had already drifted: one gated the weld
/// carry on the promotion round having produced a weld, which leaves a
/// recovery-era weld applied to the identity table and to nothing else.
///
/// Regions that emit nothing are holes in the surface, so they go back
/// through the wave. The merge table is deliberately not a reason to recover
/// here — @ref tf::cut::conform_region_welds below is what carries a weld to
/// triangles that already exist.
///
/// Returns the number of regions still empty afterwards.
template <typename Index, typename Int, typename ApplyToFace,
          typename GetMeshPoint, typename DeadLoops>
auto settle_region_triangulations(
    const tf::face_regions<Index, Int> &regions,
    const tf::intersection_graph<Index, Int> &intersection_graph,
    const ApplyToFace &apply_to_face, const GetMeshPoint &get_mesh_point,
    const DeadLoops &dead_loops, Index n_tags, Index n_intersection_points,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>> &merges,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &pending_merges,
    const tf::buffer<tf::intersect::graph::face_descriptor<Index>> &promoted,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    tf::buffer<Index> &failed,
    const tf::offset_block_buffer<Index, Index> &steiners = {}) -> Index {
  if (failed.size())
    tf::cut::recover_region_triangulations(
        regions, intersection_graph, apply_to_face, get_mesh_point, dead_loops,
        n_tags, n_intersection_points, extra_points, splits, merges, ranges,
        triangles, failed, steiners);
  tf::cut::consolidate_region_merges<Index>(merges, pending_merges);
  tf::cut::conform_region_welds(regions, n_tags, merges, promoted, ranges,
                                triangles);
  return static_cast<Index>(failed.size());
}

} // namespace tf::cut
