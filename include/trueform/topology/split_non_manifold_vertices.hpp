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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_copy_blocked.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/indirect_range.hpp"
#include "../reindex/return_index_map.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"
#include "./split/apply_vertex_fan_split.hpp"
#include "./split/make_vertex_fan_mint_offsets.hpp"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup topology_analysis
/// @brief Give every fan at a vertex a vertex of its own, and say where the
/// points came from.
///
/// A vertex whose faces walk as several fans keeps its id on the fan holding
/// its smallest face; every other fan takes a fresh id carrying the same
/// coordinates, and the fan's corners are rewired onto it. Faces keep their
/// ids, their arity and their winding, and the points are the input's
/// followed by the minted copies in id order.
///
/// An edge three faces carry is crossed by no fan, so the faces it holds are
/// fans of their own and separating them would tear that edge into boundary
/// copies. That is another verb's work: a vertex any of whose edges carries
/// three faces is left exactly as it was, and
/// @ref tf::make_non_manifold_vertices still names it. This one separates
/// only the fans that come apart without cutting an edge.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::polygons_buffer, point map: for each output
///   point the input point it copies, an original itself).
template <typename Policy>
auto split_non_manifold_vertices(const tf::polygons<Policy> &polygons,
                                 tf::return_index_map_t) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::polygons_buffer<Index, tf::coordinate_type<Policy>,
                        tf::coordinate_dims_v<Policy>,
                        tf::static_size_v<decltype(polygons[0])>>
        out;
    if constexpr (tf::static_size_v<decltype(polygons[0])> ==
                  tf::dynamic_size) {
      auto &offsets = out.faces_buffer().offsets_buffer();
      offsets.allocate(polygons.size() + 1);
      offsets[0] = 0;
      for (auto [i, face] : tf::enumerate(polygons.faces()))
        offsets[i + 1] = offsets[i] + Index(face.size());
      out.faces_buffer().data_buffer().allocate(offsets.back());
    } else {
      out.faces_buffer().allocate(polygons.size());
    }
    tf::parallel_copy_blocked(polygons.faces(), out.faces());

    auto mint_offsets = tf::topology::make_vertex_fan_mint_offsets(
        polygons.faces(), polygons.face_membership());
    auto point_map = tf::topology::apply_vertex_fan_split(
        polygons.faces(), polygons.face_membership(), mint_offsets,
        out.faces());
    out.points_buffer().allocate(point_map.size());
    tf::parallel_copy(tf::make_indirect_range(point_map, polygons.points()),
                      out.points());
    return std::make_pair(std::move(out), std::move(point_map));
  } else {
    auto fm = tf::make_face_membership(polygons);
    return tf::split_non_manifold_vertices(polygons | tf::tag(fm),
                                           tf::return_index_map);
  }
}

/// @ingroup topology_analysis
/// @brief Give every fan at a vertex a vertex of its own.
/// @overload
template <typename Policy>
auto split_non_manifold_vertices(const tf::polygons<Policy> &polygons) {
  auto split = tf::split_non_manifold_vertices(polygons, tf::return_index_map);
  return std::move(split.first);
}

} // namespace tf
