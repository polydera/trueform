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
#include "../core/algorithm/circular_decrement.hpp"
#include "../core/algorithm/circular_increment.hpp"
#include "../core/centroid.hpp"
#include "../core/faces.hpp"
#include "../core/intersects.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points.hpp"
#include "../core/polygons.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/slide_range.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/search.hpp"

namespace tf {

/// @ingroup topology_planar
/// @brief Computes parent-child relationships between faces and holes.
///
/// For each hole, determines which face contains it (the smallest face
/// that contains a point from the hole). Uses an AABB tree for efficient
/// spatial queries.
///
/// @tparam Index The integer type for indices.
/// @tparam RealT The real type for coordinates.
template <typename Index, typename RealT>
class face_hole_relations : public tf::offset_block_buffer<Index, Index> {
  using base_t = tf::offset_block_buffer<Index, Index>;

public:
  /// @brief Build face-hole relationships.
  /// @tparam Policy0 The faces policy type.
  /// @tparam Range The face areas range type.
  /// @tparam Policy1 The holes policy type.
  /// @tparam Policy2 The points policy type.
  /// @param faces The face polygons.
  /// @param face_areas The areas of each face (for picking smallest containing face).
  /// @param holes The hole polygons.
  /// @param points The vertex positions.
  template <typename Policy0, typename Range, typename Policy1,
            typename Policy2>
  auto build(const tf::faces<Policy0> &faces, const Range &face_areas,
             const tf::faces<Policy1> &holes,
             const tf::points<Policy2> &points) {
    clear();
    if(!holes.size())
      return;
    _tree.build(tf::make_polygons(faces, points), tf::config_tree(4, 4));
    build_hole_structures(faces, face_areas, holes, points);
  }

  /// @brief Clear all internal state.
  auto clear() {
    base_t::clear();
    _tree.clear();
    _hole_in_face.clear();
  }

private:
  template <typename Policy0, typename Policy1>
  auto find_first_non_equal_vertex(const tf::faces<Policy0> &faces,
                                   const tf::faces<Policy1> &holes,
                                   Index face_id, Index hole_id) {
    auto face = faces[face_id];
    auto hole = holes[hole_id];
    auto it = std::find(face.begin(), face.end(), hole[0]);
    if (it == face.end())
      return std::make_pair(true, Index(0));
    else {
      Index face_i = std::distance(face.begin(), it);
      Index hole_i = 0; // we start from hole[0]

      const Index face_n = static_cast<int>(face.size());
      const Index hole_n = static_cast<int>(hole.size());
      const Index steps = std::min(face_n, hole_n);

      for (Index i = 1; i < steps; ++i) {
        face_i = tf::circular_increment(face_i, face_n);
        hole_i = tf::circular_decrement(hole_i, hole_n);
        if (face[face_i] != hole[hole_i])
          return std::make_pair(true, hole_i);
      }

      return std::make_pair(false, Index(0));
    }
  }

  template <typename Policy0, typename Range, typename Policy1,
            typename Policy2>
  auto build_hole_structures(const tf::faces<Policy0> &faces,
                             const Range &face_areas,
                             const tf::faces<Policy1> &holes,
                             const tf::points<Policy2> &points) {
    auto &_face_holes_offset = base_t::offsets_buffer();
    auto &_face_holes = base_t::data_buffer();
    _hole_in_face.allocate(holes.size());
    _face_holes_offset.allocate(faces.size() + 1);
    std::fill(_face_holes_offset.begin(), _face_holes_offset.end(), 0);
    auto fcs = tf::make_polygons(faces, points);

    for (auto &&[hole_id, in_face] : tf::enumerate(_hole_in_face)) {
      auto hole = tf::make_polygon(holes[hole_id], points);
      auto center = tf::centroid(hole);
      in_face = -1;
      auto area = std::numeric_limits<RealT>::max();
      tf::search(_tree, tf::intersects_f(center),
                 [&, &in_face = in_face, &hole_id = hole_id](Index face_id) {
                   if (face_areas[face_id] < area) {
                     auto [not_same, v_id] = find_first_non_equal_vertex(
                         faces, holes, face_id, hole_id);
                     if (not_same && tf::contains_coplanar_point(fcs[face_id],
                                                                 hole[v_id])) {
                       in_face = face_id;
                       area = face_areas[face_id];
                     }
                   }
                 });
      if (in_face != -1)
        _face_holes_offset[in_face]++;
    }

    for (auto &&[a, b] : tf::make_slide_range<2>(_face_holes_offset))
      b += a;
    _face_holes.allocate(_face_holes_offset.back());
    for (auto [hole_id, face_id] : tf::enumerate(_hole_in_face))
      if (face_id != -1)
        _face_holes[--_face_holes_offset[face_id]] = hole_id;
  }

  tf::aabb_tree<Index, RealT, 2> _tree;
  tf::buffer<Index> _hole_in_face;
};
} // namespace tf
