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
 * Author: Ziga Sajovic
 */
#pragma once
#include "../core/algorithm/circular_decrement.hpp"
#include "../core/algorithm/circular_increment.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/faces.hpp"
#include "../core/intersects.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points.hpp"
#include "../core/polygons.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/slide_range.hpp"
#include "../exact/classify.hpp"
#include "../exact/meta.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/signed_area.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/search.hpp"

#include <type_traits>

namespace tf {

/// @ingroup topology_planar
/// @brief Computes parent-child relationships between faces and holes.
///
/// For each hole, determines which face contains it (the smallest face
/// that contains a point from the hole). Uses an AABB tree for efficient
/// spatial queries and exact orient2d for point-in-polygon.
///
/// Float points are converted via pt_converter. Areas computed
/// internally via exact signed_area_2x.
template <typename Index, typename Int = tf::exact::int32>
class face_hole_relations : public tf::offset_block_buffer<Index, Index> {
  using T2 = typename tf::exact::meta<Int>::T2;
  using base_t = tf::offset_block_buffer<Index, Index>;

public:
  template <typename Policy0, typename Policy1, typename Policy2>
  auto build(const tf::faces<Policy0> &faces, const tf::faces<Policy1> &holes,
             const tf::points<Policy2> &points) {
    clear();
    if (!holes.size())
      return;

    using coord_t = tf::coordinate_type<Policy2>;
    if constexpr (std::is_integral_v<coord_t>) {
      build_impl(faces, holes, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      build_impl(faces, holes, int_pts);
    }
  }

  auto clear() {
    base_t::clear();
    _tree.clear();
    _face_areas.clear();
    _hole_in_face.clear();
  }

private:
  /// A vertex of the hole absent from the candidate face, or
  /// `not_same = false` when the lockstep walk matches throughout —
  /// the hole is that face's reversal (an interior cycle against its
  /// island face). Planar-graph connectivity guarantees every hole is
  /// vertex-disjoint from its true parent's walk (a component sharing
  /// a vertex with a loop is absorbed into that loop's own walk), so
  /// the parent always yields a classifiable vertex here.
  template <typename Policy0, typename Policy1>
  auto find_first_non_equal_vertex(const tf::faces<Policy0> &faces,
                                   const tf::faces<Policy1> &holes,
                                   Index face_id, Index hole_id) {
    auto face = faces[face_id];
    auto hole = holes[hole_id];
    auto it = std::find(face.begin(), face.end(), hole[0]);
    if (it == face.end())
      return std::make_pair(true, Index(0));

    Index face_i = static_cast<Index>(std::distance(face.begin(), it));
    Index hole_i = 0;
    const Index face_n = static_cast<Index>(face.size());
    const Index hole_n = static_cast<Index>(hole.size());
    const Index steps = std::min(face_n, hole_n);

    for (Index i = 1; i < steps; ++i) {
      face_i = tf::circular_increment(face_i, face_n);
      hole_i = tf::circular_decrement(hole_i, hole_n);
      if (face[face_i] != hole[hole_i])
        return std::make_pair(true, hole_i);
    }
    return std::make_pair(false, Index(0));
  }

  template <typename Policy0, typename Policy1, typename Policy2>
  auto build_impl(const tf::faces<Policy0> &faces,
                  const tf::faces<Policy1> &holes,
                  const tf::points<Policy2> &points) {
    _face_areas.allocate(faces.size());
    for (std::size_t i = 0; i < faces.size(); ++i) {
      auto a = tf::exact::signed_area_2x(faces[i], [&](auto idx) {
        auto &&p = points[idx];
        return tf::exact::pt2<Int>{p[0], p[1]};
      });
      _face_areas[i] = a < 0 ? -a : a;
    }

    auto fcs = tf::make_polygons(faces, points);
    _tree.build(fcs, tf::config_tree(4, 4));

    auto &offsets = base_t::offsets_buffer();
    auto &data = base_t::data_buffer();
    _hole_in_face.allocate(holes.size());
    offsets.allocate(faces.size() + 1);
    std::fill(offsets.begin(), offsets.end(), 0);

    for (auto &&[hole_id, in_face] : tf::enumerate(_hole_in_face)) {
      auto &&test_pt = points[holes[hole_id][0]];
      in_face = -1;
      T2 best_area = T2(0);
      bool found = false;

      tf::search(_tree, tf::intersects_f(test_pt),
                 [&, &hole_id = hole_id, &in_face = in_face](Index face_id) {
                   if (found && _face_areas[face_id] >= best_area)
                     return;
                   auto [not_same, vid] = find_first_non_equal_vertex(
                       faces, holes, face_id, hole_id);
                   if (!not_same)
                     return;

                   auto &&hole_pt = points[holes[hole_id][vid]];
                   tf::point<Int, 2> qpt = {hole_pt[0], hole_pt[1]};
                   auto result = tf::exact::classify(qpt, fcs[face_id]);
                   if (result != tf::containment::outside) {
                     in_face = face_id;
                     best_area = _face_areas[face_id];
                     found = true;
                   }
                 });

      if (in_face != -1)
        offsets[in_face]++;
    }

    for (auto &&[a, b] : tf::make_slide_range<2>(offsets))
      b += a;
    data.allocate(offsets.back());
    for (auto [hole_id, face_id] : tf::enumerate(_hole_in_face))
      if (face_id != -1)
        data[--offsets[face_id]] = hole_id;
  }

  tf::aabb_tree<Index, Int, 2> _tree;
  tf::buffer<T2> _face_areas;
  tf::buffer<Index> _hole_in_face;
};
} // namespace tf
