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

#include "../core/coordinate_type.hpp"
#include "../core/edges.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/signed_area.hpp"
#include "./face_hole_relations.hpp"
#include "./planar_graph_regions.hpp"

#include <type_traits>

namespace tf {

/// @ingroup topology_planar
/// @brief Computes a planar embedding from directed edges.
///
/// A planar embedding represents a planar graph as a collection of faces
/// and holes. Faces are CCW-oriented regions with positive area, holes
/// are CW-oriented regions with negative area. The embedding also computes
/// parent-child relationships between faces and holes.
template <typename Index, typename Int = tf::exact::int32>
class planar_embedding {
public:
  template <typename Policy0, typename Policy1>
  auto build(const tf::edges<Policy0> &directed_edges,
             const tf::points<Policy1> &points) {
    clear();
    using coord_t = tf::coordinate_type<Policy1>;
    if constexpr (std::is_integral_v<coord_t>) {
      build_impl(directed_edges, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      build_impl(directed_edges, int_pts);
    }
  }

  template <typename Policy> auto build(const tf::segments<Policy> &segments) {
    return build(segments.edges(), segments.points());
  }

  auto faces() const { return tf::make_indirect_range(_faces, _pgr); }
  auto holes() const { return tf::make_indirect_range(_holes, _pgr); }

  auto holes_for_faces() const {
    return tf::make_offset_block_range(_fhr.offsets_buffer(),
                                       _fhr.data_buffer());
  }

  auto clear() {
    _pgr.clear();
    _fhr.clear();
    _faces.clear();
    _holes.clear();
  }

private:
  template <typename Policy0, typename Policy1>
  auto build_impl(const tf::edges<Policy0> &directed_edges,
                  const tf::points<Policy1> &points) {
    _pgr.build(directed_edges, points);
    classify_regions(points);
    _fhr.build(tf::make_faces(faces()), tf::make_faces(holes()), points);
  }

  template <typename Policy>
  auto classify_regions(const tf::points<Policy> &points) {
    for (auto [i, region] : tf::enumerate(_pgr)) {
      if (region.size() < 3)
        continue;
      auto sign = tf::exact::signed_area_sign(tf::make_polygon(region, points));
      if (sign > 0)
        _faces.push_back(Index(i));
      else
        _holes.push_back(Index(i));
    }
  }

  tf::planar_graph_regions<Index, Int> _pgr;
  tf::face_hole_relations<Index, Int> _fhr;
  tf::buffer<Index> _faces;
  tf::buffer<Index> _holes;
};
} // namespace tf
