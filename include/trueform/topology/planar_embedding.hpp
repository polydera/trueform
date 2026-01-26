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

#include "../core/area.hpp"
#include "../core/edges.hpp"
#include "./face_hole_relations.hpp"
#include "./planar_graph_regions.hpp"

namespace tf {

/// @ingroup topology_planar
/// @brief Computes a planar embedding from directed edges.
///
/// A planar embedding represents a planar graph as a collection of faces
/// and holes. Faces are CCW-oriented regions with positive area, holes
/// are CW-oriented regions with negative area. The embedding also computes
/// parent-child relationships between faces and holes.
///
/// @tparam Index The integer type for indices.
/// @tparam RealT The real type for coordinates (default: double).
template <typename Index, typename RealT> class planar_embedding {
public:
  /// @brief Build the planar embedding from directed edges and points.
  /// @tparam Policy0 The edges policy type.
  /// @tparam Policy1 The points policy type.
  /// @param directed_edges The directed edges of the planar graph.
  /// @param points The vertex positions.
  template <typename Policy0, typename Policy1>
  auto build(const tf::edges<Policy0> &directed_edges,
             const tf::points<Policy1> &points) {
    clear();
    _pgr.build(directed_edges, points);
    build_data(points);
    _fhr.build(tf::make_faces(faces()), face_areas(), tf::make_faces(holes()),
               points);
  }

  /// @brief Get the faces (positive area regions).
  auto faces() const { return tf::make_indirect_range(_faces, _pgr); }

  /// @brief Get the signed areas of faces.
  auto face_areas() const {
    return tf::make_indirect_range(_faces, _signed_areas);
  }

  /// @brief Get the holes (negative area regions).
  auto holes() const { return tf::make_indirect_range(_holes, _pgr); }

  /// @brief Get the signed areas of holes.
  auto hole_areas() const {
    return tf::make_indirect_range(_holes, _signed_areas);
  }

  /// @brief Get which holes belong to which faces.
  /// @return An offset-block range where each block contains hole indices for that face.
  auto holes_for_faces() const {
    return tf::make_offset_block_range(_fhr.offsets_buffer(),
                                       _fhr.data_buffer());
  }

  /// @brief Clear all internal state.
  auto clear() {
    _pgr.clear();
    _fhr.clear();
    _faces.clear();
    _holes.clear();
    _signed_areas.clear();
  }

private:
  template <typename Policy> auto build_data(const tf::points<Policy> &points) {
    Index min_area_id = -1;
    RealT min_area = std::numeric_limits<RealT>::max();
    for (const auto &region : _pgr) {
      auto sa = region.size() < 3
                    ? RealT(0)
                    : RealT(tf::signed_area(tf::make_polygon(region, points)));
      if (sa < 0 && sa < min_area) {
        min_area = sa;
        min_area_id = _signed_areas.size();
      }
      _signed_areas.push_back(sa);
    }
    for (auto [i, loop] : tf::enumerate(_pgr)) {
      if (Index(i) == min_area_id)
        continue;
      if (_signed_areas[i] > 0)
        _faces.push_back(i);
      else
        _holes.push_back(i);
    }
  }

  tf::planar_graph_regions<Index, RealT> _pgr;
  tf::face_hole_relations<Index, RealT> _fhr;
  tf::buffer<Index> _faces;
  tf::buffer<Index> _holes;
  tf::buffer<RealT> _signed_areas;
};
} // namespace tf
