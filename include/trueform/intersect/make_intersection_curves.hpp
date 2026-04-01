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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/transformed.hpp"
#include "../cut/cut_graph.hpp"
#include "../cut/dispatch/arrangement.hpp"
#include "../cut/dispatch/boolean.hpp"
#include "../cut/face_cuts.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./graph/intersection_graph.hpp"
#include "./intersect_mode.hpp"
#include "./intersections_between_polygons.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract intersection curves where two meshes intersect.
///
/// Computes the geometric intersection between two polygon meshes and
/// returns the result as connected curves.
///
/// @tparam Policy0 The policy type for the first mesh.
/// @tparam Policy1 The policy type for the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto make_intersection_curves(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_mode mode = tf::intersect_mode::sos) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [mode](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;

        tf::intersections_between_polygons<Index, RealType, Int> ibp;
        ibp.build(p0, p1, mode);

        auto &conv = ibp.converter();
        auto apply_to_face = [&](int tag, Index object, const auto &f) {
          if (tag == 0)
            f(p0.faces()[object]);
          else
            f(p1.faces()[object]);
        };
        auto get_mesh_point = [&](int tag, Index id) -> tf::point<Int, 3> {
          if (tag == 0)
            return conv.convert(
                tf::transformed(p0.points()[id], tf::frame_of(p0)));
          else
            return conv.convert(
                tf::transformed(p1.points()[id], tf::frame_of(p1)));
        };

        tf::intersection_graph<Index, Int> ig;
        ig.build(ibp, apply_to_face, get_mesh_point, mode);

        auto paths = [&]() {
          if (mode & tf::intersect_mode::sos) {
            auto edge_pairs = tf::make_mapped_range(
                ig.edge_groups(),
                [](const auto &group) -> std::array<Index, 2> {
                  return {group[0].point_0, group[0].point_1};
                });
            return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
          } else {
            tf::face_cuts<Index, Int> fc;
            fc.build(ig, apply_to_face, get_mesh_point);

            tf::cut_graph<Index> cg;
            cg.build(fc, static_cast<Index>(ig.points().size()));

            return tf::connect_edges_to_paths(
                tf::make_edges(cg.intersection_edges()));
          }
        }();

        auto ipts = ig.points();
        tf::curves_buffer<Index, RealType, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
            cb.points());
        return cb;
      });
}

namespace intersect {

template <typename Int, typename FormsRange>
auto intersection_curves_n(const FormsRange &forms, tf::intersect_mode mode) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using RealType = tf::coordinate_type<decltype(forms[0])>;

  tf::intersections_between_polygons<Index, RealType, Int> ibp;
  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<Int, 3> {
    return conv.convert(
        tf::transformed(forms[tag].points()[id], tf::frame_of(forms[tag])));
  };

  tf::intersection_graph<Index, Int> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, mode);

  auto paths = [&]() {
    if (mode & tf::intersect_mode::sos) {
      auto edge_pairs = tf::make_mapped_range(
          ig.edge_groups(), [](const auto &group) -> std::array<Index, 2> {
            return {group[0].point_0, group[0].point_1};
          });
      return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
    } else {
      tf::face_cuts<Index, Int> fc;
      fc.build(ig, apply_to_face, get_mesh_point);

      tf::cut_graph<Index> cg;
      cg.build(fc, static_cast<Index>(ig.points().size()));

      return tf::connect_edges_to_paths(
          tf::make_edges(cg.intersection_edges()));
    }
  }();

  auto ipts = ig.points();
  tf::curves_buffer<Index, RealType, 3> cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(ipts.size());
  tf::parallel_copy(
      tf::make_points(tf::make_mapped_range(
          ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
      cb.points());
  return cb;
}

} // namespace intersect

/// @ingroup intersect_curves
/// @brief Extract intersection curves from N meshes.
///
/// Computes all pairwise intersections among the given meshes and
/// returns the result as connected curves.
///
/// @tparam Range A range of @ref tf::polygons (or tagged forms).
/// @param _forms The input meshes.
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::exact::int32, typename Range>
auto make_intersection_curves(
    const Range &_forms,
    tf::intersect_mode mode = tf::intersect_mode::sos |
                              tf::intersect_mode::resolve_crossing_contours) {
  return cut::dispatch::arrangement(_forms, [mode](const auto &forms) {
    return intersect::intersection_curves_n<Int>(forms, mode);
  });
}

} // namespace tf
