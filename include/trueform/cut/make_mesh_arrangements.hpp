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
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/make_mesh_arrangements.hpp"
#include "./cut_graph.hpp"
#include "./dispatch/arrangement.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./face_cuts.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto make_mesh_arrangements(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_mode mode = tf::intersect_mode::primitives) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [mode](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType, Int>(p0, p1,
                                                                      mode);
        auto [mesh, tag_labels, face_labels, map_data] =
            tf::cut::make_mesh_arrangements(ig, fc, p0, p1, ibp.converter());
        return std::make_tuple(std::move(mesh), std::move(tag_labels),
                               std::move(face_labels));
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curve output.
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::intersect_mode mode, tf::return_curves_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [mode](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType, Int>(p0, p1,
                                                                      mode);
        auto [mesh, tag_labels, face_labels, map_data] =
            tf::cut::make_mesh_arrangements(ig, fc, p0, p1, ibp.converter());

        auto paths =
            tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
        auto &conv = ibp.converter();
        auto ipts = ig.points();
        tf::curves_buffer<Index, RealType, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
            cb.points());

        return std::make_tuple(std::move(mesh), std::move(tag_labels),
                               std::move(face_labels), std::move(cb));
      });
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curves (default intersect mode).
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_curves_t) {
  return make_mesh_arrangements<Int>(_polygons0, _polygons1,
                                     tf::intersect_mode::primitives,
                                     tf::return_curves);
}

namespace cut {

template <typename Int, typename FormsRange>
auto mesh_arrangements_n(const FormsRange &forms, tf::intersect_mode mode) {
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

  tf::face_cuts<Index, Int> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements(ig, fc, forms, conv);

  return std::make_tuple(std::move(mesh), std::move(tag_labels),
                         std::move(face_labels));
}

template <typename Int, typename FormsRange>
auto mesh_arrangements_n_curves(const FormsRange &forms,
                                tf::intersect_mode mode) {
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

  tf::face_cuts<Index, Int> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements(ig, fc, forms, conv);

  auto paths = [&]() {
    if (mode & tf::intersect_mode::sos) {
      auto edge_pairs = tf::make_mapped_range(
          ig.edge_groups(), [](const auto &group) -> std::array<Index, 2> {
            return {group[0].point_0, group[0].point_1};
          });
      return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
    } else {
      tf::cut_graph<Index> cg;
      cg.build(fc, static_cast<Index>(ig.points().size()));
      return tf::connect_edges_to_paths(
          tf::make_edges(cg.intersection_edges()));
    }
  }();

  tf::curves_buffer<Index, RealType, 3> cb;
  cb.paths_buffer() = std::move(paths);
  auto ipts = ig.points();
  cb.points_buffer().allocate(ipts.size());
  tf::parallel_copy(
      tf::make_points(tf::make_mapped_range(
          ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
      cb.points());

  return std::make_tuple(std::move(mesh), std::move(tag_labels),
                         std::move(face_labels), std::move(cb));
}

} // namespace cut

/// @ingroup cut_boolean
/// @brief Build a single merged mesh from N intersected meshes.
template <typename Int = tf::exact::int32, typename Range>
auto make_mesh_arrangements(
    const Range &_forms,
    tf::intersect_mode mode = tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours) {
  return cut::dispatch::arrangement(_forms, [mode](const auto &forms) {
    return cut::mesh_arrangements_n<Int>(forms, mode);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with intersection curves.
template <typename Int = tf::exact::int32, typename Range>
auto make_mesh_arrangements(const Range &_forms, tf::intersect_mode mode,
                            tf::return_curves_t) {
  return cut::dispatch::arrangement(_forms, [mode](const auto &forms) {
    return cut::mesh_arrangements_n_curves<Int>(forms, mode);
  });
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with curves (default intersect mode).
template <typename Int = tf::exact::int32, typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_curves_t) {
  return make_mesh_arrangements<Int>(
      forms,
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours,
      tf::return_curves);
}

} // namespace tf
