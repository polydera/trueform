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
#include "../intersect/exact/intersections_between_polygons.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./boolean_config.hpp"
#include "./face_cuts.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/make_mesh_arrangement.hpp"
#include "./impl/make_mesh_arrangements.hpp"
#include "./return_curves.hpp"
#include "./tagged_cut_faces.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
///
/// Returns all subdivided regions created by mesh intersection,
/// each classified by origin and spatial relationship (inside/outside).
/// This is the complete decomposition from which any boolean operation
/// can be reconstructed.
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @return Tuple of (vector of @ref tf::polygons_buffer, labels, @ref
/// tf::arrangement_class).
///
/// @see tf::make_boolean for combined boolean results.
/// @see tf::arrangement_class for classification values.
template <typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::boolean_config config = {}) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_mesh_arrangements<int>(p0, p1, ibp, tcf, config);
      });
}

/// @ingroup cut_boolean
/// @brief Build a single merged mesh from N intersected meshes.
///
/// Splits all faces along intersection curves and merges into one mesh.
/// Returns (mesh, tag_labels, face_labels) where tag_labels identifies
/// which input mesh each face came from, and face_labels identifies which
/// face in that mesh.
///
/// @tparam Range A range of tagged polygon forms (with tree + fm + mel).
/// @param forms The input meshes.
/// @return Tuple of (polygons_buffer, tag labels, face labels).
template <typename Range>
auto make_mesh_arrangement(
    const Range &forms,
    tf::intersect_mode mode = tf::intersect_mode::primitives) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using RealType =
      tf::coordinate_type<decltype(forms[0])>;

  tf::exact::intersections_between_polygons<Index, RealType> ibp;
  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto get_face = [&](int tag, int object) {
    return forms[tag].faces()[object];
  };
  auto get_mesh_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    return conv.convert(forms[tag].points()[id]);
  };

  tf::intersection_graph<Index> ig;
  ig.build(ibp, get_face, get_mesh_point);

  tf::face_cuts<Index> fc;
  fc.build(ig, get_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangement<Index>(ig, fc, forms, conv);

  return std::make_tuple(std::move(mesh), std::move(tag_labels),
                         std::move(face_labels));
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with intersection curves.
///
/// Same as make_mesh_arrangement, but also returns the intersection
/// curves as a curves_buffer.
///
/// @tparam Range A range of tagged polygon forms.
/// @param forms The input meshes.
/// @param tag Pass @ref tf::return_curves to get intersection curves.
/// @return Tuple of (polygons_buffer, tag labels, face labels, curves_buffer).
template <typename Range>
auto make_mesh_arrangement(const Range &forms, tf::intersect_mode mode,
                           tf::return_curves_t) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using RealType = tf::coordinate_type<decltype(forms[0])>;

  tf::exact::intersections_between_polygons<Index, RealType> ibp;
  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto get_face = [&](int tag, int object) {
    return forms[tag].faces()[object];
  };
  auto get_mesh_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    return conv.convert(forms[tag].points()[id]);
  };

  tf::intersection_graph<Index> ig;
  ig.build(ibp, get_face, get_mesh_point);

  tf::face_cuts<Index> fc;
  fc.build(ig, get_face, get_mesh_point);

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangement<Index>(ig, fc, forms, conv);

  // Extract curves from edge groups
  auto edge_pairs = tf::make_mapped_range(
      ig.edge_groups(), [](const auto &group) -> std::array<Index, 2> {
        return {group[0].point_0, group[0].point_1};
      });
  auto paths = tf::connect_edges_to_paths(tf::make_edges(edge_pairs));

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

/// @ingroup cut_boolean
/// @brief Build a merged mesh with curves (default intersect mode).
template <typename Range>
auto make_mesh_arrangement(const Range &forms, tf::return_curves_t) {
  return make_mesh_arrangement(forms, tf::intersect_mode::primitives,
                               tf::return_curves);
}

} // namespace tf
