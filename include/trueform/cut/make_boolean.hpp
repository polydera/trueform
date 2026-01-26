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
#include "../core/curves_buffer.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./boolean_op.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/make_boolean.hpp"
#include "./return_curves.hpp"
#include "./tagged_cut_faces.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Perform boolean operations on two meshes.
///
/// Computes union, intersection, or difference of two polygon meshes.
/// Accepts plain @ref tf::polygons or forms with precomputed tree policy
/// (@ref tf::tree or @ref tf::mod_tree) and topology policies
/// (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform (merge, intersection, left_difference, right_difference).
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
///
/// @see tf::make_boolean_pair for separate left/right results.
/// @see tf::make_mesh_arrangements for full region decomposition.
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_boolean<int>(p0, p1, ibp, tcf,
                                          tf::cut::make_boolean_op_spec(op));
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with face origin mapping.
/// @overload
///
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform.
/// @param tag Pass @ref tf::return_index_map to get face origin indices.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer, index map).
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_index_map_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_boolean<int>(p0, p1, ibp, tcf,
                                          tf::cut::make_boolean_op_spec(op),
                                          tf::return_index_map);
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with curve output.
/// @overload
///
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform.
/// @param tag Pass @ref tf::return_curves to get intersection curves.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer, @ref tf::curves_buffer).
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        auto res = tf::cut::make_boolean<int>(
            p0, p1, ibp, tcf, tf::cut::make_boolean_op_spec(op));
        auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
          return std::array<Index, 2>{e[0].id, e[1].id};
        });
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index,
                          tf::coordinate_type<std::decay_t<decltype(p0)>,
                                              std::decay_t<decltype(p1)>>,
                          3>
            cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return std::make_tuple(std::move(res.first), std::move(res.second),
                               std::move(cb));
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with curves and face origin mapping.
/// @overload
///
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform.
/// @param tag0 Pass @ref tf::return_curves to get intersection curves.
/// @param tag1 Pass @ref tf::return_index_map to get face origin indices.
/// @return Tuple of (@ref tf::polygons_buffer, labels, @ref tf::curves_buffer, index map).
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t, tf::return_index_map_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        auto [res_mesh, res_labels, res_im] = tf::cut::make_boolean<int>(
            p0, p1, ibp, tcf, tf::cut::make_boolean_op_spec(op),
            tf::return_index_map);
        auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
          return std::array<Index, 2>{e[0].id, e[1].id};
        });
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index,
                          tf::coordinate_type<std::decay_t<decltype(p0)>,
                                              std::decay_t<decltype(p1)>>,
                          3>
            cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return std::make_tuple(std::move(res_mesh), std::move(res_labels),
                               std::move(cb), std::move(res_im));
      });
}

} // namespace tf
