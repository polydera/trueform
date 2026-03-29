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
#include "../topology/connect_edges_to_paths.hpp"
#include "./boolean_config.hpp"
#include "./boolean_op.hpp"
#include "./construct/make_boolean.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Perform boolean operations on two meshes.
///
/// Computes union, intersection, or difference of two polygon meshes.
/// Uses exact integer predicates for robust classification.
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::boolean_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [op, config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType>(p0, p1);
        return tf::cut::make_boolean<int, Index>(
            p0, p1, ig, fc, cg, ibp.converter(),
            tf::cut::make_boolean_op_spec(op), config);
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with face origin mapping.
/// @overload
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_index_map_t,
                  tf::boolean_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [op, config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType>(p0, p1);
        return tf::cut::make_boolean<int, Index>(
            p0, p1, ig, fc, cg, ibp.converter(),
            tf::cut::make_boolean_op_spec(op), config, tf::return_index_map);
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with curve output.
/// @overload
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t,
                  tf::boolean_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [op, config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType>(p0, p1);
        auto res = tf::cut::make_boolean<int, Index>(
            p0, p1, ig, fc, cg, ibp.converter(),
            tf::cut::make_boolean_op_spec(op), config);

        auto paths = tf::connect_edges_to_paths(
            tf::make_edges(cg.intersection_edges()));
        auto &conv = ibp.converter();
        auto ipts = ig.points();
        tf::curves_buffer<Index, RealType, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
            cb.points());
        return std::make_tuple(std::move(res.first), std::move(res.second),
                               std::move(cb));
      });
}

/// @ingroup cut_boolean
/// @brief Perform boolean operations with curves and face origin mapping.
/// @overload
template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t, tf::return_index_map_t,
                  tf::boolean_config config = {}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [op, config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType>(p0, p1);
        auto [res_mesh, res_labels, res_im] =
            tf::cut::make_boolean<int, Index>(
                p0, p1, ig, fc, cg, ibp.converter(),
                tf::cut::make_boolean_op_spec(op), config,
                tf::return_index_map);

        auto paths = tf::connect_edges_to_paths(
            tf::make_edges(cg.intersection_edges()));
        auto &conv = ibp.converter();
        auto ipts = ig.points();
        tf::curves_buffer<Index, RealType, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
            cb.points());
        return std::make_tuple(std::move(res_mesh), std::move(res_labels),
                               std::move(cb), std::move(res_im));
      });
}

} // namespace tf

