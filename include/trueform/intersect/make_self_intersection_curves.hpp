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
#include "../cut/cut_graph.hpp"
#include "../cut/dispatch/self_boolean.hpp"
#include "../cut/face_cuts.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./exact/make_kernel.hpp"
#include "./graph/intersection_graph.hpp"
#include "./intersect_config.hpp"
#include "./intersections_within_polygons.hpp"

namespace tf {

namespace intersect {

template <typename Int, typename OutputCoordinateType, typename Policy>
auto self_intersection_curves(const tf::polygons<Policy> &p,
                              tf::intersect_config config) {
  using Index = std::decay_t<decltype(p.faces()[0][0])>;
  using InputReal = tf::coordinate_type<Policy>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");

  tf::intersections_within_polygons<Index, PipelineReal, ResolvedInt> iwp;
  iwp.build(p, config);

  auto &conv = iwp.converter();
  auto apply_to_face = [&](int, Index object, const auto &f) {
    f(p.faces()[object]);
  };
  auto get_mesh_point = [&](int, Index id) -> tf::point<ResolvedInt, 3> {
    return conv.convert(p.points()[id]);
  };

  tf::intersection_graph<Index, ResolvedInt> ig;
  ig.build(iwp, apply_to_face, get_mesh_point, config.mode,
           tf::exact::make_kernel(conv, config.tolerance));

  auto paths = [&]() {
    if (config.mode & tf::intersect_mode::sos) {
      auto edge_pairs = tf::make_mapped_range(
          ig.edge_groups(), [](const auto &group) -> std::array<Index, 2> {
            return {group[0].point_0, group[0].point_1};
          });
      return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
    } else {
      tf::face_cuts<Index, ResolvedInt> fc;
      fc.build(ig, apply_to_face, get_mesh_point);

      tf::cut_graph<Index> cg;
      cg.build(fc, ig, p);

      return tf::connect_edges_to_paths(
          tf::make_edges(cg.intersection_edges()));
    }
  }();

  auto ipts = ig.points();
  tf::curves_buffer<Index, RealOut, 3> cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(ipts.size());
  if constexpr (std::is_integral_v<RealOut>) {
    tf::parallel_copy(tf::make_points(ipts), cb.points());
  } else {
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
        cb.points());
  }
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    auto conv_copy = iwp.converter();
    return std::make_tuple(std::move(cb), std::move(conv_copy));
  } else {
    return cb;
  }
}

} // namespace intersect

/// @ingroup intersect_curves
/// @brief Extract curves where a mesh intersects itself.
///
/// Finds all locations where a mesh's faces intersect each other
/// (excluding adjacent faces) and returns the result as connected curves.
///
/// @tparam Policy The policy type for the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected self-intersection
/// curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_self_intersection_curves(
    const tf::polygons<Policy> &_polygons,
    tf::intersect_config config = {tf::intersect_mode::sos |
                                   tf::intersect_mode::resolve_contours}) {
  return cut::dispatch::self_boolean(_polygons, [config](const auto &p) {
    return intersect::self_intersection_curves<Int, OutputCoordinateType>(
        p, config);
  });
}

} // namespace tf
