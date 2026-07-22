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
#include "../cut/dispatch/self_boolean.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../cut/construct/extract_intersection_curves.hpp"
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

  tf::curves_buffer<Index, RealOut, 3> cb;
  if (config.mode & tf::intersect_mode::sos) {
    tf::buffer<std::array<Index, 2>> sos_edges;
    sos_edges.allocate(std::size_t(ig.edge_groups().size()));
    tf::parallel_transform(
        ig.edge_groups(), tf::make_range(sos_edges),
        [](const auto &group) -> std::array<Index, 2> {
          return {group[0].point_0, group[0].point_1};
        });
    cb = tf::cut::curves_from_seam_edges<RealOut, Index>(sos_edges,
                                                         ig.points(), conv);
  } else {
    // regions, their coplanar collapse, and the seam scan — a single
    // tag, so the non-manifold rule finds the self seams
    tf::buffer<Index> point_counts;
    point_counts.allocate(1);
    point_counts[0] = static_cast<Index>(p.points().size());
    cb = tf::cut::make_region_curves<RealOut, Index>(
        ig, apply_to_face, get_mesh_point, tf::make_range(point_counts), conv);
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
