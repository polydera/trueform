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

#include "../arrangement/config_validation.hpp"
#include "../arrangement/graph_builders.hpp"
#include "../intersect/config_validation.hpp"
#include "boolean_guards.hpp"
#include "trueform/arrangement/return_curves.hpp"
#include "trueform/core/none.hpp"
#include "trueform/cpp/csg/make_boolean.hpp"
#include "trueform/csg/expression/make_boolean_expr.hpp"
#include "trueform/csg/graph/make_boolean_labels.hpp"
#include "trueform/csg/make_csg_mesh.hpp"
#include "trueform/csg/make_intersection_curves.hpp"
#include "trueform/reindex/return_source_ids.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace boolean_detail {

/// A boolean is one arrangement of the two operands answering one expression
/// against it, so the build is the graph tier's and this reads it. The curves
/// tag is the request, so the plain entry does not pay for a seam extraction
/// it never hands back.
template <typename Curves, typename Index0, typename Real, typename Index1,
          std::size_t Ngon0, std::size_t Ngon1>
auto boolean_of(const mesh<Index0, Real, 3, Ngon0> &a,
                const mesh<Index1, Real, 3, Ngon1> &b, tf::boolean_op operation,
                std::vector<std::int32_t> sheets,
                tf::arrangement_config config) {
  using OutputIndex = common_index_t<Index0, Index1>;
  constexpr auto OutputNgon = detail::concatenated_arity_v<Ngon0, Ngon1>;
  constexpr auto want_curves = std::is_same_v<Curves, tf::return_curves_t>;
  using result_type = std::conditional_t<
      want_curves, boolean_with_curves_result<OutputIndex, Real, OutputNgon>,
      boolean_result<OutputIndex, Real, OutputNgon>>;

  require_boolean_operation(operation);
  require_sheets(sheets);
  intersect_detail::require_config<Real>(config.intersect, "make_boolean",
                                         true);
  arrangement_detail::require_triangulation(config.triangulation,
                                            "make_boolean");
  auto graph = graph_builders::build_pair_csg_graph(
      a.topology_form(), b.topology_form(), graph_builders::sheets_of(sheets),
      config);
  auto [polygons, tag_labels, face_labels] = tf::make_csg_mesh<Real>(
      graph, tf::csg::make_boolean_expr(operation), tf::return_source_ids);
  auto labels = nd_array<std::int8_t>::from_buffer(
      tf::csg::graph::make_boolean_labels(tag_labels));
  auto sources = nd_array<OutputIndex>::from_buffer(std::move(face_labels));
  if constexpr (want_curves)
    return result_type{std::move(polygons), std::move(labels),
                       std::move(sources),
                       tf::make_intersection_curves<Real>(graph)};
  else
    return result_type{std::move(polygons), std::move(labels),
                       std::move(sources)};
}

} // namespace boolean_detail

template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto make_boolean(const mesh<Index0, Real, 3, Ngon0> &a,
                  const mesh<Index1, Real, 3, Ngon1> &b,
                  tf::boolean_op operation, std::vector<std::int32_t> sheets,
                  tf::arrangement_config config)
    -> boolean_result<common_index_t<Index0, Index1>, Real,
                      detail::concatenated_arity_v<Ngon0, Ngon1>> {
  return boolean_detail::boolean_of<tf::none_t>(a, b, operation,
                                                std::move(sheets), config);
}

template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto make_boolean_with_curves(const mesh<Index0, Real, 3, Ngon0> &a,
                              const mesh<Index1, Real, 3, Ngon1> &b,
                              tf::boolean_op operation,
                              std::vector<std::int32_t> sheets,
                              tf::arrangement_config config)
    -> boolean_with_curves_result<common_index_t<Index0, Index1>, Real,
                                  detail::concatenated_arity_v<Ngon0, Ngon1>> {
  return boolean_detail::boolean_of<tf::return_curves_t>(
      a, b, operation, std::move(sheets), config);
}

} // namespace tf::cpp
