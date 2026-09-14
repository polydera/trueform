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

#include "../arrangement/graph_builders.hpp"
#include "config_validation.hpp"
#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"
#include "trueform/intersect/make_intersection_curves.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace intersect_detail {

template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto intersection_curves_pair_impl(const mesh<Index0, Real, 3, Ngon0> &a,
                                   const mesh<Index1, Real, 3, Ngon1> &b,
                                   tf::intersect_config config)
    -> tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3> {
  require_config<Real>(config, "intersection_curves");
  if (a.number_of_faces() == 0 || b.number_of_faces() == 0)
    return {};
  return tf::intersect::curves_worker<tf::none_t>(
      graph_builders::build_pair_arrangement(a.topology_form(),
                                             b.topology_form(),
                                             tf::arrangement_config{config}));
}

template <typename Index, typename Real, std::size_t Ngon>
auto intersection_curves_list_impl(
    const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
    tf::intersect_config config) -> tf::curves_buffer<Index, Real, 3> {
  require_config<Real>(config, "intersection_curves");
  if (meshes.empty())
    throw std::invalid_argument("intersection_curves: mesh list is empty");

  std::vector<graph_builders::form_t<Index, Real, Ngon>> forms;
  forms.reserve(meshes.size());
  for (const auto &operand : meshes)
    if (operand.number_of_faces() != 0)
      forms.push_back(operand.topology_form());
  if (forms.size() < 2)
    return {};
  return tf::intersect::curves_worker<tf::none_t>(
      graph_builders::build_range_arrangement(
          graph_builders::forms_range(forms), tf::arrangement_config{config}));
}

template <typename Index, typename Real, std::size_t Ngon>
auto self_intersection_curves_impl(const mesh<Index, Real, 3, Ngon> &value,
                                   tf::intersect_config config)
    -> tf::curves_buffer<Index, Real, 3> {
  require_config<Real>(config, "self_intersection_curves");
  if (value.number_of_faces() == 0)
    return {};
  // The curves are the operand's own, so it is read where it was authored;
  // the view is bound, because the graph holds the form taken from it.
  const auto operand = value.at_identity();
  return tf::intersect::curves_worker<tf::none_t>(
      graph_builders::build_self_arrangement(operand.topology_form(),
                                             tf::arrangement_config{config}));
}

} // namespace intersect_detail

template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto intersection_curves(const mesh<Index0, Real, 3, Ngon0> &a,
                         const mesh<Index1, Real, 3, Ngon1> &b,
                         tf::intersect_config config)
    -> tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3> {
  return intersect_detail::intersection_curves_pair_impl(a, b, config);
}

template <typename Index, typename Real, std::size_t Ngon>
auto intersection_curves(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                         tf::intersect_config config)
    -> tf::curves_buffer<Index, Real, 3> {
  return intersect_detail::intersection_curves_list_impl(meshes, config);
}

template <typename Index, typename Real, std::size_t Ngon>
auto self_intersection_curves(const mesh<Index, Real, 3, Ngon> &value,
                              tf::intersect_config config)
    -> tf::curves_buffer<Index, Real, 3> {
  return intersect_detail::self_intersection_curves_impl(value, config);
}

} // namespace tf::cpp
