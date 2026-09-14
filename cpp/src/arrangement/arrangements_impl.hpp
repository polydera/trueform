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
#include "trueform/arrangement/make_mesh_arrangements.hpp"
#include "trueform/arrangement/return_curves.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/none.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/arrangement/mesh_arrangements.hpp"
#include "trueform/cpp/arrangement/polygon_arrangements.hpp"

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace arrangement_detail {

template <typename Meshes> auto require_operands(const Meshes &meshes) -> void {
  if (meshes.size() < 2)
    throw std::runtime_error("mesh_arrangements: need at least 2 meshes");
  for (const auto &value : meshes)
    value.require_indices();
}

template <typename Meshes> auto all_empty(const Meshes &meshes) -> bool {
  for (const auto &value : meshes)
    if (value.number_of_faces() != 0)
      return false;
  return true;
}

/// The labels of nothing: a flat run with no entries.
template <typename Index> auto empty_labels() -> nd_array<Index> {
  tf::buffer<Index> buffer;
  return nd_array<Index>::from_buffer(std::move(buffer), {0});
}

/// A range is homogeneous, so its element's arity is the operands' own and
/// the result carries that layout. The curves tag is the request core's own
/// worker is parameterised on, so the plain entry does not pay for a seam
/// extraction it never hands back.
template <typename Curves, typename Index, typename Real, std::size_t Ngon>
auto mesh_arrangements_of(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                          tf::arrangement_config config) {
  constexpr auto want_curves = std::is_same_v<Curves, tf::return_curves_t>;
  using result_type =
      std::conditional_t<want_curves,
                         mesh_arrangement_with_curves_result<Index, Real, Ngon>,
                         mesh_arrangement_result<Index, Real, Ngon>>;

  intersect_detail::require_config<Real>(config.intersect, "mesh_arrangements",
                                         true);
  require_triangulation(config.triangulation, "mesh_arrangements");
  require_operands(meshes);
  if (all_empty(meshes)) {
    if constexpr (want_curves)
      return result_type{tf::polygons_buffer<Index, Real, 3, Ngon>{},
                         empty_labels<Index>(),
                         empty_labels<Index>(),
                         {}};
    else
      return result_type{tf::polygons_buffer<Index, Real, 3, Ngon>{},
                         empty_labels<Index>(), empty_labels<Index>()};
  }

  std::vector<graph_builders::form_t<Index, Real, Ngon>> forms;
  forms.reserve(meshes.size());
  for (const auto &operand : meshes)
    forms.push_back(operand.topology_form());
  auto product =
      tf::arrangement::arrangement_worker<tf::none_t, Curves, tf::none_t>(
          graph_builders::build_range_arrangement(
              graph_builders::forms_range(forms), config));
  auto &output = std::get<0>(product);
  auto &tag_labels = std::get<1>(product);
  auto &face_labels = std::get<2>(product);
  const auto tag_count = static_cast<int>(tag_labels.size());
  const auto face_count = static_cast<int>(face_labels.size());
  if constexpr (want_curves)
    return result_type{
        std::move(output),
        nd_array<Index>::from_buffer(std::move(tag_labels), {tag_count}),
        nd_array<Index>::from_buffer(std::move(face_labels), {face_count}),
        std::move(std::get<3>(product))};
  else
    return result_type{
        std::move(output),
        nd_array<Index>::from_buffer(std::move(tag_labels), {tag_count}),
        nd_array<Index>::from_buffer(std::move(face_labels), {face_count})};
}

template <typename Curves, typename Index, typename Real, std::size_t Ngon>
auto polygon_arrangements_of(const mesh<Index, Real, 3, Ngon> &value,
                             tf::arrangement_config config) {
  constexpr auto want_curves = std::is_same_v<Curves, tf::return_curves_t>;
  using result_type = std::conditional_t<
      want_curves, polygon_arrangement_with_curves_result<Index, Real, Ngon>,
      polygon_arrangement_result<Index, Real, Ngon>>;

  intersect_detail::require_config<Real>(config.intersect,
                                         "polygon_arrangements", true);
  require_triangulation(config.triangulation, "polygon_arrangements");
  value.require_indices();
  if (value.number_of_faces() == 0) {
    if constexpr (want_curves)
      return result_type{tf::polygons_buffer<Index, Real, 3, Ngon>{},
                         empty_labels<Index>(),
                         {}};
    else
      return result_type{tf::polygons_buffer<Index, Real, 3, Ngon>{},
                         empty_labels<Index>()};
  }

  // The arrangement is the operand's own, so it is read where it was
  // authored. The view is bound, because the graph holds the form taken
  // from it.
  const auto operand = value.at_identity();
  auto product =
      tf::arrangement::arrangement_worker<tf::none_t, Curves, tf::none_t>(
          graph_builders::build_self_arrangement(operand.topology_form(),
                                                 config));
  auto &output = std::get<0>(product);
  auto &face_labels = std::get<1>(product);
  const auto face_count = static_cast<int>(face_labels.size());
  if constexpr (want_curves)
    return result_type{
        std::move(output),
        nd_array<Index>::from_buffer(std::move(face_labels), {face_count}),
        std::move(std::get<2>(product))};
  else
    return result_type{
        std::move(output),
        nd_array<Index>::from_buffer(std::move(face_labels), {face_count})};
}

} // namespace arrangement_detail

template <typename Index, typename Real, std::size_t Ngon>
auto mesh_arrangements(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                       tf::arrangement_config config)
    -> mesh_arrangement_result<Index, Real, Ngon> {
  return arrangement_detail::mesh_arrangements_of<tf::none_t>(meshes, config);
}

template <typename Index, typename Real, std::size_t Ngon>
auto mesh_arrangements_with_curves(
    const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
    tf::arrangement_config config)
    -> mesh_arrangement_with_curves_result<Index, Real, Ngon> {
  return arrangement_detail::mesh_arrangements_of<tf::return_curves_t>(meshes,
                                                                       config);
}

template <typename Index, typename Real, std::size_t Ngon>
auto polygon_arrangements(const mesh<Index, Real, 3, Ngon> &value,
                          tf::arrangement_config config)
    -> polygon_arrangement_result<Index, Real, Ngon> {
  return arrangement_detail::polygon_arrangements_of<tf::none_t>(value, config);
}

template <typename Index, typename Real, std::size_t Ngon>
auto polygon_arrangements_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                                      tf::arrangement_config config)
    -> polygon_arrangement_with_curves_result<Index, Real, Ngon> {
  return arrangement_detail::polygon_arrangements_of<tf::return_curves_t>(
      value, config);
}

} // namespace tf::cpp
