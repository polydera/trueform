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
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/csg/expression/selection.hpp"
#include "trueform/csg/expression/selection_kind.hpp"
#include "trueform/csg/make_csg_domains.hpp"
#include "trueform/csg/make_csg_mesh.hpp"
#include "trueform/csg/make_intersection_curves.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace csg_graph_detail {

inline auto require_domain_config(tf::domain_config config) -> void {
  constexpr auto known =
      static_cast<int>(tf::domain_config::exclude_outer_shell) |
      static_cast<int>(tf::domain_config::ignore_open_fragments);
  if ((static_cast<int>(config) & ~known) != 0)
    throw std::invalid_argument("make_csg_domains: invalid domain config");
}

inline auto require_expression(const tf::csg::expr &expression,
                               std::size_t number_of_tags) -> void {
  if (expression.node_kind() == tf::csg::expr::kind::leaf) {
    const auto operand = expression.operand_id();
    if (operand < 0 || static_cast<std::size_t>(operand) >= number_of_tags)
      throw std::out_of_range("csg expression operand is out of range");
    return;
  }
  const auto &children = expression.children();
  if (expression.node_kind() == tf::csg::expr::kind::complement &&
      children.size() != 1)
    throw std::invalid_argument(
        "csg complement expression must have exactly one child");
  for (const auto &child : children)
    require_expression(child, number_of_tags);
}

inline auto require_selection(const tf::csg::selection_t &selection,
                              std::size_t number_of_tags) -> void {
  for (const auto tag : selection.tags())
    if (tag < 0 || static_cast<std::size_t>(tag) >= number_of_tags)
      throw std::out_of_range("csg selection tag is out of range");
  if (selection.has_expression())
    require_expression(selection.expression(), number_of_tags);
}

inline auto require_boundary_selection(const tf::csg::selection_t &selection)
    -> void {
  if (selection.kind() != tf::csg::selection_kind::boundary)
    throw std::invalid_argument(
        "make_csg_domains: selection must be a boundary read");
}

template <typename Index, typename Real>
auto require_meshes(const std::vector<cpp::mesh<Index, Real, 3, 3>> &meshes,
                    const std::vector<std::int32_t> &sheets,
                    tf::arrangement_config config) -> void {
  if (meshes.size() < 2)
    throw std::invalid_argument("make_csg_graph: need at least two meshes");
  if (meshes.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::length_error("make_csg_graph: too many meshes");
  intersect_detail::require_config<Real>(config.intersect, "make_csg_graph",
                                         true);
  arrangement_detail::require_triangulation(config.triangulation,
                                            "make_csg_graph");
  for (const auto sheet : sheets)
    if (sheet < 0 || static_cast<std::size_t>(sheet) >= meshes.size())
      throw std::out_of_range("make_csg_graph: sheet index out of range");
}

template <typename Graph> auto require_graph(const Graph &graph) -> void {
  if (!graph.is_valid())
    throw std::invalid_argument("csg graph must be valid");
}

template <typename Index>
auto blocks_to_arrays(tf::offset_block_buffer<Index, Index> &&blocks) {
  return std::make_pair(
      nd_array<Index>::from_buffer(std::move(blocks.offsets_buffer())),
      nd_array<Index>::from_buffer(std::move(blocks.data_buffer())));
}

inline auto
inclusion_to_array(const tf::blocked_buffer<bool, tf::dynamic_size> &inclusion)
    -> nd_array<std::int8_t> {
  tf::buffer<std::int8_t> values;
  values.allocate(inclusion.data_buffer().size());
  tf::parallel_transform(
      tf::make_range(inclusion.data_buffer()), tf::make_range(values),
      [](bool value) { return value ? std::int8_t{1} : std::int8_t{0}; },
      tf::checked);
  return nd_array<std::int8_t>::from_buffer(
      std::move(values), {static_cast<int>(inclusion.size()),
                          static_cast<int>(inclusion.block_size())});
}

/// The cells as the result states them: core's own storage, in the container
/// the facade hands back.
template <typename Index, typename Real, typename Cells>
auto cells_to_meshes(Cells &cells)
    -> std::vector<tf::polygons_buffer<Index, Real, 3, 3>> {
  std::vector<tf::polygons_buffer<Index, Real, 3, 3>> meshes;
  meshes.reserve(cells.size());
  for (auto &cell : cells)
    meshes.push_back(std::move(cell));
  return meshes;
}

/// The graph reads the operands it was built from for as long as it lives, so
/// it holds their views: each retains its own storage and its structures.
template <typename Index, typename Real> struct graph_data {
  using mesh_type = cpp::mesh<Index, Real, 3, 3>;
  using form_type = graph_builders::form_t<Index, Real, 3>;
  using graph_type = graph_builders::range_csg_graph_t<
      graph_builders::forms_range_t<Index, Real, 3>>;

  std::vector<mesh_type> meshes;
  std::vector<form_type> forms;
  graph_type graph;

  static auto make_forms(const std::vector<mesh_type> &values)
      -> std::vector<form_type> {
    std::vector<form_type> result;
    result.reserve(values.size());
    for (const auto &value : values)
      result.push_back(value.topology_form());
    return result;
  }

  graph_data(std::vector<mesh_type> values,
             const std::vector<std::int32_t> &sheets,
             tf::arrangement_config config)
      : meshes(std::move(values)), forms(make_forms(meshes)),
        graph(graph_builders::build_range_csg_graph(
            graph_builders::forms_range(forms),
            graph_builders::sheets_of(sheets), config)) {}
};

} // namespace csg_graph_detail

template <typename Index, typename Real>
struct csg_graph<Index, Real>::data
    : csg_graph_detail::graph_data<Index, Real> {
  using csg_graph_detail::graph_data<Index, Real>::graph_data;
};

namespace csg_graph_detail {

struct access {
  template <typename Index, typename Real>
  static auto of(const csg_graph<Index, Real> &graph) -> const
      typename csg_graph<Index, Real>::data & {
    require_graph(graph);
    return *graph._data;
  }
};

} // namespace csg_graph_detail

template <typename Index, typename Real>
csg_graph<Index, Real>::csg_graph() = default;

template <typename Index, typename Real>
auto csg_graph<Index, Real>::create(std::vector<mesh<Index, Real, 3, 3>> meshes,
                                    std::vector<std::int32_t> sheets,
                                    tf::arrangement_config config)
    -> csg_graph {
  csg_graph_detail::require_meshes(meshes, sheets, config);
  csg_graph result;
  result._data =
      std::make_shared<data>(std::move(meshes), sheets, std::move(config));
  return result;
}

template <typename Index, typename Real>
auto csg_graph<Index, Real>::is_valid() const -> bool {
  return static_cast<bool>(_data);
}

template <typename Index, typename Real>
auto csg_graph<Index, Real>::destroy() -> void {
  _data.reset();
}

template <typename Index, typename Real>
auto make_csg_graph(std::vector<mesh<Index, Real, 3, 3>> meshes,
                    std::vector<std::int32_t> sheets,
                    tf::arrangement_config config) -> csg_graph<Index, Real> {
  return csg_graph<Index, Real>::create(std::move(meshes), std::move(sheets),
                                        config);
}

template <typename Index, typename Real>
auto csg_created_points(const csg_graph<Index, Real> &graph) -> nd_array<Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  const auto &created = data.graph.created_points();
  tf::buffer<Real> output;
  output.allocate(created.size() * 3);
  const auto &converter = data.graph.converter();
  tf::parallel_transform(tf::make_range(created),
                         tf::make_blocked_range<3>(tf::make_range(output)),
                         [&](const auto &point) {
                           const auto converted = converter.deconvert(point);
                           return std::array<Real, 3>{Real(converted[0]),
                                                      Real(converted[1]),
                                                      Real(converted[2])};
                         });
  return nd_array<Real>::from_buffer(std::move(output),
                                     {static_cast<int>(created.size()), 3});
}

template <typename Index, typename Real>
auto csg_intersection_curves(const csg_graph<Index, Real> &graph)
    -> tf::curves_buffer<Index, Real, 3> {
  const auto &data = csg_graph_detail::access::of(graph);
  return tf::make_intersection_curves(data.graph);
}

template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  const auto &data = csg_graph_detail::access::of(graph);
  return tf::make_csg_mesh(data.graph);
}

template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph)
    -> csg_mesh_labeled_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  auto [output, tag_labels, face_labels] =
      tf::make_csg_mesh(data.graph, tf::return_source_ids);
  return {std::move(output),
          nd_array<Index>::from_buffer(std::move(tag_labels)),
          nd_array<Index>::from_buffer(std::move(face_labels))};
}

template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph,
                   const tf::csg::selection_t &selection)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  return tf::make_csg_mesh(data.graph, selection);
}

template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph,
                               const tf::csg::selection_t &selection)
    -> csg_mesh_labeled_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  auto [output, tag_labels, face_labels] =
      tf::make_csg_mesh(data.graph, selection, tf::return_source_ids);
  return {std::move(output),
          nd_array<Index>::from_buffer(std::move(tag_labels)),
          nd_array<Index>::from_buffer(std::move(face_labels))};
}

template <typename Index, typename Real>
auto make_csg_mesh_with_index_map(const csg_graph<Index, Real> &graph,
                                  const tf::csg::selection_t &selection)
    -> csg_mesh_index_map_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  auto [output, map] =
      tf::make_csg_mesh(data.graph, selection, tf::return_index_map);
  auto [point_f_offsets, point_f_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.point_f));
  const auto n_uncut_tags = static_cast<int>(map.uncut_faces.size());
  return {std::move(output),
          nd_array<Index>::from_buffer(std::move(map.point_tag_labels)),
          nd_array<Index>::from_buffer(std::move(map.point_labels)),
          nd_array<Index>::from_buffer(std::move(map.face_tag_labels)),
          nd_array<Index>::from_buffer(std::move(map.face_labels)),
          std::move(point_f_offsets),
          std::move(point_f_data),
          nd_array<Index>::from_buffer(std::move(map.uncut_faces.data_buffer()),
                                       {n_uncut_tags, 2}),
          static_cast<Index>(map.n_original_points),
          static_cast<Index>(map.n_tags),
          static_cast<Index>(map.n_output_points)};
}

template <typename Index, typename Real>
auto make_csg_domains(const csg_graph<Index, Real> &graph,
                      tf::domain_config config)
    -> csg_domains_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids] = tf::make_csg_domains(data.graph, config);
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids))};
}

template <typename Index, typename Real>
auto make_csg_domains(const csg_graph<Index, Real> &graph,
                      const tf::csg::selection_t &selection,
                      tf::domain_config config)
    -> csg_domains_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  csg_graph_detail::require_boundary_selection(selection);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids] = tf::make_csg_domains(data.graph, selection, config);
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids))};
}

template <typename Index, typename Real>
auto make_csg_domains_with_labels(const csg_graph<Index, Real> &graph,
                                  tf::domain_config config)
    -> csg_domains_labeled_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids, tag_blocks, face_blocks] =
      tf::make_csg_domains(data.graph, config, tf::return_source_ids);
  auto [tag_offsets, tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(tag_blocks));
  auto [face_offsets, face_data] =
      csg_graph_detail::blocks_to_arrays(std::move(face_blocks));
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids)),
          std::move(tag_offsets),
          std::move(tag_data),
          std::move(face_offsets),
          std::move(face_data)};
}

template <typename Index, typename Real>
auto make_csg_domains_with_labels(const csg_graph<Index, Real> &graph,
                                  const tf::csg::selection_t &selection,
                                  tf::domain_config config)
    -> csg_domains_labeled_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  csg_graph_detail::require_boundary_selection(selection);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids, tag_blocks, face_blocks] = tf::make_csg_domains(
      data.graph, selection, config, tf::return_source_ids);
  auto [tag_offsets, tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(tag_blocks));
  auto [face_offsets, face_data] =
      csg_graph_detail::blocks_to_arrays(std::move(face_blocks));
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids)),
          std::move(tag_offsets),
          std::move(tag_data),
          std::move(face_offsets),
          std::move(face_data)};
}

template <typename Index, typename Real>
auto make_csg_domains_with_index_map(const csg_graph<Index, Real> &graph,
                                     tf::domain_config config)
    -> csg_domains_index_map_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids, map] =
      tf::make_csg_domains(data.graph, config, tf::return_index_map);
  auto [face_tag_offsets, face_tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.face_tag_blocks));
  auto [face_offsets, face_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.face_blocks));
  auto [point_tag_offsets, point_tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.point_tag_blocks));
  auto [point_offsets, point_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.point_blocks));
  auto inclusion = csg_graph_detail::inclusion_to_array(map.inclusion);
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids)),
          std::move(face_tag_offsets),
          std::move(face_tag_data),
          std::move(face_offsets),
          std::move(face_data),
          std::move(point_tag_offsets),
          std::move(point_tag_data),
          std::move(point_offsets),
          std::move(point_data),
          static_cast<Index>(map.n_original_points),
          static_cast<Index>(map.n_tags),
          static_cast<Index>(map.n_output_points),
          std::move(inclusion)};
}

template <typename Index, typename Real>
auto make_csg_domains_with_index_map(const csg_graph<Index, Real> &graph,
                                     const tf::csg::selection_t &selection,
                                     tf::domain_config config)
    -> csg_domains_index_map_result<Index, Real> {
  const auto &data = csg_graph_detail::access::of(graph);
  csg_graph_detail::require_selection(selection, data.meshes.size());
  csg_graph_detail::require_boundary_selection(selection);
  csg_graph_detail::require_domain_config(config);
  auto [cells, ids, map] =
      tf::make_csg_domains(data.graph, selection, config, tf::return_index_map);
  auto [face_tag_offsets, face_tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.face_tag_blocks));
  auto [face_offsets, face_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.face_blocks));
  auto [point_tag_offsets, point_tag_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.point_tag_blocks));
  auto [point_offsets, point_data] =
      csg_graph_detail::blocks_to_arrays(std::move(map.point_blocks));
  auto inclusion = csg_graph_detail::inclusion_to_array(map.inclusion);
  return {csg_graph_detail::cells_to_meshes<Index, Real>(cells),
          nd_array<Index>::from_buffer(std::move(ids)),
          std::move(face_tag_offsets),
          std::move(face_tag_data),
          std::move(face_offsets),
          std::move(face_data),
          std::move(point_tag_offsets),
          std::move(point_tag_data),
          std::move(point_offsets),
          std::move(point_data),
          static_cast<Index>(map.n_original_points),
          static_cast<Index>(map.n_tags),
          static_cast<Index>(map.n_output_points),
          std::move(inclusion)};
}

} // namespace tf::cpp
