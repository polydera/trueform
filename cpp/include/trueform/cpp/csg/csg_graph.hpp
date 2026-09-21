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

#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/core/curves_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace tf::cpp {

template <typename Index, typename Real> struct csg_mesh_labeled_result {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  nd_array<Index> tag_labels;
  nd_array<Index> face_labels;
};

template <typename Index, typename Real> struct csg_mesh_index_map_result {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  nd_array<Index> point_tag_labels;
  nd_array<Index> point_labels;
  nd_array<Index> face_tag_labels;
  nd_array<Index> face_labels;
  nd_array<Index> point_f_offsets;
  nd_array<Index> point_f_data;
  /// (n_tags, 2): each tag's [begin, end) span of uncut output faces
  nd_array<Index> uncut_faces;
  Index number_of_original_points = 0;
  Index number_of_tags = 0;
  Index number_of_output_points = 0;
};

template <typename Index, typename Real> struct csg_domains_result {
  std::vector<tf::polygons_buffer<Index, Real, 3, 3>> meshes;
  nd_array<Index> ids;
};

template <typename Index, typename Real> struct csg_domains_labeled_result {
  std::vector<tf::polygons_buffer<Index, Real, 3, 3>> meshes;
  nd_array<Index> ids;
  nd_array<Index> tag_offsets;
  nd_array<Index> tag_data;
  nd_array<Index> face_offsets;
  nd_array<Index> face_data;
};

template <typename Index, typename Real> struct csg_domains_index_map_result {
  std::vector<tf::polygons_buffer<Index, Real, 3, 3>> meshes;
  nd_array<Index> ids;
  nd_array<Index> face_tag_offsets;
  nd_array<Index> face_tag_data;
  nd_array<Index> face_offsets;
  nd_array<Index> face_data;
  nd_array<Index> point_tag_offsets;
  nd_array<Index> point_tag_data;
  nd_array<Index> point_offsets;
  nd_array<Index> point_data;
  Index number_of_original_points = 0;
  Index number_of_tags = 0;
  Index number_of_output_points = 0;
  nd_array<std::int8_t> inclusion;
};

namespace csg_graph_detail {
/// What a read reaches the graph through, so the graph itself is the
/// arrangement and its handle and nothing more, and every read of it is a free
/// function -- one public spelling per operation.
struct access;
} // namespace csg_graph_detail

/// @brief Copyable handle to one logically immutable, reusable CSG arrangement.
///
/// The graph reads its operands for as long as it lives, so it holds their
/// meshes: the geometry and the cache each names outlive the graph, which is
/// the borrow law every view in trueform obeys, and a caller whose own handle
/// cannot outlive it assembles with the keepalive the mesh constructor takes.
/// Inputs are fixed-connectivity triangle meshes in 3D.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real> class csg_graph {
  struct data;
  std::shared_ptr<data> _data;

  friend struct csg_graph_detail::access;

public:
  csg_graph();

  static auto create(std::vector<mesh<Index, Real, 3, 3>> meshes,
                     std::vector<std::int32_t> sheets = {},
                     tf::arrangement_config config = {}) -> csg_graph;

  auto is_valid() const -> bool;
  auto destroy() -> void;
};

/// @brief Build and classify one reusable arrangement of at least two meshes.
///
/// A csg arrangement is triangles, so the operands are read at arity three.
template <typename Index, typename Real>
auto make_csg_graph(std::vector<mesh<Index, Real, 3, 3>> meshes,
                    std::vector<std::int32_t> sheets = {},
                    tf::arrangement_config config = {})
    -> csg_graph<Index, Real>;

#define TF_CPP_EXTERN_CSG_GRAPH(Index, Real)                                   \
  extern template class csg_graph<Index, Real>;                                \
  extern template auto make_csg_graph<Index, Real>(                            \
      std::vector<mesh<Index, Real, 3, 3>>, std::vector<std::int32_t>,         \
      tf::arrangement_config) -> csg_graph<Index, Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_GRAPH)

#undef TF_CPP_EXTERN_CSG_GRAPH

} // namespace tf::cpp
