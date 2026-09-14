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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/csg/expression.hpp"

namespace tf::cpp {

/// @brief The mesh an expression names, whole or with its source labels.
template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph)
    -> tf::polygons_buffer<Index, Real, 3, 3>;
template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph,
                   const tf::csg::expr &expression)
    -> tf::polygons_buffer<Index, Real, 3, 3>;
template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph)
    -> csg_mesh_labeled_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph,
                               const tf::csg::expr &expression)
    -> csg_mesh_labeled_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_mesh_with_index_map(const csg_graph<Index, Real> &graph,
                                  const tf::csg::expr &expression)
    -> csg_mesh_index_map_result<Index, Real>;

#define TF_CPP_EXTERN_CSG_MESH(Index, Real)                                    \
  extern template auto make_csg_mesh(const csg_graph<Index, Real> &)           \
      -> tf::polygons_buffer<Index, Real, 3, 3>;                               \
  extern template auto make_csg_mesh(const csg_graph<Index, Real> &,           \
                                     const tf::csg::expr &)                    \
      -> tf::polygons_buffer<Index, Real, 3, 3>;                               \
  extern template auto make_csg_mesh_with_labels(                              \
      const csg_graph<Index, Real> &) -> csg_mesh_labeled_result<Index, Real>; \
  extern template auto make_csg_mesh_with_labels(                              \
      const csg_graph<Index, Real> &, const tf::csg::expr &)                   \
      -> csg_mesh_labeled_result<Index, Real>;                                 \
  extern template auto make_csg_mesh_with_index_map(                           \
      const csg_graph<Index, Real> &, const tf::csg::expr &)                   \
      -> csg_mesh_index_map_result<Index, Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_MESH)

#undef TF_CPP_EXTERN_CSG_MESH

} // namespace tf::cpp
