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

#include <cstdint>
#include <vector>

#define TF_CPP_INSTANTIATE_CSG(Index, Real)                                    \
  template class csg_graph<Index, Real>;                                       \
  template auto make_csg_graph<Index, Real>(                                   \
      std::vector<mesh<Index, Real, 3, 3>>, std::vector<std::int32_t>,         \
      tf::arrangement_config)                                                  \
      ->csg_graph<Index, Real>;                                                \
  template auto csg_created_points(const csg_graph<Index, Real> &)             \
      -> nd_array<Real>;                                                       \
  template auto csg_intersection_curves(const csg_graph<Index, Real> &)        \
      -> tf::curves_buffer<Index, Real, 3>;                                    \
  template auto make_csg_mesh(const csg_graph<Index, Real> &)                  \
      -> tf::polygons_buffer<Index, Real, 3, 3>;                               \
  template auto make_csg_mesh(const csg_graph<Index, Real> &,                  \
                              const tf::csg::selection_t &)                    \
      -> tf::polygons_buffer<Index, Real, 3, 3>;                               \
  template auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &)      \
      -> csg_mesh_labeled_result<Index, Real>;                                 \
  template auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &,      \
                                          const tf::csg::selection_t &)        \
      -> csg_mesh_labeled_result<Index, Real>;                                 \
  template auto make_csg_mesh_with_index_map(const csg_graph<Index, Real> &,   \
                                             const tf::csg::selection_t &)     \
      -> csg_mesh_index_map_result<Index, Real>;                               \
  template auto make_csg_domains(const csg_graph<Index, Real> &,               \
                                 tf::domain_config)                            \
      -> csg_domains_result<Index, Real>;                                      \
  template auto make_csg_domains(                                              \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_result<Index, Real>;                   \
  template auto make_csg_domains_with_labels(const csg_graph<Index, Real> &,   \
                                             tf::domain_config)                \
      -> csg_domains_labeled_result<Index, Real>;                              \
  template auto make_csg_domains_with_labels(                                  \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_labeled_result<Index, Real>;           \
  template auto make_csg_domains_with_index_map(                               \
      const csg_graph<Index, Real> &, tf::domain_config)                       \
      -> csg_domains_index_map_result<Index, Real>;                            \
  template auto make_csg_domains_with_index_map(                               \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_index_map_result<Index, Real>
