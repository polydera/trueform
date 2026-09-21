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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/csg/expression.hpp"
#include "trueform/csg/expression/selection.hpp"
#include "trueform/topology/domain_config.hpp"

namespace tf::cpp {

/// @brief One watertight mesh per kept domain. A boundary selection only:
///        an inside read is refused.
template <typename Index, typename Real>
auto make_csg_domains(const csg_graph<Index, Real> &graph,
                      tf::domain_config config = tf::domain_config::none)
    -> csg_domains_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_domains(const csg_graph<Index, Real> &graph,
                      const tf::csg::selection_t &selection,
                      tf::domain_config config = tf::domain_config::none)
    -> csg_domains_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_domains_with_labels(
    const csg_graph<Index, Real> &graph,
    tf::domain_config config = tf::domain_config::none)
    -> csg_domains_labeled_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_domains_with_labels(
    const csg_graph<Index, Real> &graph, const tf::csg::selection_t &selection,
    tf::domain_config config = tf::domain_config::none)
    -> csg_domains_labeled_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_domains_with_index_map(
    const csg_graph<Index, Real> &graph,
    tf::domain_config config = tf::domain_config::none)
    -> csg_domains_index_map_result<Index, Real>;
template <typename Index, typename Real>
auto make_csg_domains_with_index_map(
    const csg_graph<Index, Real> &graph, const tf::csg::selection_t &selection,
    tf::domain_config config = tf::domain_config::none)
    -> csg_domains_index_map_result<Index, Real>;

#define TF_CPP_EXTERN_CSG_DOMAINS(Index, Real)                                 \
  extern template auto make_csg_domains(const csg_graph<Index, Real> &,        \
                                        tf::domain_config)                     \
      -> csg_domains_result<Index, Real>;                                      \
  extern template auto make_csg_domains(                                       \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_result<Index, Real>;                   \
  extern template auto make_csg_domains_with_labels(                           \
      const csg_graph<Index, Real> &, tf::domain_config)                       \
      -> csg_domains_labeled_result<Index, Real>;                              \
  extern template auto make_csg_domains_with_labels(                           \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_labeled_result<Index, Real>;           \
  extern template auto make_csg_domains_with_index_map(                        \
      const csg_graph<Index, Real> &, tf::domain_config)                       \
      -> csg_domains_index_map_result<Index, Real>;                            \
  extern template auto make_csg_domains_with_index_map(                        \
      const csg_graph<Index, Real> &, const tf::csg::selection_t &,            \
      tf::domain_config) -> csg_domains_index_map_result<Index, Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_DOMAINS)

#undef TF_CPP_EXTERN_CSG_DOMAINS

} // namespace tf::cpp
