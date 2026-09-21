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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/csg/csg_domains.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/csg/expression.hpp"
#include "trueform/csg/expression/selection.hpp"
#include "trueform/topology/domain_config.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_CSG_ASYNC_DOMAINS(Name, Result)                                 \
  template <typename Resolver, typename Index, typename Real>                  \
  auto Name(Resolver &&resolver, const csg_graph<Index, Real> &graph,          \
            tf::domain_config config = tf::domain_config::none)                \
      -> resolver_result_t<Resolver, Result<Index, Real>> {                    \
    auto owned_graph = graph;                                                  \
    return submit<Result<Index, Real>>(                                        \
        std::forward<Resolver>(resolver),                                      \
        [graph = std::move(owned_graph), config] {                             \
          return cpp::Name(graph, config);                                     \
        });                                                                    \
  }                                                                            \
  template <typename Resolver, typename Index, typename Real>                  \
  auto Name(Resolver &&resolver, const csg_graph<Index, Real> &graph,          \
            const tf::csg::selection_t &selection,                             \
            tf::domain_config config = tf::domain_config::none)                \
      -> resolver_result_t<Resolver, Result<Index, Real>> {                    \
    auto owned_graph = graph;                                                  \
    auto owned_selection = selection;                                          \
    return submit<Result<Index, Real>>(                                        \
        std::forward<Resolver>(resolver),                                      \
        [graph = std::move(owned_graph),                                       \
         selection = std::move(owned_selection),                               \
         config] { return cpp::Name(graph, selection, config); });             \
  }                                                                            \
  template <typename Index, typename Real>                                     \
  auto Name(const csg_graph<Index, Real> &graph,                               \
            tf::domain_config config = tf::domain_config::none)                \
      -> std::future<Result<Index, Real>> {                                    \
    return async::Name(future_resolver{}, graph, config);                      \
  }                                                                            \
  template <typename Index, typename Real>                                     \
  auto Name(const csg_graph<Index, Real> &graph,                               \
            const tf::csg::selection_t &selection,                             \
            tf::domain_config config = tf::domain_config::none)                \
      -> std::future<Result<Index, Real>> {                                    \
    return async::Name(future_resolver{}, graph, selection, config);           \
  }

TF_CPP_CSG_ASYNC_DOMAINS(make_csg_domains, csg_domains_result)
TF_CPP_CSG_ASYNC_DOMAINS(make_csg_domains_with_labels,
                         csg_domains_labeled_result)
TF_CPP_CSG_ASYNC_DOMAINS(make_csg_domains_with_index_map,
                         csg_domains_index_map_result)

#undef TF_CPP_CSG_ASYNC_DOMAINS

} // namespace tf::cpp::async
