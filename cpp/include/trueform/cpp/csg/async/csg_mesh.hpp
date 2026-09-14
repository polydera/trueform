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
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/cpp/csg/csg_mesh.hpp"
#include "trueform/csg/expression.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real>
auto make_csg_mesh(Resolver &&resolver, const csg_graph<Index, Real> &graph)
    -> resolver_result_t<Resolver, tf::polygons_buffer<Index, Real, 3, 3>> {
  auto owned_graph = graph;
  return submit<tf::polygons_buffer<Index, Real, 3, 3>>(
      std::forward<Resolver>(resolver),
      [graph = std::move(owned_graph)] { return cpp::make_csg_mesh(graph); });
}

template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph)
    -> std::future<tf::polygons_buffer<Index, Real, 3, 3>> {
  return async::make_csg_mesh(future_resolver{}, graph);
}

template <typename Resolver, typename Index, typename Real>
auto make_csg_mesh(Resolver &&resolver, const csg_graph<Index, Real> &graph,
                   const tf::csg::expr &expression)
    -> resolver_result_t<Resolver, tf::polygons_buffer<Index, Real, 3, 3>> {
  auto owned_graph = graph;
  auto owned_expression = expression;
  return submit<tf::polygons_buffer<Index, Real, 3, 3>>(
      std::forward<Resolver>(resolver),
      [graph = std::move(owned_graph),
       expression = std::move(owned_expression)] {
        return cpp::make_csg_mesh(graph, expression);
      });
}

template <typename Index, typename Real>
auto make_csg_mesh(const csg_graph<Index, Real> &graph,
                   const tf::csg::expr &expression)
    -> std::future<tf::polygons_buffer<Index, Real, 3, 3>> {
  return async::make_csg_mesh(future_resolver{}, graph, expression);
}

template <typename Resolver, typename Index, typename Real>
auto make_csg_mesh_with_labels(Resolver &&resolver,
                               const csg_graph<Index, Real> &graph)
    -> resolver_result_t<Resolver, csg_mesh_labeled_result<Index, Real>> {
  auto owned_graph = graph;
  return submit<csg_mesh_labeled_result<Index, Real>>(
      std::forward<Resolver>(resolver), [graph = std::move(owned_graph)] {
        return cpp::make_csg_mesh_with_labels(graph);
      });
}

template <typename Resolver, typename Index, typename Real>
auto make_csg_mesh_with_labels(Resolver &&resolver,
                               const csg_graph<Index, Real> &graph,
                               const tf::csg::expr &expression)
    -> resolver_result_t<Resolver, csg_mesh_labeled_result<Index, Real>> {
  auto owned_graph = graph;
  auto owned_expression = expression;
  return submit<csg_mesh_labeled_result<Index, Real>>(
      std::forward<Resolver>(resolver),
      [graph = std::move(owned_graph),
       expression = std::move(owned_expression)] {
        return cpp::make_csg_mesh_with_labels(graph, expression);
      });
}

template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph)
    -> std::future<csg_mesh_labeled_result<Index, Real>> {
  return async::make_csg_mesh_with_labels(future_resolver{}, graph);
}

template <typename Index, typename Real>
auto make_csg_mesh_with_labels(const csg_graph<Index, Real> &graph,
                               const tf::csg::expr &expression)
    -> std::future<csg_mesh_labeled_result<Index, Real>> {
  return async::make_csg_mesh_with_labels(future_resolver{}, graph, expression);
}

template <typename Resolver, typename Index, typename Real>
auto make_csg_mesh_with_index_map(Resolver &&resolver,
                                  const csg_graph<Index, Real> &graph,
                                  const tf::csg::expr &expression)
    -> resolver_result_t<Resolver, csg_mesh_index_map_result<Index, Real>> {
  auto owned_graph = graph;
  auto owned_expression = expression;
  return submit<csg_mesh_index_map_result<Index, Real>>(
      std::forward<Resolver>(resolver),
      [graph = std::move(owned_graph),
       expression = std::move(owned_expression)] {
        return cpp::make_csg_mesh_with_index_map(graph, expression);
      });
}

template <typename Index, typename Real>
auto make_csg_mesh_with_index_map(const csg_graph<Index, Real> &graph,
                                  const tf::csg::expr &expression)
    -> std::future<csg_mesh_index_map_result<Index, Real>> {
  return async::make_csg_mesh_with_index_map(future_resolver{}, graph,
                                             expression);
}

} // namespace tf::cpp::async
