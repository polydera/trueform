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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <optional>
#include <trueform/core/edges.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/make_cdt.hpp>

namespace tf::py {

template <typename Coord>
auto make_cdt_impl(
    nanobind::ndarray<nanobind::numpy, const Coord, nanobind::shape<-1, 2>>
        points) {
  auto pts_range =
      tf::make_points<2>(tf::make_range(points.data(), points.shape(0) * 2));
  return make_numpy_array(tf::make_cdt(pts_range));
}

template <typename Coord>
auto make_cdt_with_maps_impl(
    nanobind::ndarray<nanobind::numpy, const Coord, nanobind::shape<-1, 2>>
        points) {
  auto pts_range =
      tf::make_points<2>(tf::make_range(points.data(), points.shape(0) * 2));
  auto [polys, im] = tf::make_cdt(pts_range, tf::return_index_map);
  return nanobind::make_tuple(make_numpy_array(std::move(polys)),
                              make_numpy_array(std::move(im)));
}

template <typename Coord>
auto make_cdt_edges_impl(
    nanobind::ndarray<nanobind::numpy, const Coord, nanobind::shape<-1, 2>>
        points,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1, 2>>
        edges,
    std::optional<nanobind::ndarray<nanobind::numpy, const bool,
                                    nanobind::shape<-1>>>
        edge_mask) {
  auto pts_range =
      tf::make_points<2>(tf::make_range(points.data(), points.shape(0) * 2));
  auto edges_range = tf::make_edges(tf::make_blocked_range<2>(
      tf::make_range(edges.data(), edges.shape(0) * 2)));

  if (edge_mask) {
    auto mask_range = tf::make_range(edge_mask->data(), edges.shape(0));
    return make_numpy_array(tf::make_cdt(pts_range, edges_range, mask_range));
  }
  return make_numpy_array(tf::make_cdt(pts_range, edges_range));
}

template <typename Coord>
auto make_cdt_edges_with_maps_impl(
    nanobind::ndarray<nanobind::numpy, const Coord, nanobind::shape<-1, 2>>
        points,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1, 2>>
        edges,
    std::optional<nanobind::ndarray<nanobind::numpy, const bool,
                                    nanobind::shape<-1>>>
        edge_mask) {
  auto pts_range =
      tf::make_points<2>(tf::make_range(points.data(), points.shape(0) * 2));
  auto edges_range = tf::make_edges(tf::make_blocked_range<2>(
      tf::make_range(edges.data(), edges.shape(0) * 2)));

  if (edge_mask) {
    auto mask_range = tf::make_range(edge_mask->data(), edges.shape(0));
    auto [polys, im] = tf::make_cdt(pts_range, edges_range, mask_range,
                                    tf::return_index_map);
    return nanobind::make_tuple(make_numpy_array(std::move(polys)),
                                make_numpy_array(std::move(im)));
  }
  auto [polys, im] =
      tf::make_cdt(pts_range, edges_range, tf::return_index_map);
  return nanobind::make_tuple(make_numpy_array(std::move(polys)),
                              make_numpy_array(std::move(im)));
}

} // namespace tf::py
