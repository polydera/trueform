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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/aabb.hpp>
#include <trueform/core/line_like.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/plane.hpp>
#include <trueform/core/point_view.hpp>
#include <trueform/core/polygon.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/ray_like.hpp>
#include <trueform/core/segment.hpp>
#include <trueform/core/unit_vector_view.hpp>
#include <trueform/core/unsafe.hpp>
#include <trueform/core/vector_view.hpp>

namespace tf::py {

// Helper to create views from primitive data
template <std::size_t Dims, typename RealT>
auto make_point_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<Dims>>
        data) {
  return tf::make_point_view<Dims>(data.data());
}

template <std::size_t Dims, typename RealT>
auto make_segment_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<2, Dims>>
        data) {
  auto pts = tf::make_points<Dims>(tf::make_range(data.data(), 2 * Dims));
  return tf::make_segment(pts);
}

template <std::size_t Dims, typename RealT>
auto make_polygon_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        data) {
  std::size_t num_vertices = data.shape(0);
  auto pts =
      tf::make_points<Dims>(tf::make_range(data.data(), num_vertices * Dims));
  return tf::make_polygon(pts);
}

template <std::size_t V, std::size_t Dims, typename RealT>
auto make_polygon_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<V, Dims>>
        data) {
  std::size_t num_vertices = data.shape(0);
  auto pts =
      tf::make_points<Dims>(tf::make_range(data.data(), num_vertices * Dims));
  return tf::make_polygon<V>(pts);
}

template <std::size_t Dims, typename RealT>
auto make_polygon_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT> data) {
  std::size_t num_vertices = data.shape(0);
  auto pts =
      tf::make_points<Dims>(tf::make_range(data.data(), num_vertices * Dims));
  return tf::make_polygon(pts);
}

template <std::size_t Dims, typename RealT>
auto make_ray_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<2, Dims>>
        data) {
  auto origin = tf::make_point_view<Dims>(data.data());
  auto direction = tf::make_vector_view<Dims>(data.data() + Dims);
  return tf::make_ray_like(origin, direction);
}

template <std::size_t Dims, typename RealT>
auto make_line_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<2, Dims>>
        data) {
  auto origin = tf::make_point_view<Dims>(data.data());
  auto direction = tf::make_vector_view<Dims>(data.data() + Dims);
  return tf::make_line_like(origin, direction);
}

template <std::size_t Dims, typename RealT>
auto make_aabb_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<2, Dims>>
        data) {
  auto min = tf::make_point_view<Dims>(data.data());
  auto max = tf::make_point_view<Dims>(data.data() + Dims);
  return tf::make_aabb_like(min, max);
}

template <std::size_t Dims, typename RealT>
auto make_plane_from_array(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<Dims + 1>>
        data) {
  auto normal = tf::make_unit_vector_view(
      tf::unsafe, tf::make_vector_view<Dims>(data.data()));
  auto offset = data.data()[Dims];
  return tf::make_plane(normal, offset);
}
} // namespace tf::py
