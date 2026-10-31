/*
 * Copyright (c) 2026 XLAB
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

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/none.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/geometry/dirichlet_order.hpp>
#include <trueform/geometry/make_dirichlet_field.hpp>
#include <trueform/geometry/make_geodesic_distances.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/vertex_link.hpp>
#include <utility>

namespace tf::py {

/// The ids, values and weights arrive validated and contiguous from the
/// facade: ids in range (and distinct for the Dirichlet field), values
/// finite, three weights per face.
template <typename Index, typename RealT, std::size_t Ngon>
auto geodesic_distances_impl(
    mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> seeds,
    double radius) {
  auto primitives = mesh.make_primitive_range();
  auto polygons =
      tf::make_polygons(primitives.faces(), primitives.points()) |
      tf::tag(mesh.face_membership()) | tf::tag(mesh.vertex_link());
  return make_numpy_array(tf::make_geodesic_distances(
      polygons, tf::make_range(seeds.data(), seeds.shape(0)), RealT(radius)));
}

/// The field and the status its system answered, over the corner weights a
/// range states or the cotangent weights `tf::none` states. `order` arrives
/// as a validated @ref tf::dirichlet_order value.
template <typename Index, typename RealT, typename Weights>
auto dirichlet_field_impl(
    mesh_wrapper<Index, RealT, 3, 3> &mesh, const Weights &corner_weights,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        fixed_ids,
    nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
        fixed_values,
    int order) {
  auto primitives = mesh.make_primitive_range();
  auto result = tf::make_dirichlet_field(
      tf::make_polygons(primitives.faces(), primitives.points()) |
          tf::tag(mesh.face_membership()),
      corner_weights, tf::make_range(fixed_ids.data(), fixed_ids.shape(0)),
      tf::make_range(fixed_values.data(), fixed_values.shape(0)),
      static_cast<tf::dirichlet_order>(order));
  return nanobind::make_tuple(make_numpy_array(std::move(result.field)),
                              static_cast<int>(result.status));
}

} // namespace tf::py
