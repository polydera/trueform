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
#include "trueform/python/geometry/surface_fields.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>

namespace tf::py {
namespace {

template <typename Index, typename RealT, std::size_t Ngon>
auto register_geodesic_distances_of(nanobind::module_ &m,
                                    const std::string &suffix) -> void {
  using namespace nanobind;
  m.def(("geodesic_distances_" + suffix).c_str(),
        &geodesic_distances_impl<Index, RealT, Ngon>, arg("mesh"), arg("seeds"),
        arg("radius"));
}

template <typename Index, typename RealT>
auto register_dirichlet_field_of(nanobind::module_ &m,
                                 const std::string &suffix) -> void {
  using namespace nanobind;
  const auto name = "dirichlet_field_" + suffix;
  m.def(
      name.c_str(),
      [](mesh_wrapper<Index, RealT, 3, 3> &mesh,
         ndarray<numpy, const Index, shape<-1>> fixed_ids,
         ndarray<numpy, const double, shape<-1>> fixed_values, int order) {
        return dirichlet_field_impl<Index, RealT>(mesh, tf::none, fixed_ids,
                                                  fixed_values, order);
      },
      arg("mesh"), arg("fixed_ids"), arg("fixed_values"), arg("order"));
  m.def(
      name.c_str(),
      [](mesh_wrapper<Index, RealT, 3, 3> &mesh,
         ndarray<numpy, const double, shape<-1>> corner_weights,
         ndarray<numpy, const Index, shape<-1>> fixed_ids,
         ndarray<numpy, const double, shape<-1>> fixed_values, int order) {
        return dirichlet_field_impl<Index, RealT>(
            mesh,
            tf::make_range(corner_weights.data(), corner_weights.shape(0)),
            fixed_ids, fixed_values, order);
      },
      arg("mesh"), arg("corner_weights"), arg("fixed_ids"),
      arg("fixed_values"), arg("order"));
}

} // namespace

auto register_surface_fields(nanobind::module_ &m) -> void {
  // 2 index x 2 real x 2 arities for the distances, triangles for the
  // field; 3D meshes
  register_geodesic_distances_of<int, float, 3>(m, "int3float3d");
  register_geodesic_distances_of<int, double, 3>(m, "int3double3d");
  register_geodesic_distances_of<int64_t, float, 3>(m, "int643float3d");
  register_geodesic_distances_of<int64_t, double, 3>(m, "int643double3d");
  register_geodesic_distances_of<int, float, dynamic_size>(m, "intdynfloat3d");
  register_geodesic_distances_of<int, double, dynamic_size>(m,
                                                            "intdyndouble3d");
  register_geodesic_distances_of<int64_t, float, dynamic_size>(
      m, "int64dynfloat3d");
  register_geodesic_distances_of<int64_t, double, dynamic_size>(
      m, "int64dyndouble3d");

  register_dirichlet_field_of<int, float>(m, "int3float3d");
  register_dirichlet_field_of<int, double>(m, "int3double3d");
  register_dirichlet_field_of<int64_t, float>(m, "int643float3d");
  register_dirichlet_field_of<int64_t, double>(m, "int643double3d");
}

} // namespace tf::py
