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
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/form_intersects_form.hpp>

namespace tf::py {

auto register_point_cloud_intersects_point_cloud(nanobind::module_ &m)
    -> void {

  // ============================================================================
  // PointCloud intersects PointCloud
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 = 4 functions
  // ============================================================================

  // float, 2D
  m.def("intersects_point_cloud_point_cloud_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1) {
          return form_intersects_form(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"));

  // float, 3D
  m.def("intersects_point_cloud_point_cloud_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1) {
          return form_intersects_form(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"));

  // double, 2D
  m.def("intersects_point_cloud_point_cloud_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1) {
          return form_intersects_form(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"));

  // double, 3D
  m.def("intersects_point_cloud_point_cloud_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1) {
          return form_intersects_form(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"));
}

} // namespace tf::py
