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
#include <trueform/python/geometry/chamfer_error.hpp>

namespace tf::py {

auto register_chamfer_error(nanobind::module_ &m) -> void {

  // float, 2D
  m.def("chamfer_error_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1) {
          return chamfer_error_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Compute one-way Chamfer error from cloud0 to cloud1.\n"
        "Returns mean nearest-neighbor distance.");

  // float, 3D
  m.def("chamfer_error_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1) {
          return chamfer_error_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Compute one-way Chamfer error from cloud0 to cloud1.\n"
        "Returns mean nearest-neighbor distance.");

  // double, 2D
  m.def("chamfer_error_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1) {
          return chamfer_error_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Compute one-way Chamfer error from cloud0 to cloud1.\n"
        "Returns mean nearest-neighbor distance.");

  // double, 3D
  m.def("chamfer_error_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1) {
          return chamfer_error_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Compute one-way Chamfer error from cloud0 to cloud1.\n"
        "Returns mean nearest-neighbor distance.");
}

} // namespace tf::py
