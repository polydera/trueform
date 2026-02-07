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
#include <trueform/python/geometry/fit_rigid_alignment.hpp>

namespace tf::py {

auto register_fit_rigid_alignment(nanobind::module_ &m) -> void {

  // ============================================================
  // Point-to-point alignment
  // ============================================================

  // float, 2D
  m.def("fit_rigid_alignment_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1) {
          return fit_rigid_alignment_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Fit rigid transformation from cloud0 to cloud1 (Kabsch/Procrustes).\n"
        "Returns a 3x3 transformation matrix.");

  // float, 3D
  m.def("fit_rigid_alignment_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1) {
          return fit_rigid_alignment_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Fit rigid transformation from cloud0 to cloud1 (Kabsch/Procrustes).\n"
        "Returns a 4x4 transformation matrix.");

  // double, 2D
  m.def("fit_rigid_alignment_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1) {
          return fit_rigid_alignment_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Fit rigid transformation from cloud0 to cloud1 (Kabsch/Procrustes).\n"
        "Returns a 3x3 transformation matrix.");

  // double, 3D
  m.def("fit_rigid_alignment_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1) {
          return fit_rigid_alignment_impl(cloud0, cloud1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        "Fit rigid transformation from cloud0 to cloud1 (Kabsch/Procrustes).\n"
        "Returns a 4x4 transformation matrix.");

  // ============================================================
  // Point-to-plane alignment (target has normals)
  // ============================================================

  // float, 3D
  m.def("fit_rigid_alignment_p2plane_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals1) {
          return fit_rigid_alignment_p2plane_impl(cloud0, cloud1, normals1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("normals1"),
        "Fit rigid transformation using point-to-plane distance.\n"
        "Returns a 4x4 transformation matrix.");

  // double, 3D
  m.def("fit_rigid_alignment_p2plane_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals1) {
          return fit_rigid_alignment_p2plane_impl(cloud0, cloud1, normals1);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("normals1"),
        "Fit rigid transformation using point-to-plane distance.\n"
        "Returns a 4x4 transformation matrix.");

  // ============================================================
  // Normal weighting alignment (both have normals)
  // ============================================================

  // float, 3D
  m.def("fit_rigid_alignment_weighted_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals0,
           point_cloud_wrapper<float, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals1) {
          return fit_rigid_alignment_weighted_impl(cloud0, normals0, cloud1,
                                                   normals1);
        },
        nanobind::arg("cloud0"), nanobind::arg("normals0"),
        nanobind::arg("cloud1"), nanobind::arg("normals1"),
        "Fit rigid transformation with normal weighting.\n"
        "Returns a 4x4 transformation matrix.");

  // double, 3D
  m.def("fit_rigid_alignment_weighted_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals0,
           point_cloud_wrapper<double, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals1) {
          return fit_rigid_alignment_weighted_impl(cloud0, normals0, cloud1,
                                                   normals1);
        },
        nanobind::arg("cloud0"), nanobind::arg("normals0"),
        nanobind::arg("cloud1"), nanobind::arg("normals1"),
        "Fit rigid transformation with normal weighting.\n"
        "Returns a 4x4 transformation matrix.");
}

} // namespace tf::py
