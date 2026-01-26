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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_point_cloud_gather_ids_point_cloud(nanobind::module_ &m)
    -> void {

  // ============================================================================
  // PointCloud gather_ids PointCloud
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 = 4 functions
  // ============================================================================

  // float, 2D
  m.def("gather_ids_point_cloud_point_cloud_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(cloud0, cloud1, predicate_type,
                                                 threshold);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // float, 3D
  m.def("gather_ids_point_cloud_point_cloud_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(cloud0, cloud1, predicate_type,
                                                 threshold);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // double, 2D
  m.def("gather_ids_point_cloud_point_cloud_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1,
           const std::string &predicate_type,
           std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(cloud0, cloud1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // double, 3D
  m.def("gather_ids_point_cloud_point_cloud_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1,
           const std::string &predicate_type,
           std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(cloud0, cloud1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());
}

} // namespace tf::py
