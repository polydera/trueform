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

#include "trueform/python/topology/split_non_manifold_vertices.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace tf::py {

auto register_topology_split_non_manifold_vertices(nanobind::module_ &m)
    -> void {

  // ========== Fixed triangles (ngon=3) ==========

  m.def("split_non_manifold_vertices_int3float2d",
        &split_non_manifold_vertices<int, float, 3, 2>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int3float3d",
        &split_non_manifold_vertices<int, float, 3, 3>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int3double2d",
        &split_non_manifold_vertices<int, double, 3, 2>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int3double3d",
        &split_non_manifold_vertices<int, double, 3, 3>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int643float2d",
        &split_non_manifold_vertices<int64_t, float, 3, 2>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int643float3d",
        &split_non_manifold_vertices<int64_t, float, 3, 3>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int643double2d",
        &split_non_manifold_vertices<int64_t, double, 3, 2>, nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int643double3d",
        &split_non_manifold_vertices<int64_t, double, 3, 3>, nb::arg("mesh"));

  // ========== Dynamic mesh (variable n-gons) ==========

  m.def("split_non_manifold_vertices_intdynfloat2d",
        &split_non_manifold_vertices<int, float, tf::dynamic_size, 2>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_intdynfloat3d",
        &split_non_manifold_vertices<int, float, tf::dynamic_size, 3>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_intdyndouble2d",
        &split_non_manifold_vertices<int, double, tf::dynamic_size, 2>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_intdyndouble3d",
        &split_non_manifold_vertices<int, double, tf::dynamic_size, 3>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int64dynfloat2d",
        &split_non_manifold_vertices<int64_t, float, tf::dynamic_size, 2>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int64dynfloat3d",
        &split_non_manifold_vertices<int64_t, float, tf::dynamic_size, 3>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int64dyndouble2d",
        &split_non_manifold_vertices<int64_t, double, tf::dynamic_size, 2>,
        nb::arg("mesh"));
  m.def("split_non_manifold_vertices_int64dyndouble3d",
        &split_non_manifold_vertices<int64_t, double, tf::dynamic_size, 3>,
        nb::arg("mesh"));
}

} // namespace tf::py
