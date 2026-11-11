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

#include "trueform/python/tetmesh.hpp"
#include "trueform/python/tetmesh/tet_mesh.hpp"
#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

namespace tf::py {

auto register_tet_mesh_float(nanobind::module_ &m) -> void {
  using namespace nanobind;
  // 2 index types, triangle meshes, 3D
  m.def(
      "tet_mesh_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh, bool split_constraints, double tolerance,
         std::optional<std::size_t> site_budget_ratio, bool return_report,
         bool return_index_map) {
        return tet_mesh_impl<int, float>(mesh, split_constraints, tolerance,
                                          site_budget_ratio, return_report,
                                          return_index_map);
      },
      arg("mesh"), arg("split_constraints"), arg("tolerance"),
      arg("site_budget_ratio").none(), arg("return_report"),
      arg("return_index_map"));
  m.def(
      "tet_mesh_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh, bool split_constraints, double tolerance,
         std::optional<std::size_t> site_budget_ratio, bool return_report,
         bool return_index_map) {
        return tet_mesh_impl<int64_t, float>(mesh, split_constraints, tolerance,
                                          site_budget_ratio, return_report,
                                          return_index_map);
      },
      arg("mesh"), arg("split_constraints"), arg("tolerance"),
      arg("site_budget_ratio").none(), arg("return_report"),
      arg("return_index_map"));
}

} // namespace tf::py
