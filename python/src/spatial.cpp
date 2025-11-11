/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/spatial.hpp>

namespace tf::py {

auto register_spatial_module(nanobind::module_ &m) -> void {
  auto spatial_module = m.def_submodule("spatial", "Spatial operations");
  register_point_cloud_neighbor_search(spatial_module);
  register_mesh_neighbor_search(spatial_module);
  register_edge_mesh_neighbor_search(spatial_module);
  register_point_cloud_ray_cast(spatial_module);
  register_mesh_ray_cast(spatial_module);
  register_edge_mesh_ray_cast(spatial_module);
}

} // namespace tf::py
