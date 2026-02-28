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
#include <trueform/python/spatial.hpp>

namespace tf::py {

auto register_spatial_module(nanobind::module_ &m) -> void {
  // Create spatial submodule
  auto spatial_module = m.def_submodule("spatial", "Spatial operations");

  // Register spatial form wrapper classes to submodule
  register_point_cloud(spatial_module);
  register_mesh(spatial_module);
  register_edge_mesh(spatial_module);

  // Register spatial operations (form x form)
  register_point_cloud_neighbor_search_point_cloud(spatial_module);
  register_edge_mesh_neighbor_search_edge_mesh(spatial_module);
  register_edge_mesh_neighbor_search_point_cloud(spatial_module);
  register_mesh_neighbor_search_point_cloud(spatial_module);
  register_mesh_neighbor_search_edge_mesh(spatial_module);
  register_mesh_neighbor_search_mesh(spatial_module);
  register_point_cloud_gather_ids_point_cloud(spatial_module);
  register_edge_mesh_gather_ids_edge_mesh(spatial_module);
  register_edge_mesh_gather_ids_point_cloud(spatial_module);
  register_mesh_gather_ids_edge_mesh(spatial_module);
  register_mesh_gather_ids_point_cloud(spatial_module);
  register_mesh_gather_ids_mesh(spatial_module);
  register_point_cloud_intersects_point_cloud(spatial_module);
  register_edge_mesh_intersects_edge_mesh(spatial_module);
  register_edge_mesh_intersects_point_cloud(spatial_module);
  register_mesh_intersects_point_cloud(spatial_module);
  register_mesh_intersects_edge_mesh(spatial_module);
  register_mesh_intersects_mesh(spatial_module);

  // Unified prim × prim operations (dispatch-based)
  register_prim_prim_ops(spatial_module);

  // Primitive geometry operations (area, normals)
  register_prim_geometry_ops(spatial_module);

  // Unified form × prim operations (dispatch-based)
  register_mesh_fp(spatial_module);
  register_edge_mesh_fp(spatial_module);
  register_point_cloud_fp(spatial_module);
}

} // namespace tf::py
