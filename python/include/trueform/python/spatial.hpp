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
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for spatial form types
void register_point_cloud(nanobind::module_ &m);

void register_mesh(nanobind::module_ &m);

void register_edge_mesh(nanobind::module_ &m);

void register_point_cloud_neighbor_search(nanobind::module_ &m);

void register_mesh_neighbor_search(nanobind::module_ &m);

void register_edge_mesh_neighbor_search(nanobind::module_ &m);

void register_point_cloud_neighbor_search_point_cloud(nanobind::module_ &m);

void register_edge_mesh_neighbor_search_edge_mesh(nanobind::module_ &m);

void register_edge_mesh_neighbor_search_point_cloud(nanobind::module_ &m);

void register_mesh_neighbor_search_point_cloud(nanobind::module_ &m);

void register_mesh_neighbor_search_edge_mesh(nanobind::module_ &m);

void register_mesh_neighbor_search_mesh(nanobind::module_ &m);

void register_point_cloud_ray_cast(nanobind::module_ &m);

void register_mesh_ray_cast(nanobind::module_ &m);

void register_edge_mesh_ray_cast(nanobind::module_ &m);

void register_mesh_intersects_primitive(nanobind::module_ &m);

void register_edge_mesh_intersects_primitive(nanobind::module_ &m);

void register_point_cloud_intersects_primitive(nanobind::module_ &m);

void register_mesh_gather_ids_primitive(nanobind::module_ &m);

void register_edge_mesh_gather_ids_primitive(nanobind::module_ &m);

void register_point_cloud_gather_ids_primitive(nanobind::module_ &m);

void register_point_cloud_gather_ids_point_cloud(nanobind::module_ &m);

void register_edge_mesh_gather_ids_edge_mesh(nanobind::module_ &m);

void register_mesh_gather_ids_edge_mesh(nanobind::module_ &m);

void register_mesh_gather_ids_mesh(nanobind::module_ &m);

void register_edge_mesh_gather_ids_point_cloud(nanobind::module_ &m);

void register_mesh_gather_ids_point_cloud(nanobind::module_ &m);

void register_point_cloud_intersects_point_cloud(nanobind::module_ &m);

void register_edge_mesh_intersects_edge_mesh(nanobind::module_ &m);

void register_edge_mesh_intersects_point_cloud(nanobind::module_ &m);

void register_mesh_intersects_point_cloud(nanobind::module_ &m);

void register_mesh_intersects_edge_mesh(nanobind::module_ &m);

void register_mesh_intersects_mesh(nanobind::module_ &m);

void register_spatial_module(nanobind::module_ &m);

} // namespace tf::py
