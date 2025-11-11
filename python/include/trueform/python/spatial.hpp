/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

void register_point_cloud_neighbor_search(nanobind::module_ &m);

void register_mesh_neighbor_search(nanobind::module_ &m);

void register_edge_mesh_neighbor_search(nanobind::module_ &m);

void register_point_cloud_ray_cast(nanobind::module_ &m);

void register_mesh_ray_cast(nanobind::module_ &m);

void register_edge_mesh_ray_cast(nanobind::module_ &m);

void register_spatial_module(nanobind::module_ &m);

} // namespace tf::py
