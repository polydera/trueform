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

auto register_csg(nanobind::module_ &m) -> void;

auto register_csg_graph_int3float3d(nanobind::module_ &m) -> void;
auto register_csg_graph_int3double3d(nanobind::module_ &m) -> void;
auto register_csg_graph_int643float3d(nanobind::module_ &m) -> void;
auto register_csg_graph_int643double3d(nanobind::module_ &m) -> void;

auto register_outer_shell_int3float3d(nanobind::module_ &m) -> void;
auto register_outer_shell_int3double3d(nanobind::module_ &m) -> void;
auto register_outer_shell_int643float3d(nanobind::module_ &m) -> void;
auto register_outer_shell_int643double3d(nanobind::module_ &m) -> void;

auto register_csg_boolean(nanobind::module_ &m) -> void;

} // namespace tf::py
