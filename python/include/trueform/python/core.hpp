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

// Forward declarations for core module registration functions
auto register_offset_blocked_array(nanobind::module_ &m) -> void;

auto register_core_closest_metric_point_pair(nanobind::module_ &m) -> void;

auto register_core_ray_cast(nanobind::module_ &m) -> void;

auto register_core_intersects(nanobind::module_ &m) -> void;

auto register_core_distance(nanobind::module_ &m) -> void;

auto register_core_distance_field(nanobind::module_ &m) -> void;

auto register_core(nanobind::module_ &m) -> void;

} // namespace tf::py
