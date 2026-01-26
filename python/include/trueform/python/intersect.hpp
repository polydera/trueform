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

// Forward declarations for intersect module registration
auto register_intersect_isocontours(nanobind::module_ &m) -> void;
auto register_intersect_intersection_curves(nanobind::module_ &m) -> void;
auto register_intersect_self_intersection_curves(nanobind::module_ &m) -> void;

auto register_intersect(nanobind::module_ &m) -> void;

} // namespace tf::py
