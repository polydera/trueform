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

// Forward declarations for arrangement module registration
auto register_arrangement_mesh_arrangements(nanobind::module_ &m) -> void;
auto register_arrangement_polygon_arrangement(nanobind::module_ &m) -> void;

auto register_arrangement(nanobind::module_ &m) -> void;

} // namespace tf::py
