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

// Forward declarations for volume module registration
auto register_volume_float(nanobind::module_ &m) -> void;
auto register_volume_double(nanobind::module_ &m) -> void;
auto register_volume_int16(nanobind::module_ &m) -> void;
auto register_volume_uint16(nanobind::module_ &m) -> void;
auto register_volume_uint8(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf(nanobind::module_ &m) -> void;

auto register_volume(nanobind::module_ &m) -> void;

} // namespace tf::py
