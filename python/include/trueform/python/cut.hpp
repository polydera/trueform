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

// Forward declarations for cut module registration
auto register_cut_isobands(nanobind::module_ &m) -> void;
auto register_cut_boolean(nanobind::module_ &m) -> void;
auto register_cut_embedded_self_intersection_curves(nanobind::module_ &m) -> void;
auto register_cut_embedded_intersection_curves(nanobind::module_ &m) -> void;

auto register_cut(nanobind::module_ &m) -> void;

} // namespace tf::py
