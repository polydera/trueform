/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for intersect module registration
auto register_intersect_isocontours(nanobind::module_ &m) -> void;
auto register_intersect_isobands(nanobind::module_ &m) -> void;

auto register_intersect(nanobind::module_ &m) -> void;

} // namespace tf::py
