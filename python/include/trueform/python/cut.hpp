/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for cut module registration
auto register_cut_isobands(nanobind::module_ &m) -> void;
auto register_cut_boolean(nanobind::module_ &m) -> void;

auto register_cut(nanobind::module_ &m) -> void;

} // namespace tf::py
