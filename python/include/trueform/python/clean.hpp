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

// Unified clean module registration
auto register_clean(nanobind::module_ &m) -> void;

// Forward declarations for clean module registration (internal)
auto register_clean_cleaned(nanobind::module_ &m) -> void;

} // namespace tf::py
