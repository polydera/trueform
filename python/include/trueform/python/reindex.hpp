/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

// Unified reindex module registration
auto register_reindex(nanobind::module_ &m) -> void;

// Forward declarations for reindex module registration (internal)
auto register_reindex_reindex_by_ids(nanobind::module_ &m) -> void;
auto register_reindex_reindex_by_mask(nanobind::module_ &m) -> void;
auto register_reindex_split_into_components(nanobind::module_ &m) -> void;

} // namespace tf::py
