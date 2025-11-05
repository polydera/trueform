/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/metric_point.hpp"
#include "./tree_metric_info.hpp"

namespace tf {
template <typename Index, typename RealT, std::size_t Dims>
using nearest_neighbor =
    tree_metric_info<Index, tf::metric_point<RealT, Dims>>;
}
