/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/metric_point_pair.hpp"
#include "./tree_metric_info_pair.hpp"

namespace tf {
template <typename Index, typename RealT, std::size_t Dims>
using nearest_neighbor_pair =
    tree_metric_info_pair<Index, tf::metric_point_pair<RealT, Dims>>;
}
