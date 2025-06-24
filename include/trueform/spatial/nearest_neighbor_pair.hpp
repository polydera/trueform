/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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
