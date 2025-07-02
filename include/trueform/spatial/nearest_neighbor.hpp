/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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
