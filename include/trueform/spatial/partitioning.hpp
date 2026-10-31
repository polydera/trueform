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

#include "../core/algorithm/nth_element.hpp"
#include <utility>

namespace tf {

/// @defgroup spatial_partitioners Partitioning Strategies
/// @ingroup spatial_configuration
/// @brief Partitioning algorithm for tree construction.
///
/// The tree is built with @ref tf::spatial::pdq, the branchless
/// pattern-defeating quickselect of @ref tf::nth_element.

namespace tree_partition {
struct pdq {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return core::pdqselect<true>(std::move(begin), std::move(mid),
                                 std::move(end), std::move(compare));
  }
};
} // namespace tree_partition

namespace spatial {

/// @ingroup spatial_partitioners
/// @brief Pattern-defeating quickselect partitioner (default).
using pdq_t = tree_partition::pdq;
static constexpr pdq_t pdq;
} // namespace spatial

} // namespace tf
