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

#include "./external/miniselect/floyd_rivest_select.h"
#include "./external/miniselect/heap_select.h"
#include "./external/miniselect/median_of_3_random.h"
#include "./external/miniselect/median_of_medians.h"
#include "./external/miniselect/median_of_ninthers.h"
#include "./external/miniselect/pdqselect.h"

namespace tf {

/// @defgroup spatial_partitioners Partitioning Strategies
/// @ingroup spatial_configuration
/// @brief Partitioning algorithms for tree construction.
///
/// These strategies control how primitives are partitioned during tree
/// construction. The default is @ref tf::spatial::nth_element.
///
/// Available strategies:
/// - @ref tf::spatial::nth_element - Standard library nth_element (default)
/// - @ref tf::spatial::pdq - Pattern-defeating quickselect
/// - @ref tf::spatial::floyd_rivest - Floyd-Rivest selection
/// - @ref tf::spatial::median_of_medians - Median of medians (guaranteed O(n))
/// - @ref tf::spatial::median_of_ninthers - Median of ninthers
/// - @ref tf::spatial::median_of_3_random - Randomized median of 3
/// - @ref tf::spatial::heap_select - Heap-based selection

namespace tree_partition {
struct pdq {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::pdqselect_branchless(std::move(begin), std::move(mid),
                                            std::move(end), std::move(compare));
  }
};

struct floyd_rivest {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::floyd_rivest_select(std::move(begin), std::move(mid),
                                           std::move(end), std::move(compare));
  }
};

struct median_of_medians {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::median_of_medians_select(
        std::move(begin), std::move(mid), std::move(end), std::move(compare));
  }
};

struct median_of_ninthers {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::median_of_ninthers_select(
        std::move(begin), std::move(mid), std::move(end), std::move(compare));
  }
};

struct median_of_3_random {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::median_of_3_random_select(
        std::move(begin), std::move(mid), std::move(end), std::move(compare));
  }
};

struct heap_select {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return miniselect::heap_select(std::move(begin), std::move(mid),
                                   std::move(end), std::move(compare));
  }
};

struct nth_element {
  template <typename Iter, typename F>
  static auto partition(Iter begin, Iter mid, Iter end, F compare) {
    return std::nth_element(std::move(begin), std::move(mid), std::move(end),
                            std::move(compare));
  }
};
} // namespace tree_partition

namespace spatial {

/// @ingroup spatial_partitioners
/// @brief Pattern-defeating quickselect partitioner.
using pdq_t = tree_partition::pdq;
static constexpr pdq_t pdq;
//
/// @ingroup spatial_partitioners
/// @brief Standard library nth_element partitioner (default).
using nth_element_t = tree_partition::nth_element;
static constexpr nth_element_t nth_element;
//
/// @ingroup spatial_partitioners
/// @brief Floyd-Rivest selection partitioner.
using floyd_rivest_t = tree_partition::floyd_rivest;
static constexpr floyd_rivest_t floyd_rivest;
//
/// @ingroup spatial_partitioners
/// @brief Median of medians partitioner (guaranteed O(n)).
using median_of_medians_t = tree_partition::median_of_medians;
static constexpr median_of_medians_t median_of_medians;
//
/// @ingroup spatial_partitioners
/// @brief Median of ninthers partitioner.
using median_of_ninthers_t = tree_partition::median_of_ninthers;
static constexpr median_of_ninthers_t median_of_ninthers;
//
/// @ingroup spatial_partitioners
/// @brief Randomized median of 3 partitioner.
using median_of_3_random_t = tree_partition::median_of_3_random;
static constexpr median_of_3_random_t median_of_3_random;
//
/// @ingroup spatial_partitioners
/// @brief Heap-based selection partitioner.
using heap_select_t = tree_partition::heap_select;
static constexpr heap_select_t heap_select;
} // namespace strategy

} // namespace tf
