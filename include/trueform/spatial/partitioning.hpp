/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./external/miniselect/floyd_rivest_select.h"
#include "./external/miniselect/heap_select.h"
#include "./external/miniselect/median_of_3_random.h"
#include "./external/miniselect/median_of_medians.h"
#include "./external/miniselect/median_of_ninthers.h"
#include "./external/miniselect/pdqselect.h"

namespace tf {
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
using pdq_t = tree_partition::pdq;
static constexpr pdq_t pdq;
//
using nth_element_t = tree_partition::nth_element;
static constexpr nth_element_t nth_element;
//
using floyd_rivest_t = tree_partition::floyd_rivest;
static constexpr floyd_rivest_t floyd_rivest;
//
using median_of_medians_t = tree_partition::median_of_medians;
static constexpr median_of_medians_t median_of_medians;
//
using median_of_ninthers_t = tree_partition::median_of_ninthers;
static constexpr median_of_ninthers_t median_of_ninthers;
//
using median_of_3_random_t = tree_partition::median_of_3_random;
static constexpr median_of_3_random_t median_of_3_random;
//
using heap_select_t = tree_partition::heap_select;
static constexpr heap_select_t heap_select;
} // namespace strategy

} // namespace tf
