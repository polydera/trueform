/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <cmath>
namespace tf::spatial {
template <typename Index>
auto max_nodes_in_tree(Index n_elements, Index inner_size, Index leaf_size) {
  n_elements = (n_elements + leaf_size - 1) / leaf_size;
  Index sum = 1;
  Index prod = 1;
  for (Index i = 1;
       i <= Index(std::ceil(std::log(n_elements) / std::log(inner_size)));
       ++i) {
    prod *= inner_size;
    sum += prod;
  }
  return sum;
}
} // namespace tf::implementation
