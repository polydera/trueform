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
#include <cmath>
namespace tf::spatial {
template <typename Index>
auto max_nodes_in_tree(Index n_elements, int inner_size, int leaf_size) {
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
} // namespace tf::spatial
