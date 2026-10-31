/*
 * Copyright (c) 2026 XLAB
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
#include "../core/buffer.hpp"
#include "../core/config/verify.hpp"
#include "./cholesky/analyze_cholesky_order.hpp"
#include "./cholesky/make_symmetric_adjacency.hpp"
#include "./cholesky/order_minimum_degree.hpp"
#include "./cholesky_pattern.hpp"
#include "./cholesky_scratch.hpp"
#include "./symmetric_lower.hpp"
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief Read the structure of the Cholesky factor of a symmetric pattern,
/// once for every value set it will carry.
///
/// The order is @ref tf::linalg::order_minimum_degree, and
/// @ref tf::linalg::analyze_cholesky_order reads the factor's supernodes and
/// schedule under it. Only the pattern is read; `matrix.values` may be
/// absent. Every row must state its diagonal as its last entry. Returns false
/// where the index type cannot name the factor's values.
template <typename Index>
auto analyze_cholesky(const tf::symmetric_lower<Index> &matrix,
                      tf::cholesky_scratch<Index> &scratch,
                      tf::cholesky_pattern<Index> &pattern) -> bool {
  if constexpr (tf::core::verification)
    for (std::size_t i = 0; i < matrix.size(); ++i) {
      const std::size_t first = std::size_t(matrix.row_offsets[i]);
      const std::size_t last = std::size_t(matrix.row_offsets[i + 1]);
      tf::core::verify(last > first &&
                           std::size_t(matrix.columns[last - 1]) == i,
                       "a row states its diagonal last");
      for (std::size_t at = first + 1; at < last; ++at)
        tf::core::verify(matrix.columns[at - 1] < matrix.columns[at],
                         "a row's columns ascend");
    }
  tf::linalg::make_symmetric_adjacency(
      matrix, scratch.ordering.cursor, pattern.adjacency_offsets,
      pattern.adjacency, pattern.adjacency_entries);
  tf::linalg::order_minimum_degree(pattern.adjacency_offsets, pattern.adjacency,
                                   scratch.ordering, scratch.minimum_degree);
  return tf::linalg::analyze_cholesky_order(matrix, scratch.minimum_degree,
                                            scratch, pattern);
}

} // namespace tf
