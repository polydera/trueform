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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "./domain_inclusions.hpp"

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Apply a boolean expression to each domain's inclusion
///        bitvector.
///
/// Returns `result[d] = E(inc.make_range()[d])` for every domain.
/// `E` is a callable taking a per-domain word-block view (the same
/// type returned by `inc.make_range()[d]`) and returning `bool`.
///
/// Typical usage: pass a lambda built from a CSG expression. For
/// `n_tags <= 32`, the block is a single `uint32_t` word; bit `i`
/// represents "this domain is inside form `i`".
template <typename Expr>
auto evaluate_per_domain(const tf::csg::graph::domain_inclusions &inc, Expr E)
    -> tf::buffer<bool> {
  auto blocks = inc.make_range();
  tf::buffer<bool> result;
  result.allocate(blocks.size());
  tf::parallel_for_each(tf::enumerate(blocks), [&](auto t) {
    auto &&[d, block] = t;
    result[d] = E(block);
  });
  return result;
}

} // namespace tf::csg::graph
