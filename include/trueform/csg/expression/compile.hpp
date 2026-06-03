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
#include "./compile_cluster_node.hpp"
#include "./compiled_expr.hpp"
#include "./expr.hpp"
#include "./make_word_mask.hpp"
#include <utility>
#include <vector>

namespace tf::csg {

/// @brief Lower an @ref expr to a @ref compiled_expr — leaf clusters
///        under `merge` / `intersection` collapse into per-word
///        masks; mixed and non-cluster nodes recurse.
inline auto expr::compile() const -> compiled_expr {
  switch (_kind) {
  case kind::leaf:
    return compiled_expr::make_leaf_cluster_any(
        {detail::make_word_mask(_operand_id)});
  case kind::merge:
    return detail::compile_cluster_node(
        _children, &compiled_expr::make_leaf_cluster_any,
        compiled_expr::kind::merge);
  case kind::intersection:
    return detail::compile_cluster_node(
        _children, &compiled_expr::make_leaf_cluster_all,
        compiled_expr::kind::intersection);
  case kind::difference: {
    // Empty difference matches the walker's semantics (false).
    if (_children.empty())
      return compiled_expr::make_leaf_cluster_any({});
    if (_children.size() == 2)
      return compiled_expr::make_difference(_children[0].compile(),
                                              _children[1].compile());
    // n-ary difference: lhs AND NOT merge(rhs1, rhs2, ...).
    std::vector<expr> rhs;
    rhs.reserve(_children.size() - 1);
    for (std::size_t i = 1; i < _children.size(); ++i)
      rhs.push_back(_children[i]);
    auto rhs_compiled = expr{kind::merge, std::move(rhs)}.compile();
    return compiled_expr::make_difference(_children.front().compile(),
                                            std::move(rhs_compiled));
  }
  case kind::complement:
    return compiled_expr::make_complement(_children.front().compile());
  }
  return compiled_expr::make_leaf_cluster_any({});
}

} // namespace tf::csg
