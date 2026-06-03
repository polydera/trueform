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
#include "./coalesce_words.hpp"
#include "./compiled_expr.hpp"
#include "./expr.hpp"
#include "./make_word_mask.hpp"
#include <utility>
#include <vector>

namespace tf::csg::detail {

/// @ingroup csg
/// @brief Compile a `merge` / `intersection` node by batching leaf
///        children into a single @ref tf::csg::compiled_expr
///        leaf-cluster and recursively compiling the rest.
///
/// @param children            The original tree children.
/// @param make_cluster        Factory for the leaf cluster
///                            (any/all variant).
/// @param kind_when_mixed     The compiled kind to wrap around mixed
///                            children when there is both a cluster
///                            and non-leaf children.
template <typename MakeCluster>
auto compile_cluster_node(const std::vector<expr> &children,
                          MakeCluster &&make_cluster,
                          compiled_expr::kind kind_when_mixed)
    -> compiled_expr {
  const expr::kind absorb_kind = (kind_when_mixed == compiled_expr::kind::merge)
                                     ? expr::kind::merge
                                     : expr::kind::intersection;
  std::vector<compiled_expr::word_mask> leaves;
  std::vector<compiled_expr> non_leaves;
  leaves.reserve(children.size());
  non_leaves.reserve(children.size());

  auto visit = [&](auto &self, const expr &c) -> void {
    if (c.node_kind() == expr::kind::leaf)
      leaves.push_back(make_word_mask(c.operand_id()));
    else if (c.node_kind() == absorb_kind)
      for (const auto &g : c.children())
        self(self, g);
    else
      non_leaves.push_back(c.compile());
  };
  for (const auto &c : children)
    visit(visit, c);

  auto words = coalesce_words(std::move(leaves));
  if (non_leaves.empty())
    return make_cluster(std::move(words));
  if (!words.empty())
    non_leaves.insert(non_leaves.begin(), make_cluster(std::move(words)));
  if (non_leaves.size() == 1)
    return std::move(non_leaves[0]);
  if (kind_when_mixed == compiled_expr::kind::merge)
    return compiled_expr::make_merge(std::move(non_leaves));
  return compiled_expr::make_intersection(std::move(non_leaves));
}

} // namespace tf::csg::detail
