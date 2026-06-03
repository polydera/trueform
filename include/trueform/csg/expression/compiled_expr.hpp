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
#include "../../core/buffer.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace tf::csg {

/// @ingroup csg
/// @brief Compiled boolean expression.
///
/// Flat-buffer representation (mirrors @ref tf::tree): one
/// `tf::buffer<node>` packed in pre-order DFS, one
/// `tf::buffer<index_type>` pool of child node indices, one
/// `tf::buffer<word_mask>` pool of leaf word-masks. Each `node` is a
/// `kind` tag plus a `[lo, hi)` range; the range either points into
/// `_words` (for `leaf_cluster_*` kinds) or into `_children` (every
/// other kind).
///
/// Same-kind chains of leaves collapse into one `leaf_cluster_*` at
/// compile time, so eval cost scales with #words touched, not
/// #operands: `any_of({1, …, 200})` becomes a single node holding
/// ~7 `(word, mask)` pairs.
class compiled_expr {
public:
  using index_type = std::int32_t;

  struct word_mask {
    int idx;
    std::uint32_t mask;
  };

  enum class kind : std::uint8_t {
    leaf_cluster_any, ///< OR of all words: `(bv[w] & mask) != 0`.
    leaf_cluster_all, ///< AND of all words: `(bv[w] & mask) == mask`.
    merge,
    intersection,
    difference,
    complement,
  };

  struct node {
    kind k;
    index_type lo;
    index_type hi;
  };

  static auto make_leaf_cluster_any(std::vector<word_mask> words)
      -> compiled_expr {
    return _make_leaf(kind::leaf_cluster_any, std::move(words));
  }
  static auto make_leaf_cluster_all(std::vector<word_mask> words)
      -> compiled_expr {
    return _make_leaf(kind::leaf_cluster_all, std::move(words));
  }
  static auto make_merge(std::vector<compiled_expr> children) -> compiled_expr {
    return _make_nary(kind::merge, std::move(children));
  }
  static auto make_intersection(std::vector<compiled_expr> children)
      -> compiled_expr {
    return _make_nary(kind::intersection, std::move(children));
  }
  static auto make_difference(compiled_expr lhs, compiled_expr rhs)
      -> compiled_expr {
    std::vector<compiled_expr> kids;
    kids.reserve(2);
    kids.push_back(std::move(lhs));
    kids.push_back(std::move(rhs));
    return _make_nary(kind::difference, std::move(kids));
  }
  static auto make_complement(compiled_expr x) -> compiled_expr {
    std::vector<compiled_expr> kids;
    kids.reserve(1);
    kids.push_back(std::move(x));
    return _make_nary(kind::complement, std::move(kids));
  }

  template <typename BV> auto evaluate(const BV &bv) const -> bool {
    return _evaluate(index_type(0), bv);
  }

  auto evaluator() const & {
    return [e = *this](const auto &bv) { return e.evaluate(bv); };
  }
  auto evaluator() && {
    return [e = std::move(*this)](const auto &bv) { return e.evaluate(bv); };
  }

private:
  static auto _make_leaf(kind k, std::vector<word_mask> words)
      -> compiled_expr {
    compiled_expr e;
    e._words.reserve(words.size());
    for (const auto &w : words)
      e._words.push_back(w);
    e._nodes.push_back({k, index_type(0), index_type(words.size())});
    return e;
  }

  static auto _make_nary(kind k, std::vector<compiled_expr> children)
      -> compiled_expr {
    compiled_expr e;
    const auto n = index_type(children.size());
    for (index_type i = 0; i < n; ++i)
      e._children.push_back(index_type(-1));
    e._nodes.push_back({k, index_type(0), n});
    for (index_type i = 0; i < n; ++i)
      e._children[i] = e._append_subtree(children[i]);
    return e;
  }

  auto _append_subtree(const compiled_expr &src) -> index_type {
    const index_type node_offset = index_type(_nodes.size());
    const index_type child_offset = index_type(_children.size());
    const index_type word_offset = index_type(_words.size());
    for (const auto &nd : src._nodes) {
      const bool leaf =
          (nd.k == kind::leaf_cluster_any || nd.k == kind::leaf_cluster_all);
      const index_type off = leaf ? word_offset : child_offset;
      _nodes.push_back({nd.k, nd.lo + off, nd.hi + off});
    }
    for (index_type c : src._children)
      _children.push_back(c + node_offset);
    for (const auto &w : src._words)
      _words.push_back(w);
    return node_offset;
  }

  template <typename BV>
  auto _evaluate(index_type n, const BV &bv) const -> bool {
    const auto &nd = _nodes[n];
    switch (nd.k) {
    case kind::leaf_cluster_any:
      for (index_type w = nd.lo; w < nd.hi; ++w)
        if (bv[_words[w].idx] & _words[w].mask)
          return true;
      return false;
    case kind::leaf_cluster_all:
      for (index_type w = nd.lo; w < nd.hi; ++w)
        if ((bv[_words[w].idx] & _words[w].mask) != _words[w].mask)
          return false;
      return true;
    case kind::merge:
      for (index_type c = nd.lo; c < nd.hi; ++c)
        if (_evaluate(_children[c], bv))
          return true;
      return false;
    case kind::intersection:
      for (index_type c = nd.lo; c < nd.hi; ++c)
        if (!_evaluate(_children[c], bv))
          return false;
      return true;
    case kind::difference:
      return _evaluate(_children[nd.lo], bv) &&
             !_evaluate(_children[nd.lo + 1], bv);
    case kind::complement:
      return !_evaluate(_children[nd.lo], bv);
    }
    return false;
  }

  tf::buffer<node> _nodes;
  tf::buffer<index_type> _children;
  tf::buffer<word_mask> _words;
};

} // namespace tf::csg
