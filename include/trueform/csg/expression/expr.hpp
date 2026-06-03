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
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::csg {

class compiled_expr;

/// @ingroup csg
/// @brief Runtime boolean expression over operand bits.
///
/// An @ref expr is either a *leaf* (an operand id `i` representing
/// "this domain is inside form `i`") or an internal node combining
/// children with `merge` / `intersection` / `difference` /
/// `complement`. Build them with the free-function builders in the
/// `tf::csg::` namespace; integral arguments are auto-promoted to
/// leaves.
///
/// @code{.cpp}
/// auto bunnies = tf::csg::any_of(tf::make_sequence_range(1, n + 1));
/// auto tree    = tf::csg::difference(0, bunnies);
///
/// auto E_walk  = tree.evaluator();              // per-leaf walk
/// auto E_fast  = tree.compile().evaluator();    // word-mask lowered
/// @endcode
///
/// The leaf id `i` maps to bit `i` of the per-domain inclusion
/// bitvector. Both evaluators are value-capturing lambdas — the
/// underlying tree may go out of scope after construction.
class expr {
public:
  enum class kind {
    leaf,         ///< operand bit lookup.
    merge,        ///< OR over children (a.k.a. union).
    intersection, ///< AND over children.
    difference,   ///< children[0] AND NOT children[1..].
    complement,   ///< NOT children[0].
  };

  /// @brief Construct a leaf from an integral operand id.
  template <typename T,
            typename = std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
  expr(T operand_id)
      : _kind(kind::leaf), _operand_id(static_cast<int>(operand_id)) {}

  /// @brief Construct an internal node. Used by the builders.
  expr(kind k, std::vector<expr> children)
      : _kind(k), _children(std::move(children)) {}

  /// @brief Evaluate by walking the tree once per call. `bv` must be
  ///        word-indexable (`bv[w]` returns a `uint32_t`).
  template <typename BV>
  auto evaluate(const BV &bv) const -> bool {
    return _evaluate(bv);
  }

  /// @brief Return a callable `(bv) -> bool` that evaluates this
  ///        tree by walking. Owns a copy of the tree.
  auto evaluator() const {
    return [tree = *this](const auto &bv) -> bool {
      return tree._evaluate(bv);
    };
  }

  /// @brief Lower to a word-mask form: leaf clusters under
  ///        `merge` / `intersection` collapse to per-word masks.
  ///        Defined in `compile.hpp`.
  auto compile() const -> compiled_expr;

  auto node_kind() const -> kind { return _kind; }
  auto operand_id() const -> int { return _operand_id; }
  auto children() const -> const std::vector<expr> & { return _children; }

private:
  kind _kind = kind::leaf;
  int _operand_id = 0;
  std::vector<expr> _children;

  template <typename BV>
  auto _evaluate(const BV &bv) const -> bool {
    switch (_kind) {
    case kind::leaf: {
      const auto i = _operand_id;
      return (bv[i / 32] & (std::uint32_t(1) << (i % 32))) != 0;
    }
    case kind::merge:
      for (const auto &c : _children)
        if (c._evaluate(bv))
          return true;
      return false;
    case kind::intersection:
      for (const auto &c : _children)
        if (!c._evaluate(bv))
          return false;
      return true;
    case kind::difference:
      if (_children.empty() || !_children.front()._evaluate(bv))
        return false;
      for (std::size_t i = 1; i < _children.size(); ++i)
        if (_children[i]._evaluate(bv))
          return false;
      return true;
    case kind::complement:
      return !_children.front()._evaluate(bv);
    }
    return false;
  }
};

} // namespace tf::csg
