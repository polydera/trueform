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
#include "../../core/small_vector.hpp"
#include "./expr.hpp"
#include "./selection_kind.hpp"
#include <algorithm>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>

namespace tf::csg {

/// @ingroup csg_expression
/// @brief What an extraction reads off the graph: an optional boolean
///        expression and a surface restriction.
///
/// The expression decides which faces bound the result and with which
/// winding; the restriction decides WHOSE faces reach the output. The
/// two are orthogonal: an empty tag list means every form's, and an
/// absent expression means no classification — the named surfaces cut
/// by everything, original winding (the embedded read).
///
/// The @ref selection_kind decides WHICH pieces of the named surfaces
/// the expression emits: the pieces bounding its region (`boundary`,
/// exactly one side in it, wound outward) or the pieces lying inside it
/// (`inside`, both sides in it, the form's stored winding). Without an
/// expression there is no region and the kind is `boundary`.
///
/// Implicitly constructible from @ref expr (and from an operand id, an
/// expression leaf): a plain expression names the boundary selection of
/// every operand. Build restricted selections with the
/// @ref selection factory: a braced list or range of ids names the
/// surfaces, an id paired with an expression restricts that expression;
/// @ref inside builds the inside read of the same pairing.
///
/// Coincident (coplanar-stacked) walls exist once in the arrangement,
/// under the stack's owner tag: a selection that includes the owner
/// carries the wall, one that excludes it does not.
class selection_t {
public:
  /// @brief This expression, every form's surface — the boundary read.
  selection_t(expr e)
      : selection_t(tf::small_vector<int, 4>{}, std::move(e),
                    selection_kind::boundary) {}

  /// @brief Operand-id leaf, every form's surface.
  template <typename T,
            typename = std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
  selection_t(T operand_id) : selection_t(expr(operand_id)) {}

  /// @brief Tags + optional expression + kind — the factories'
  ///        constructor, the sole producer of the kind.
  selection_t(tf::small_vector<int, 4> tags, std::optional<expr> e,
              selection_kind kind)
      : _expr(std::move(e)), _tags(std::move(tags)), _kind(kind) {
    std::sort(_tags.begin(), _tags.end());
    _tags.erase(std::unique(_tags.begin(), _tags.end()), _tags.end());
  }

  auto has_expression() const -> bool { return _expr.has_value(); }
  auto expression() const -> const expr & { return *_expr; }

  /// @brief Boundary or inside — what the expression emits of the
  ///        selected surfaces. Always `boundary` without an expression.
  auto kind() const -> selection_kind { return _kind; }

  /// @brief Restricted to a subset of forms? Empty tags = all of them.
  auto restricted() const -> bool { return !_tags.empty(); }
  auto tags() const -> const tf::small_vector<int, 4> & { return _tags; }

  /// @brief Per-tag emission mask over `n_tags` operands.
  template <typename Index>
  auto mask(Index n_tags) const -> tf::buffer<char> {
    tf::buffer<char> out;
    out.allocate(std::size_t(n_tags));
    std::fill(out.begin(), out.end(), _tags.empty() ? char(1) : char(0));
    for (int tag : _tags)
      if (tag >= 0 && Index(tag) < n_tags)
        out[std::size_t(tag)] = char(1);
    return out;
  }

private:
  std::optional<expr> _expr;
  tf::small_vector<int, 4> _tags;
  selection_kind _kind = selection_kind::boundary;
};

/// @cond INTERNAL
inline auto selection_tags(int tag) -> tf::small_vector<int, 4> {
  return tf::small_vector<int, 4>{tag};
}

inline auto selection_tags(std::initializer_list<int> tags)
    -> tf::small_vector<int, 4> {
  return tf::small_vector<int, 4>(tags.begin(), tags.end());
}

template <typename Range>
auto selection_tags(const Range &tags) -> tf::small_vector<int, 4> {
  tf::small_vector<int, 4> ids;
  for (auto tag : tags)
    ids.push_back(static_cast<int>(tag));
  return ids;
}

/// @endcond

/// @ingroup csg_expression
/// @brief This expression, every form's surface. (An integral argument
///        reads as an operand leaf through @ref expr's conversion.)
inline auto selection(expr e) -> selection_t {
  return selection_t{tf::small_vector<int, 4>{}, std::move(e),
                     selection_kind::boundary};
}

/// @ingroup csg_expression
/// @brief The named forms' surfaces — the embedded read: cut by every
///        other operand, once, original winding, no classification.
inline auto selection(std::initializer_list<int> tags) -> selection_t {
  return selection_t{selection_tags(tags), std::nullopt,
                     selection_kind::boundary};
}

/// @copydoc selection(std::initializer_list<int>)
template <typename Range,
          typename = std::enable_if_t<
              !std::is_integral_v<std::decay_t<Range>> &&
              !std::is_same_v<std::decay_t<Range>, expr>>,
          typename = decltype(std::declval<const Range &>().begin())>
auto selection(const Range &tags) -> selection_t {
  return selection_t{selection_tags(tags), std::nullopt,
                     selection_kind::boundary};
}

/// @ingroup csg_expression
/// @brief The boolean `e`, restricted to the faces the named forms
///        contributed — one result, split by provenance.
template <typename T,
          typename = std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
auto selection(T tag, expr e) -> selection_t {
  return selection_t{selection_tags(static_cast<int>(tag)), std::move(e),
                     selection_kind::boundary};
}

/// @copydoc selection(T, expr)
inline auto selection(std::initializer_list<int> tags, expr e) -> selection_t {
  return selection_t{selection_tags(tags), std::move(e),
                     selection_kind::boundary};
}

/// @copydoc selection(T, expr)
template <typename Range,
          typename = std::enable_if_t<
              !std::is_integral_v<std::decay_t<Range>>>,
          typename = decltype(std::declval<const Range &>().begin())>
auto selection(const Range &tags, expr e) -> selection_t {
  return selection_t{selection_tags(tags), std::move(e),
                     selection_kind::boundary};
}

/// @ingroup csg_expression
/// @brief Every form's surface lying inside the region `e` bounds —
///        both sides in it — with the stored winding. (An integral
///        argument reads as an operand leaf through @ref expr's
///        conversion.)
inline auto inside(expr e) -> selection_t {
  return selection_t{tf::small_vector<int, 4>{}, std::move(e),
                     selection_kind::inside};
}

/// @ingroup csg_expression
/// @brief The named forms' surfaces lying inside the region `e` bounds:
///        a piece is emitted iff both its sides satisfy `e`, with the
///        form's stored winding. Where `e` does not name the form this is
///        `selection(tag, op(tag) & e)`; where it does, the form's own
///        bit differs across each piece and `e` is read with each side's
///        bit — `inside(i, op(i))` is empty.
template <typename T,
          typename = std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
auto inside(T tag, expr e) -> selection_t {
  return selection_t{selection_tags(static_cast<int>(tag)), std::move(e),
                     selection_kind::inside};
}

/// @copydoc inside(T, expr)
inline auto inside(std::initializer_list<int> tags, expr e) -> selection_t {
  return selection_t{selection_tags(tags), std::move(e),
                     selection_kind::inside};
}

/// @copydoc inside(T, expr)
template <typename Range,
          typename = std::enable_if_t<!std::is_integral_v<std::decay_t<Range>>>,
          typename = decltype(std::declval<const Range &>().begin())>
auto inside(const Range &tags, expr e) -> selection_t {
  return selection_t{selection_tags(tags), std::move(e),
                     selection_kind::inside};
}

} // namespace tf::csg
