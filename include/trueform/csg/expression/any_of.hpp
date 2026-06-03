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
#include "./expr.hpp"
#include "./to_expr.hpp"
#include <initializer_list>
#include <utility>
#include <vector>

namespace tf::csg {

/// @ingroup csg
/// @brief n-ary `merge` over a range. Each element is promoted via
///        @ref detail::to_expr — integral elements become leaves,
///        @ref expr values pass through.
///
/// `Range` must expose `.size()` (`tf::range`, `std::vector`,
/// `std::array`, etc.).
template <typename Range>
auto any_of(Range &&items) -> expr {
  std::vector<expr> children;
  children.reserve(items.size());
  for (auto &&x : items)
    children.emplace_back(detail::to_expr(std::forward<decltype(x)>(x)));
  return expr{expr::kind::merge, std::move(children)};
}

/// @ingroup csg
/// @brief @ref any_of overload accepting a braced `{...}` literal.
///        Lets users write `any_of({1, 2, 3})` directly — template
///        argument deduction can't otherwise deduce a range type
///        from a braced-init-list.
template <typename T>
auto any_of(std::initializer_list<T> items) -> expr {
  std::vector<expr> children;
  children.reserve(items.size());
  for (const auto &x : items)
    children.emplace_back(detail::to_expr(x));
  return expr{expr::kind::merge, std::move(children)};
}

} // namespace tf::csg
