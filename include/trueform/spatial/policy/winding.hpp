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
#include "../../core/policy/unwrap.hpp"
#include "../../core/static_size.hpp"
#include "../winding_moments.hpp"
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {
namespace policy {
template <typename Range, typename Base> struct tag_winding;

template <typename Range, typename Base>
auto has_winding(const tag_winding<Range, Base> *) -> std::true_type;
auto has_winding(const void *) -> std::false_type;
} // namespace policy

/// @ingroup spatial_policies
/// @brief Checks if a type has winding moments attached.
///
/// True if the type was wrapped with `tf::tag(winding_moments)`.
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_winding_policy = decltype(policy::has_winding(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_winding : Base {
  using Base::operator=;

  tag_winding(Range _winding_moments, const Base &base)
      : Base{base}, _winding_moments{std::move(_winding_moments)} {}

  tag_winding(Range _winding_moments, Base &&base)
      : Base{std::move(base)}, _winding_moments{std::move(_winding_moments)} {}

  auto winding_moments() const -> const Range & { return _winding_moments; }

  auto winding_moments() -> Range & { return _winding_moments; }

private:
  Range _winding_moments;

  friend auto unwrap(const tag_winding &val) -> const Base & {
    return static_cast<const Base &>(val);
  }
  friend auto unwrap(tag_winding &val) -> Base & {
    return static_cast<Base &>(val);
  }
  friend auto unwrap(tag_winding &&val) -> Base && {
    return static_cast<Base &&>(val);
  }
  template <typename T> friend auto wrap_like(const tag_winding &val, T &&t) {
    return tag_winding<Range, std::decay_t<T>>{val._winding_moments,
                                               static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_winding<Range, Base>> : static_size<Base> {};

template <typename Range, typename Base>
auto tag_winding(Range &&_winding_moments, Base &&base) {
  if constexpr (has_winding_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base,
                     policy::tag_winding<std::decay_t<Range>,
                                         std::decay_t<decltype(b_base)>>{
                         static_cast<Range &&>(_winding_moments), b_base});
  }
}

namespace policy {
template <typename Range> struct tag_winding_op {
  Range winding_moments_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_winding_op<Range> t) {
  return tf::tag_winding(std::move(t.winding_moments_range),
                         static_cast<U &&>(u));
}
} // namespace policy

template <typename RealT> auto tag_winding(tf::winding_moments<RealT> &m) {
  return policy::tag_winding_op<decltype(m.moments())>{m.moments()};
}

template <typename RealT>
auto tag_winding(const tf::winding_moments<RealT> &m) {
  return policy::tag_winding_op<decltype(m.moments())>{m.moments()};
}

template <typename RealT>
auto tag_winding(tf::winding_moments<RealT> &&m) = delete;

/// Generic overload of @ref tf::tag() that auto-detects the spatial type.
/// Equivalent to `tf::tag_winding(m)`.
///
/// @tparam RealT The moments' coordinate type.
/// @param m The winding moments structure.
/// @return A tag operator for use with pipe syntax.
template <typename RealT> auto tag(tf::winding_moments<RealT> &m) {
  return tag_winding(m);
}

/// @overload
template <typename RealT> auto tag(const tf::winding_moments<RealT> &m) {
  return tag_winding(m);
}

template <typename RealT> auto tag(tf::winding_moments<RealT> &&m) = delete;

} // namespace tf

namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_winding<Range, Base>> : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_winding<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
