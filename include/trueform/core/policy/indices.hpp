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
#include "../static_size.hpp"
#include "../views/indirect_range.hpp"
#include "./type.hpp"
#include "./unwrap.hpp"
#include <utility>

namespace tf {
namespace policy {
template <typename Range, typename Base> struct tag_indices : Base {
  /**
   * @brief Constructs an instance.
   */
  tag_indices(const Range &_indices, const Base &base)
      : Base{base}, _indices{_indices} {}

  /**
   * @brief Constructs an instance.
   */
  tag_indices(Range &&_indices, Base &&base)
      : Base{std::move(base)}, _indices{std::move(_indices)} {}

  /**
   * @brief Returns a const reference to the indices.
   */
  auto indices() const -> const Range & { return _indices; }

  /**
   * @brief Returns a mutable reference to the indices.
   */
  auto indices() -> Range & { return _indices; }

private:
  Range _indices;

  friend auto unwrap(const tag_indices &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_indices &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_indices &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_indices &val, T &&t) {
    return tag_indices<Range, std::decay_t<T>>{val._indices,
                                               static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_indices<Range, Base>> : static_size<Base> {};

namespace policy {
template <typename Range, typename Base>
auto has_indices(type, const tag_indices<Range, Base> *) -> std::true_type;

template <typename Policy>
auto has_indices(type, const views::indirect_range<Policy> *) -> std::true_type;

auto has_indices(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_indices_policy = decltype(has_indices(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

/// @ingroup core_policies
/// @brief Inject index metadata into a range.
///
/// Adds `.indices()` accessor for indirect access patterns.
template <typename Range, typename Base>
auto tag_indices(Range &&indices, Base &&base) {
  if constexpr (has_indices_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base, policy::tag_indices<std::decay_t<Range>,
                                               std::decay_t<decltype(b_base)>>{
                               static_cast<Range &&>(indices), b_base});
  }
}

} // namespace tf

namespace std {
template <typename Range, typename Policy>
struct tuple_size<tf::policy::tag_indices<Range, Policy>> : tuple_size<Policy> {
};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::tag_indices<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
