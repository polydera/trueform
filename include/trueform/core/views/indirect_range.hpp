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

#include "../array_like.hpp"
#include "../iter/indirect_iterator.hpp"
#include "../range.hpp"
#include "../static_size.hpp"

namespace tf {
namespace views {

/// @ingroup core_ranges
/// @brief A view that indirectly accesses elements of a data range using an
/// index range.
///
/// This range uses an iterator over indices to dereference into another data
/// range, enabling indirection-based access patterns. This is useful for
/// operations like reordering or filtering without modifying the original data.
///
/// Inherits from `tf::range`, and allows access to the underlying ID range via
/// `indices()`.
///
/// @tparam Iterator The index iterator type.
/// @tparam RandomIterator The data iterator type.
/// @tparam N Static size or `tf::dynamic_size`.
template <typename Policy> class indirect_range : public Policy {

public:
  using Policy::Policy;
  using Policy::operator=;
  indirect_range() = default;
  indirect_range(const Policy &policy) : Policy{policy} {}
  indirect_range(Policy &&policy) : Policy{std::move(policy)} {}

  /// @brief Returns a range over the index IDs used for indirection.
  ///
  /// If the indirect range has static size `N`, the returned ID range is also
  /// statically sized. Otherwise, a dynamic range is returned.
  auto indices() const {
    if constexpr (tf::static_size_v<Policy> == tf::dynamic_size)
      return tf::make_range(Policy::begin().base_iter(),
                            Policy::end().base_iter());
    else
      return tf::make_array_like(tf::make_range<tf::static_size_v<Policy>>(
          Policy::begin().base_iter()));
  }

  friend auto unwrap(const indirect_range &val) -> const Policy & {
    return static_cast<const Policy &>(val);
  }
  friend auto unwrap(indirect_range &val) -> Policy & {
    return static_cast<Policy &>(val);
  }
  friend auto unwrap(indirect_range &&val) -> Policy && {
    return static_cast<Policy &&>(val);
  }

  template <typename U>
  friend auto wrap_like(const indirect_range &, U &&u) {
    return indirect_range<std::decay_t<U>>{static_cast<U &&>(u)};
  }
};

template <typename Policy> auto make_indirect_range(Policy &&policy) {
  return indirect_range<std::decay_t<Policy>>{static_cast<Policy &&>(policy)};
}

template <typename Range0, typename Range1>
auto make_indirect_range_base(Range0 &&ids, Range1 &&data) {
  auto begin = tf::iter::make_indirect(ids.begin(), data.begin());
  auto end = tf::iter::make_indirect(ids.end(), data.begin());
  if constexpr (tf::static_size_v<Range0> == tf::dynamic_size)
    return tf::make_range<tf::static_size_v<Range0>>(begin, end);
  else
    return tf::make_array_like(
        tf::make_range<tf::static_size_v<Range0>>(begin, end));
}

/// @brief Specialization of `tf::static_size` for `tf::indirect_range`.
///
/// Allows `indirect_range` to propagate its static size through the type
/// system.
} // namespace views
template <typename Policy>
struct static_size<views::indirect_range<Policy>> : static_size<Policy> {};

/// @ingroup core_ranges
/// @brief Creates an `indirect_range` that views `data` using indices from
/// `ids`.
///
/// @tparam Range0 A range of integral indices.
/// @tparam Range1 A range of data values to be accessed indirectly.
/// @param ids A range of indices (e.g. `std::vector<std::size_t>`).
/// @param data A range of values to be indirectly accessed.
/// @return A `tf::indirect_range` using the given index and data ranges.
///
/// @note The resulting range is a view — it does not copy or own the underlying
/// data.
/// @note The static size is propagated using @ref tf::static_size.
template <typename Range0, typename Range1>
auto make_indirect_range(Range0 &&ids, Range1 &&data) {
  return views::make_indirect_range(views::make_indirect_range_base(
      static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data)));
}
} // namespace tf

namespace std {
template <typename Policy>
struct tuple_size<tf::views::indirect_range<Policy>> : tuple_size<Policy> {};
template <std::size_t I, typename Policy>
struct tuple_element<I, tf::views::indirect_range<Policy>>
    : tuple_element<I, Policy> {};
} // namespace std
