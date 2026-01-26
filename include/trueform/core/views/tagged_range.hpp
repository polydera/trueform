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

#include "../range.hpp"
#include "../static_size.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief A lightweight view over a fixed-size range of elements
/// preceded by a tag element.
///
/// This class models a range, where the element immediately preceding
/// the range (i.e., at `begin() - 1`) is treated as a **tag**—a separate,
/// logically associated value that is not part of the iteration range but can
/// be accessed explicitly.
///
/// It inherits from @ref tf::range<Iterator, N>, providing the same interface,
/// while exposing a `tag()` accessor to retrieve the preceding element.
///
/// @tparam Iterator The iterator type.
/// @tparam N The number of elements in the range (excluding the tag).
///
/// @note The caller must guarantee that `begin() - 1` is a valid
/// dereferenceable iterator.
///       This typically implies that the tag is stored contiguously before the
///       range.
///
/// @see tf::range
template <typename Iterator, std::size_t N>
class tagged_range : public tf::range<Iterator, N> {
public:
  using tf::range<Iterator, N>::range;

  auto tag() const -> typename tf::range<Iterator, N>::reference {
    return *(tf::range<Iterator, N>::begin() - 1);
  }
};
/// @brief Specialization of @ref tf::static_size for @ref tf::tagged_range.
///
/// This trait exposes the compile-time size `N` for `tf::range<Iterator, N>`
/// so that size-aware algorithms and views can propagate it automatically.
///
/// @tparam T The element type.
/// @tparam N The static number of elements in the array.
template <typename Iterator, std::size_t N>
struct static_size<tagged_range<Iterator, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, typename Iterator_, std::size_t N_>
auto get(const tagged_range<Iterator_, N_> &s) -> decltype(auto) {
  static_assert(N_ != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < N_, "Index out of bounds");
  return s[I];
}
} // namespace tf

namespace std {
template <typename Iterator, std::size_t N>
struct tuple_size<tf::tagged_range<Iterator, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, typename Iterator, std::size_t N>
struct tuple_element<I, tf::tagged_range<Iterator, N>> {
  using type = typename std::iterator_traits<Iterator>::reference;
};

template <typename Iterator>
struct tuple_size<tf::tagged_range<Iterator, tf::dynamic_size>>;

template <std::size_t I, typename Iterator>
struct tuple_element<I, tf::tagged_range<Iterator, tf::dynamic_size>>;
} // namespace std

namespace tf {
/**
 * @ingroup core_ranges
 * @brief Constructs a `tagged_range` view by skipping the tag element.
 *
 * This function takes a range where the first element (at `begin()`) is
 * interpreted as a **tag**, and returns a `tagged_range` that begins from the
 * second element. The tag can later be accessed via `.tag()`.
 *
 * If the input range has a known static size `N`, the returned `tagged_range`
 * will have static size `N - 1`. If the input range is dynamically sized,
 * the result will be a dynamic `tagged_range`.
 *
 * @tparam Range A range type supporting `begin()` and `end()`.
 * @param r The input range, where the first element represents the tag.
 * @return A `tf::tagged_range` over the tail of the input, with `.tag()`
 * access.
 */
template <typename Range> auto make_tagged_range(Range &&r) {
  if constexpr (tf::static_size_v<Range> != tf::dynamic_size)
    return tf::tagged_range<typename std::decay_t<Range>::iterator,
                            tf::static_size_v<Range> - 1>(r.begin() + 1);
  else
    return tf::tagged_range<typename std::decay_t<Range>::iterator,
                            tf::dynamic_size>(r.begin() + 1, r.end());
}
} // namespace tf
