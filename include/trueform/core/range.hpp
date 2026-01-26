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

#include "./static_size.hpp"
#include <iterator>

namespace tf {

/// @ingroup core_ranges
/// @brief A lightweight view over a fixed-size range of elements.
///
/// `tf::range<Iterator, N>` represents a random-access sequence of `N` elements
/// defined by a pair of iterators. It supports structure-bindings
/// decomposition.
///
/// @tparam Iterator The iterator type.
/// @tparam N The number of elements in the range.
template <typename Iterator, std::size_t N> class range {
public:
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  using reference = typename std::iterator_traits<Iterator>::reference;
  using pointer = typename std::iterator_traits<Iterator>::pointer;
  using iterator = Iterator;
  using const_iterator = Iterator;
  using size_type = std::size_t;

  range() = default;

  explicit range(Iterator it) : _begin(it) {}
  range(Iterator it, Iterator) : _begin(it) {}

  auto begin() const -> Iterator { return _begin; }
  auto end() const -> Iterator { return _begin + N; }

  constexpr auto size() const -> size_type { return N; }
  auto empty() const -> bool { return N == 0; }

  auto front() const -> reference { return *_begin; }
  auto back() const -> reference { return *(_begin + N - 1); }
  auto operator[](size_type i) const -> reference { return *(_begin + i); }

private:
  Iterator _begin = {};
};
/// @brief Specialization of @ref tf::static_size for @ref tf::range.
///
/// This trait exposes the compile-time size `N` for `tf::range<Iterator, N>`
/// so that size-aware algorithms and views can propagate it automatically.
///
/// @tparam T The element type.
/// @tparam N The static number of elements in the array.
template <typename Iterator, std::size_t N>
struct static_size<range<Iterator, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, typename Iterator_, std::size_t N_>
auto get(const range<Iterator_, N_> &s) -> decltype(auto) {
  static_assert(N_ != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < N_, "Index out of bounds");
  return s[I];
}

template <std::size_t I, typename Iterator_, std::size_t N_>
auto get(range<Iterator_, N_> &s) -> decltype(auto) {
  static_assert(N_ != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < N_, "Index out of bounds");
  return s[I];
}

template <std::size_t I, typename Iterator_, std::size_t N_>
auto get(range<Iterator_, N_> &&s) -> decltype(auto) {
  static_assert(N_ != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < N_, "Index out of bounds");
  return s[I];
}
} // namespace tf

namespace std {
template <typename Iterator, std::size_t N>
struct tuple_size<tf::range<Iterator, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, typename Iterator, std::size_t N>
struct tuple_element<I, tf::range<Iterator, N>> {
  using type = typename std::iterator_traits<Iterator>::reference;
};

template <typename Iterator>
struct tuple_size<tf::range<Iterator, tf::dynamic_size>>;

template <std::size_t I, typename Iterator>
struct tuple_element<I, tf::range<Iterator, tf::dynamic_size>>;
} // namespace std

namespace tf {

/// @ingroup core_ranges
/// @brief A lightweight view over a dynamically-sized range of elements.
///
/// This specialization allows runtime-sized ranges, storing both `begin()`
/// and `end()` iterators internally. Compatible with STL-style iteration.
///
/// @tparam Iterator The iterator type.
template <typename Iterator> class range<Iterator, tf::dynamic_size> {
public:
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  using reference = typename std::iterator_traits<Iterator>::reference;
  using pointer = typename std::iterator_traits<Iterator>::pointer;
  using iterator = Iterator;
  using const_iterator = Iterator;
  using size_type = std::size_t;

  /// @brief Default constructor for an empty range.
  range() = default;

  /// @brief Constructs a range from two iterators.
  /// @param it The beginning iterator.
  /// @param end The end iterator (one past the last element).
  range(Iterator it, Iterator end) : _begin(it), _end(end) {}

  /// @brief Returns the beginning iterator of the range.
  /// @return The iterator pointing to the first element.
  auto begin() const -> Iterator { return _begin; }

  /// @brief Returns the end iterator of the range.
  /// @return The iterator pointing one past the last element.
  auto end() const -> Iterator { return _end; }

  /// @brief Returns the number of elements in the range.
  /// @return The size (number of elements).
  auto size() const -> size_type {
    return static_cast<size_type>(_end - _begin);
  }

  /// @brief Checks whether the range is empty.
  /// @return `true` if the range contains no elements, otherwise `false`.
  auto empty() const -> bool { return _begin == _end; }

  /// @brief Returns a reference to the first element in the range.
  /// @return Reference to the first element.
  auto front() const -> reference { return *_begin; }

  /// @brief Returns a reference to the last element in the range.
  /// @return Reference to the last element.
  auto back() const -> reference { return *(_end - 1); }

  /// @brief Accesses the i-th element in the range by index.
  /// @param i The index into the range.
  /// @return Reference to the i-th element.
  auto operator[](size_type i) const -> reference { return *(_begin + i); }

private:
  Iterator _begin;
  Iterator _end;
};

/// @ingroup core_ranges
/// @brief Create a dynamically-sized range from a beginning iterator and size.
/// @param it The beginning iterator.
/// @param size The number of elements in the range.
/// @return A `range` over `[it, it + size)`.
template <typename Iterator>
auto make_range(Iterator it, std::size_t size)
    -> range<Iterator, tf::dynamic_size> {
  return range<Iterator, tf::dynamic_size>(it, it + size);
}
/// @ingroup core_ranges
/// @brief Create a fixed-size range from a beginning iterator.
/// @tparam N The number of elements (static).
/// @param it The beginning iterator.
/// @return A `range<Iterator, N>` over `[it, it + N)`.
template <std::size_t N, typename Iterator>
auto make_range(Iterator it) -> range<Iterator, N> {
  return range<Iterator, N>(it);
}
/// @ingroup core_ranges
/// @brief Create a dynamically-sized range from a pair of iterators.
/// @param it The beginning iterator.
/// @param end The past-the-end iterator.
/// @return A `range` over `[it, end)`.
template <typename Iterator>
auto make_range(Iterator it, Iterator end)
    -> range<Iterator, tf::dynamic_size> {
  return range<Iterator, tf::dynamic_size>(it, end);
}
/// @ingroup core_ranges
/// @brief Create a fixed-size or dynamic range from a pair of iterators, based
/// on the value of `N`. See @ref tf::dynamic_size
/// @tparam N The number of elements (static).
/// @param it The beginning iterator.
/// @param end The past-the-end iterator.
/// @return A `range<Iterator, N>` over `[it, end)`.
template <std::size_t N, typename Iterator>
auto make_range(Iterator it, Iterator end) -> range<Iterator, N> {
  return range<Iterator, N>(it, end);
}
/// @ingroup core_ranges
/// @brief Create a range from a container reference.
///
/// If @ref tf::static_size "tf::static_size<Container>" is specialized,
/// the static size is propagated to the resulting range. Otherwise, the range
/// will default to @ref tf::dynamic_size.
///
/// This enables compile-time optimizations in fixed-size algorithms when
/// applicable.
///
/// @tparam Container A container type that may (optionally) specialize @ref
/// tf::static_size.
/// @param c The container to view.
/// @return A `range` with either static or dynamic size, depending on
/// specialization.
template <typename Container>
auto make_range(Container &c)
    -> range<decltype(std::begin(c)), tf::static_size_v<Container>> {
  return range<decltype(std::begin(c)), tf::static_size_v<Container>>(
      std::begin(c), std::end(c));
}
/// @ingroup core_ranges
/// @brief Create a range from a const container reference.
///
/// If @ref tf::static_size "tf::static_size<Container>" is specialized,
/// the static size is propagated to the resulting range. Otherwise, the range
/// will default to @ref tf::dynamic_size.
///
/// Supports const containers and enables compile-time optimization when static
/// size information is available.
///
/// @tparam Container A container type that may (optionally) specialize @ref
/// tf::static_size.
/// @param c The const container to view.
/// @return A `range` with either static or dynamic size, depending on
/// specialization.
template <typename Container>
auto make_range(const Container &c)
    -> range<decltype(std::begin(c)), tf::static_size_v<Container>> {
  return range<decltype(std::begin(c)), tf::static_size_v<Container>>(
      std::begin(c), std::end(c));
}
} // namespace tf
