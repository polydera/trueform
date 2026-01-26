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
#include "./iter/zip_iterator.hpp"
#include "./static_size.hpp"
#include "./tuple.hpp"

namespace tf {
namespace core {
template <typename... Ranges> class zip_range {
public:
  using iterator =
      decltype(iter::make_zip_iterator(std::declval<Ranges &>().begin()...));
  using const_iterator = decltype(iter::make_zip_iterator(
      std::declval<const Ranges &>().begin()...));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using size_type = typename std::iterator_traits<iterator>::difference_type;

  zip_range() = default;
  zip_range(const Ranges &..._ranges) : _ranges{_ranges...} {}
  zip_range(Ranges &&..._ranges) : _ranges{std::move(_ranges)...} {}

  auto begin() const -> const_iterator {
    return std::apply(
        [](auto &&...ranges) {
          return iter::make_zip_iterator(ranges.begin()...);
        },
        _ranges);
  }

  auto begin() -> iterator {
    return std::apply(
        [](auto &&...ranges) {
          return iter::make_zip_iterator(ranges.begin()...);
        },
        _ranges);
  }

  auto end() const -> const_iterator {
    return std::apply(
        [](auto &&...ranges) {
          return iter::make_zip_iterator(ranges.end()...);
        },
        _ranges);
  }

  auto end() -> iterator {
    return std::apply(
        [](auto &&...ranges) {
          return iter::make_zip_iterator(ranges.end()...);
        },
        _ranges);
  }

  auto size() const -> size_type {
    using std::get;
    return get<0>(_ranges).size();
  }

  auto empty() const -> bool { return size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto back() const -> const_reference { return *(end() - 1); }

  auto back() -> reference { return *(end() - 1); }

  auto operator[](std::size_t i) const -> const_reference {
    return *(begin() + i);
  }

  auto ranges() const -> const tf::tuple<Ranges...> & { return _ranges; }

  auto ranges() -> tf::tuple<Ranges...> & { return _ranges; }

  auto operator[](std::size_t i) -> reference { return *(begin() + i); }

  template <std::size_t I>
  friend auto get(const zip_range &r) -> decltype(auto) {
    using std::get;
    return get<I>(r._ranges);
  }

  template <std::size_t I> friend auto get(zip_range &r) -> decltype(auto) {
    using std::get;
    return get<I>(r._ranges);
  }

  template <std::size_t I> friend auto get(zip_range &&r) -> decltype(auto) {
    using std::get;
    return get<I>(r._ranges);
  }

private:
  tf::tuple<Ranges...> _ranges;
};

template <typename Range> auto make_zip_range(Range &&r) -> Range && {
  return static_cast<Range &&>(r);
}
template <typename Range0, typename Range1, typename... Ranges>
auto make_zip_range(Range0 &&r0, Range1 &&r1, Ranges &&...r) {
  return tf::core::zip_range<std::decay_t<Range0>, std::decay_t<Range1>,
                             std::decay_t<Ranges>...>(
      static_cast<Range0 &&>(r0), static_cast<Range1 &&>(r1),
      static_cast<Ranges &&>(r)...);
}
} // namespace core

template <typename Range, typename... Ranges>
struct static_size<core::zip_range<Range, Ranges...>> : static_size<Range> {};
} // namespace tf
namespace std {

template <std::size_t I, typename... Ts>
struct tuple_element<I, tf::core::zip_range<Ts...>>
    : tuple_element<I, tf::tuple<Ts...>> {};

template <typename... Ts>
struct tuple_size<tf::core::zip_range<Ts...>> : tuple_size<tf::tuple<Ts...>> {};

} // namespace std
