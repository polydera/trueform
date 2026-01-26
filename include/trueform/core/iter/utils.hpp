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

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
namespace tf::iter {

template <typename Iterator>
struct iterator_traits : std::iterator_traits<Iterator> {};

template <typename Iterator0, typename Iterator1>
struct iterator_traits<std::pair<Iterator0, Iterator1>> {
  using iterator_category = std::common_type_t<
      typename std::iterator_traits<Iterator0>::iterator_category,
      typename std::iterator_traits<Iterator1>::iterator_category>;
  using difference_type = std::common_type_t<
      typename std::iterator_traits<Iterator0>::difference_type,
      typename std::iterator_traits<Iterator1>::difference_type>;
};

template <typename... Iterators>
struct iterator_traits<std::tuple<Iterators...>> {
  using iterator_category = std::common_type_t<
      typename std::iterator_traits<Iterators>::iterator_category...>;
  using difference_type = std::common_type_t<
      typename std::iterator_traits<Iterators>::difference_type...>;
};

template <typename Iterator> auto add(Iterator &it, std::size_t n) { it += n; }

template <typename... Iterators>
auto add(std::tuple<Iterators...> &it, std::size_t n) {
  std::apply([n](auto &&...its) { ((its += n), ...); }, it);
}

template <typename Iterator0, typename Iterator1>
auto add(std::pair<Iterator0, Iterator1> &it, std::size_t n) {
  it.first += n;
  it.second += n;
}

template <typename Iterator> auto subtract(Iterator &it, std::size_t n) {
  it -= n;
}

template <typename... Iterators>
auto subtract(std::tuple<Iterators...> &it, std::size_t n) {
  std::apply([n](auto &&...its) { ((its -= n), ...); }, it);
}

template <typename Iterator0, typename Iterator1>
auto subtract(std::pair<Iterator0, Iterator1> &it, std::size_t n) {
  it.first -= n;
  it.second -= n;
}

template <typename Iterator> auto increment(Iterator &it) { ++it; }

template <typename... Iterators> auto increment(std::tuple<Iterators...> &it) {
  std::apply([](auto &&...its) { ((++its), ...); }, it);
}

template <typename Iterator0, typename Iterator1>
auto increment(std::pair<Iterator0, Iterator1> &it) {
  ++it.first;
  ++it.second;
}

template <typename Iterator> auto decrement(Iterator &it) { --it; }

template <typename... Iterators> auto decrement(std::tuple<Iterators...> &it) {
  std::apply([](auto &&...its) { ((--its), ...); }, it);
}

template <typename Iterator0, typename Iterator1>
auto decrement(std::pair<Iterator0, Iterator1> &it) {
  --it.first;
  --it.second;
}

template <typename Iterator>
auto difference(const Iterator &it0, const Iterator &it1)
    -> typename iterator_traits<Iterator>::difference_type {
  return it0 - it1;
}

template <typename... Iterators>
auto difference(const std::tuple<Iterators...> &it0,
                const std::tuple<Iterators...> &it1)
    -> typename iterator_traits<std::tuple<Iterators...>>::difference_type {
  return std::get<0>(it0) - std::get<0>(it1);
}

template <typename Iterator0, typename Iterator1>
auto difference(const std::pair<Iterator0, Iterator1> &it0,
                const std::pair<Iterator0, Iterator1> &it1)
    -> typename iterator_traits<std::pair<Iterator0, Iterator1>>::difference_type {
  return it0.first - it1.first;
}
} // namespace tf::iter
