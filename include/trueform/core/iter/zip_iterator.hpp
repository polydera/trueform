/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../tuple.hpp"
#include "./mapped_iterator.hpp"

namespace tf::iter {

template <typename... Iterators>
struct zip_reference
    : tf::tuple<typename tf::iter::iterator_traits<Iterators>::reference...> {
  using base_t =
      tf::tuple<typename tf::iter::iterator_traits<Iterators>::reference...>;
  using base_t::operator=;
  zip_reference() = default;
  zip_reference(Iterators... iters) : base_t{*iters...} {}

  friend auto swap(zip_reference &&zr0, zip_reference &&zr1) -> void {
    using std::swap;
    return swap(static_cast<base_t &>(zr0), static_cast<base_t &>(zr1));
  }

  friend auto swap(zip_reference &zr0, zip_reference &zr1) -> void {
    using std::swap;
    return swap(static_cast<base_t &>(zr0), static_cast<base_t &>(zr1));
  }
};

template <typename... Iterators>
struct zip_value
    : tf::tuple<typename tf::iter::iterator_traits<Iterators>::value_type...> {
  using base_t =
      tf::tuple<typename tf::iter::iterator_traits<Iterators>::value_type...>;
  using base_t::operator=;
  zip_value() = default;
  zip_value(const zip_reference<Iterators...> &ref) : base_t{ref} {}
};

struct zip_dereferencer {
  template <typename... Iterators>
  auto operator()(std::tuple<Iterators...> its) const {
    using reference = zip_reference<Iterators...>;
    return std::apply([](auto... its) { return reference{its...}; }, its);
  }
};

template <typename... Iterators>
struct zip_iterator
    : mapped_crtp_picker<
          typename iterator_traits<std::tuple<Iterators...>>::iterator_category,
          zip_iterator<Iterators...>, std::tuple<Iterators...>,
          zip_dereferencer, true>::type {
private:
  using base_t = typename mapped_crtp_picker<
      typename iterator_traits<std::tuple<Iterators...>>::iterator_category,
      zip_iterator<Iterators...>, std::tuple<Iterators...>, zip_dereferencer,
      true>::type;

public:
  zip_iterator() = default;
  zip_iterator(Iterators... iters)
      : base_t{std::make_tuple(iters...), zip_dereferencer{}} {}
  using reference = zip_reference<Iterators...>;
  using value_type = zip_value<Iterators...>;
};

template <typename... Iterators> auto make_zip_iterator(Iterators... iters) {
  return zip_iterator<Iterators...>{std::move(iters)...};
}
} // namespace tf::iter
namespace std {
template <std::size_t I, typename... Ts>
struct tuple_element<I, tf::iter::zip_value<Ts...>>
    : tuple_element<I, typename tf::iter::zip_value<Ts...>::base_t> {};

template <typename... Ts>
struct tuple_size<tf::iter::zip_value<Ts...>>
    : tuple_size<typename tf::iter::zip_value<Ts...>::base_t> {};

template <std::size_t I, typename... Ts>
struct tuple_element<I, tf::iter::zip_reference<Ts...>>
    : tuple_element<I, typename tf::iter::zip_reference<Ts...>::base_t> {};

template <typename... Ts>
struct tuple_size<tf::iter::zip_reference<Ts...>>
    : tuple_size<typename tf::iter::zip_reference<Ts...>::base_t> {};
} // namespace std
