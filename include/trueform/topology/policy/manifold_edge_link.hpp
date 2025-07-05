/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../manifold_edge_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, std::size_t N, typename Base>
struct tag_manifold_edge_link;
template <typename Index, std::size_t N, typename Base>
auto has_manifold_edge_link(const tag_manifold_edge_link<Index, N, Base> *)
    -> std::true_type;

auto has_manifold_edge_link(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_manifold_edge_link_policy =
    decltype(policy::has_manifold_edge_link(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, std::size_t N, typename Base>
struct tag_manifold_edge_link : Base {
  using Base::operator=;
  tag_manifold_edge_link(tf::manifold_edge_link<Index, N> *_manifold_edge_link,
                         const Base &base)
      : Base{base}, _manifold_edge_link{_manifold_edge_link} {}

  tag_manifold_edge_link(tf::manifold_edge_link<Index, N> *_manifold_edge_link,
                         Base &&base)
      : Base{std::move(base)}, _manifold_edge_link{_manifold_edge_link} {}

  auto manifold_edge_link() const -> const tf::manifold_edge_link<Index, N> & {
    return *_manifold_edge_link;
  }

  auto manifold_edge_link() -> tf::manifold_edge_link<Index, N> & {
    return *_manifold_edge_link;
  }

private:
  tf::manifold_edge_link<Index, N> *_manifold_edge_link;

  friend auto unwrap(const tag_manifold_edge_link &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_manifold_edge_link &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_manifold_edge_link &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_manifold_edge_link &val, T &&t) {
    return tag_manifold_edge_link<Index, N, std::decay_t<T>>{
        val._manifold_edge_link, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Index, std::size_t N, typename Base>
struct static_size<policy::tag_manifold_edge_link<Index, N, Base>>
    : static_size<Base> {};

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> *_manifold_edge_link, Base &&base) {
  if constexpr (has_manifold_edge_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_manifold_edge_link<Index, N,
                                             std::decay_t<decltype(b_base)>>{
                  _manifold_edge_link, b_base});
  }
}

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link, Base &&base) {
  return tag_manifold_edge_link(&_manifold_edge_link,
                                static_cast<Base &&>(base));
}

namespace policy {
template <typename Index, std::size_t N> struct tag_manifold_edge_link_op {
  tf::manifold_edge_link<Index, N> *manifold_edge_link;
};

template <typename U, typename Index, std::size_t N>
auto operator|(U &&u, tag_manifold_edge_link_op<Index, N> t) {
  return tf::tag_manifold_edge_link(t.manifold_edge_link, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<Index, N>{&_manifold_edge_link};
}

template <typename Index, std::size_t N>
auto tag(tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<Index, N>{&_manifold_edge_link};
}

} // namespace tf
namespace std {
template <typename Index, std::size_t N, typename Base>
struct tuple_size<tf::policy::tag_manifold_edge_link<Index, N, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Index, std::size_t N, typename Base>
struct tuple_element<I, tf::policy::tag_manifold_edge_link<Index, N, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
