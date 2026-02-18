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
#include "../half_edges.hpp"
#include "../half_edges_like.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename HEViewPolicy, typename Base> struct tag_half_edges;

template <typename HEViewPolicy, typename Base>
auto has_half_edges(const tag_half_edges<HEViewPolicy, Base> *)
    -> std::true_type;

auto has_half_edges(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_half_edges_policy =
    decltype(policy::has_half_edges(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {

template <typename HEViewPolicy, typename Base>
struct tag_half_edges : Base {
  using Base::operator=;

  tag_half_edges(tf::half_edges_like<HEViewPolicy> _half_edges_view,
                 const Base &base)
      : Base{base}, _half_edges_view{std::move(_half_edges_view)} {}

  tag_half_edges(tf::half_edges_like<HEViewPolicy> _half_edges_view,
                 Base &&base)
      : Base{std::move(base)},
        _half_edges_view{std::move(_half_edges_view)} {}

  auto half_edges() const -> const tf::half_edges_like<HEViewPolicy> & {
    return _half_edges_view;
  }

  auto half_edges() -> tf::half_edges_like<HEViewPolicy> & {
    return _half_edges_view;
  }

private:
  tf::half_edges_like<HEViewPolicy> _half_edges_view;

  friend auto unwrap(const tag_half_edges &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_half_edges &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_half_edges &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_half_edges &val, T &&t) {
    return tag_half_edges<HEViewPolicy, std::decay_t<T>>{
        val._half_edges_view, static_cast<T &&>(t)};
  }
};

} // namespace policy

template <typename HEViewPolicy, typename Base>
struct static_size<policy::tag_half_edges<HEViewPolicy, Base>>
    : static_size<Base> {};

template <typename HEViewPolicy, typename Base>
auto tag_half_edges(tf::half_edges_like<HEViewPolicy> &&_half_edges_view,
                    Base &&base) {
  if constexpr (has_half_edges_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_half_edges<HEViewPolicy,
                               std::decay_t<decltype(b_base)>>{
            std::move(_half_edges_view), b_base});
  }
}

template <typename Index, typename Base>
auto tag_half_edges(tf::half_edges<Index> &_half_edges, Base &&base) {
  return tag_half_edges(tf::make_half_edges_view(_half_edges),
                        static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_half_edges(const tf::half_edges<Index> &_half_edges, Base &&base) {
  return tag_half_edges(tf::make_half_edges_view(_half_edges),
                        static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_half_edges(tf::half_edges<Index> &&_half_edges, Base &&base) = delete;

namespace policy {

template <typename HEViewPolicy> struct tag_half_edges_op {
  tf::half_edges_like<HEViewPolicy> half_edges_view;
};

template <typename U, typename HEViewPolicy>
auto operator|(U &&u, tag_half_edges_op<HEViewPolicy> t) {
  return tf::tag_half_edges(std::move(t.half_edges_view),
                            static_cast<U &&>(u));
}

} // namespace policy

template <typename HEViewPolicy>
auto tag_half_edges(tf::half_edges_like<HEViewPolicy> &&_half_edges_view) {
  return policy::tag_half_edges_op<HEViewPolicy>{
      std::move(_half_edges_view)};
}

template <typename Index>
auto tag_half_edges(tf::half_edges<Index> &_half_edges) {
  return policy::tag_half_edges_op<topology::half_edges_ranges<Index>>{
      tf::make_half_edges_view(_half_edges)};
}

template <typename Index>
auto tag_half_edges(const tf::half_edges<Index> &_half_edges) {
  return policy::tag_half_edges_op<topology::half_edges_ranges<Index>>{
      tf::make_half_edges_view(_half_edges)};
}

template <typename Index>
auto tag_half_edges(tf::half_edges<Index> &&_half_edges) = delete;

template <typename Index> auto tag(tf::half_edges<Index> &_half_edges) {
  return tag_half_edges(_half_edges);
}

template <typename Index>
auto tag(const tf::half_edges<Index> &_half_edges) {
  return tag_half_edges(_half_edges);
}

template <typename Index>
auto tag(tf::half_edges<Index> &&_half_edges) = delete;

template <typename Policy>
auto tag(tf::half_edges_like<Policy> &_half_edges_view) {
  using Index = typename Policy::index_type;
  return policy::tag_half_edges_op<topology::half_edges_ranges<Index>>{
      tf::make_half_edges_view(_half_edges_view)};
}

template <typename Policy>
auto tag(const tf::half_edges_like<Policy> &_half_edges_view) {
  using Index = typename Policy::index_type;
  return policy::tag_half_edges_op<topology::half_edges_ranges<Index>>{
      tf::make_half_edges_view(_half_edges_view)};
}

template <typename Policy>
auto tag(tf::half_edges_like<Policy> &&_half_edges_view) {
  return tag(_half_edges_view);
}

} // namespace tf

namespace std {

template <typename HEViewPolicy, typename Base>
struct tuple_size<tf::policy::tag_half_edges<HEViewPolicy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename HEViewPolicy, typename Base>
struct tuple_element<I, tf::policy::tag_half_edges<HEViewPolicy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};

} // namespace std
