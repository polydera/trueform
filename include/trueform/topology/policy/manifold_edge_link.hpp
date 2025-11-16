/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../manifold_edge_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_manifold_edge_link;
template <typename Range, typename Base>
auto has_manifold_edge_link(const tag_manifold_edge_link<Range, Base> *)
    -> std::true_type;

auto has_manifold_edge_link(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_manifold_edge_link_policy =
    decltype(policy::has_manifold_edge_link(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_manifold_edge_link : Base {
  using Base::operator=;

  tag_manifold_edge_link(
      tf::manifold_edge_link_like<Range> _manifold_edge_link_range,
      const Base &base)
      : Base{base},
        _manifold_edge_link_range{std::move(_manifold_edge_link_range)} {}

  tag_manifold_edge_link(
      tf::manifold_edge_link_like<Range> _manifold_edge_link_range, Base &&base)
      : Base{std::move(base)},
        _manifold_edge_link_range{std::move(_manifold_edge_link_range)} {}

  auto manifold_edge_link() const
      -> const tf::manifold_edge_link_like<Range> & {
    return _manifold_edge_link_range;
  }

  auto manifold_edge_link() -> tf::manifold_edge_link_like<Range> & {
    return _manifold_edge_link_range;
  }

private:
  tf::manifold_edge_link_like<Range> _manifold_edge_link_range;

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
    return tag_manifold_edge_link<Range, std::decay_t<T>>{
        val._manifold_edge_link_range, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_manifold_edge_link<Range, Base>>
    : static_size<Base> {};

template <typename Range, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link_like<Range> &&_manifold_edge_link_range,
    Base &&base) {
  if constexpr (has_manifold_edge_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_manifold_edge_link<Range, std::decay_t<decltype(b_base)>>{
            std::move(_manifold_edge_link_range), b_base});
  }
}

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link, Base &&base) {
  return tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(tf::make_range(_manifold_edge_link)),
      static_cast<Base &&>(base));
}

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    const tf::manifold_edge_link<Index, N> &_manifold_edge_link, Base &&base) {
  return tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(tf::make_range(_manifold_edge_link)),
      static_cast<Base &&>(base));
}

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &&_manifold_edge_link,
    Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_manifold_edge_link_op {
  Range manifold_edge_link_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_manifold_edge_link_op<Range> t) {
  return tf::tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(t.manifold_edge_link_range),
      static_cast<U &&>(u));
}
} // namespace policy

template <typename Range>
auto tag_manifold_edge_link(Range &&_manifold_edge_link_range) {
  return policy::tag_manifold_edge_link_op<Range>{
      static_cast<Range &&>(_manifold_edge_link_range)};
}

template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    const tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &&_manifold_edge_link) = delete;

template <typename Index, std::size_t N>
auto tag(tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag(const tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag(tf::manifold_edge_link<Index, N> &&_manifold_edge_link) = delete;

template <typename Policy>
auto tag(tf::manifold_edge_link_like<Policy> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Policy>
auto tag(const tf::manifold_edge_link_like<Policy> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Policy>
auto tag(tf::manifold_edge_link_like<Policy> &&_manifold_edge_link) {
  return tag(_manifold_edge_link);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_manifold_edge_link<Range, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_manifold_edge_link<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
