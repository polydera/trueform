/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../edge_membership.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, typename Base> struct tag_edge_membership;
template <typename Index, typename Base>
auto has_edge_membership(const tag_edge_membership<Index, Base> *)
    -> std::true_type;

auto has_edge_membership(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_edge_membership_policy =
    decltype(policy::has_edge_membership(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, typename Base> struct tag_edge_membership : Base {
  using Base::operator=;
  tag_edge_membership(tf::edge_membership<Index> *_edge_membership,
                      const Base &base)
      : Base{base}, _edge_membership{_edge_membership} {}

  tag_edge_membership(tf::edge_membership<Index> *_edge_membership, Base &&base)
      : Base{std::move(base)}, _edge_membership{_edge_membership} {}

  auto edge_membership() const -> const tf::edge_membership<Index> & {
    return *_edge_membership;
  }

  auto edge_membership() -> tf::edge_membership<Index> & {
    return *_edge_membership;
  }

private:
  tf::edge_membership<Index> *_edge_membership;

  friend auto unwrap(const tag_edge_membership &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_edge_membership &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_edge_membership &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_edge_membership &val, T &&t) {
    return tag_edge_membership<Index, std::decay_t<T>>{val._edge_membership,
                                                       static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Index, typename Base>
struct static_size<policy::tag_edge_membership<Index, Base>>
    : static_size<Base> {};

template <typename Index, typename Base>
auto tag_edge_membership(tf::edge_membership<Index> *_edge_membership,
                         Base &&base) {
  if constexpr (has_edge_membership_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_edge_membership<Index, std::decay_t<decltype(b_base)>>{
            _edge_membership, b_base});
  }
}

template <typename Index, typename Base>
auto tag_edge_membership(tf::edge_membership<Index> &_edge_membership,
                         Base &&base) {
  return tag_edge_membership(&_edge_membership, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index> struct tag_edge_membership_op {
  tf::edge_membership<Index> *edge_membership;
};

template <typename U, typename Index>
auto operator|(U &&u, tag_edge_membership_op<Index> t) {
  return tf::tag_edge_membership(t.edge_membership, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index>
auto tag_edge_membership(tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<Index>{&_edge_membership};
}

template <typename Index>
auto tag(tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<Index>{&_edge_membership};
}

} // namespace tf
namespace std {
template <typename Index, typename Base>
struct tuple_size<tf::policy::tag_edge_membership<Index, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Index, typename Base>
struct tuple_element<I, tf::policy::tag_edge_membership<Index, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
