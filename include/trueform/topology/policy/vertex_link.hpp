/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../vertex_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, typename Base> struct tag_vertex_link;
template <typename Index, typename Base>
auto has_vertex_link(const tag_vertex_link<Index, Base> *) -> std::true_type;

auto has_vertex_link(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_vertex_link_policy = decltype(policy::has_vertex_link(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, typename Base> struct tag_vertex_link : Base {
  using Base::operator=;
  tag_vertex_link(tf::vertex_link<Index> *_vertex_link, const Base &base)
      : Base{base}, _vertex_link{_vertex_link} {}

  tag_vertex_link(tf::vertex_link<Index> *_vertex_link, Base &&base)
      : Base{std::move(base)}, _vertex_link{_vertex_link} {}

  auto vertex_link() const -> const tf::vertex_link<Index> & {
    return *_vertex_link;
  }

  auto vertex_link() -> tf::vertex_link<Index> & { return *_vertex_link; }

private:
  tf::vertex_link<Index> *_vertex_link;

  friend auto unwrap(const tag_vertex_link &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_vertex_link &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_vertex_link &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_vertex_link &val, T &&t) {
    return tag_vertex_link<Index, std::decay_t<T>>{val._vertex_link,
                                                   static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Index, typename Base>
struct static_size<policy::tag_vertex_link<Index, Base>> : static_size<Base> {};

template <typename Index, typename Base>
auto tag_vertex_link(tf::vertex_link<Index> *_vertex_link, Base &&base) {
  if constexpr (has_vertex_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_vertex_link<Index, std::decay_t<decltype(b_base)>>{
                  _vertex_link, b_base});
  }
}

template <typename Index, typename Base>
auto tag_vertex_link(tf::vertex_link<Index> &_vertex_link, Base &&base) {
  return tag_vertex_link(&_vertex_link, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index> struct tag_vertex_link_op {
  tf::vertex_link<Index> *vertex_link;
};

template <typename U, typename Index>
auto operator|(U &&u, tag_vertex_link_op<Index> t) {
  return tf::tag_vertex_link(t.vertex_link, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index>
auto tag_vertex_link(tf::vertex_link<Index> &_vertex_link) {
  return policy::tag_vertex_link_op<Index>{&_vertex_link};
}

template <typename Index> auto tag(tf::vertex_link<Index> &_vertex_link) {
  return policy::tag_vertex_link_op<Index>{&_vertex_link};
}

} // namespace tf
namespace std {
template <typename Index, typename Base>
struct tuple_size<tf::policy::tag_vertex_link<Index, Base>> : tuple_size<Base> {
};

template <std::size_t I, typename Index, typename Base>
struct tuple_element<I, tf::policy::tag_vertex_link<Index, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
