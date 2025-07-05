/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../face_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, typename Base> struct tag_face_link;
template <typename Index, typename Base>
auto has_face_link(const tag_face_link<Index, Base> *) -> std::true_type;

auto has_face_link(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_face_link_policy = decltype(policy::has_face_link(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, typename Base> struct tag_face_link : Base {
  using Base::operator=;
  tag_face_link(tf::face_link<Index> *_face_link, const Base &base)
      : Base{base}, _face_link{_face_link} {}

  tag_face_link(tf::face_link<Index> *_face_link, Base &&base)
      : Base{std::move(base)}, _face_link{_face_link} {}

  auto face_link() const -> const tf::face_link<Index> & { return *_face_link; }

  auto face_link() -> tf::face_link<Index> & { return *_face_link; }

private:
  tf::face_link<Index> *_face_link;

  friend auto unwrap(const tag_face_link &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_face_link &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_face_link &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_face_link &val, T &&t) {
    return tag_face_link<Index, std::decay_t<T>>{val._face_link,
                                                 static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Index, typename Base>
struct static_size<policy::tag_face_link<Index, Base>> : static_size<Base> {};

template <typename Index, typename Base>
auto tag_face_link(tf::face_link<Index> *_face_link, Base &&base) {
  if constexpr (has_face_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_face_link<Index, std::decay_t<decltype(b_base)>>{
                  _face_link, b_base});
  }
}

template <typename Index, typename Base>
auto tag_face_link(tf::face_link<Index> &_face_link, Base &&base) {
  return tag_face_link(&_face_link, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index> struct tag_face_link_op {
  tf::face_link<Index> *face_link;
};

template <typename U, typename Index>
auto operator|(U &&u, tag_face_link_op<Index> t) {
  return tf::tag_face_link(t.face_link, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index> auto tag_face_link(tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<Index>{&_face_link};
}

template <typename Index> auto tag(tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<Index>{&_face_link};
}

} // namespace tf
namespace std {
template <typename Index, typename Base>
struct tuple_size<tf::policy::tag_face_link<Index, Base>> : tuple_size<Base> {};

template <std::size_t I, typename Index, typename Base>
struct tuple_element<I, tf::policy::tag_face_link<Index, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
