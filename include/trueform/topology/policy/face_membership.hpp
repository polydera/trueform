/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../face_membership.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Index, typename Base> struct tag_face_membership;
template <typename Index, typename Base>
auto has_face_membership(const tag_face_membership<Index, Base> *)
    -> std::true_type;

auto has_face_membership(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_face_membership_policy =
    decltype(policy::has_face_membership(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Index, typename Base> struct tag_face_membership : Base {
  using Base::operator=;
  tag_face_membership(tf::face_membership<Index> *_face_membership,
                      const Base &base)
      : Base{base}, _face_membership{_face_membership} {}

  tag_face_membership(tf::face_membership<Index> *_face_membership, Base &&base)
      : Base{std::move(base)}, _face_membership{_face_membership} {}

  auto face_membership() const -> const tf::face_membership<Index> & {
    return *_face_membership;
  }

  auto face_membership() -> tf::face_membership<Index> & {
    return *_face_membership;
  }

private:
  tf::face_membership<Index> *_face_membership;

  friend auto unwrap(const tag_face_membership &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_face_membership &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_face_membership &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_face_membership &val, T &&t) {
    return tag_face_membership<Index, std::decay_t<T>>{val._face_membership,
                                                       static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Index, typename Base>
struct static_size<policy::tag_face_membership<Index, Base>>
    : static_size<Base> {};

template <typename Index, typename Base>
auto tag_face_membership(tf::face_membership<Index> *_face_membership,
                         Base &&base) {
  if constexpr (has_face_membership_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_face_membership<Index, std::decay_t<decltype(b_base)>>{
            _face_membership, b_base});
  }
}

template <typename Index, typename Base>
auto tag_face_membership(tf::face_membership<Index> &_face_membership,
                         Base &&base) {
  return tag_face_membership(&_face_membership, static_cast<Base &&>(base));
}

namespace policy {
template <typename Index> struct tag_face_membership_op {
  tf::face_membership<Index> *face_membership;
};

template <typename U, typename Index>
auto operator|(U &&u, tag_face_membership_op<Index> t) {
  return tf::tag_face_membership(t.face_membership, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index>
auto tag_face_membership(tf::face_membership<Index> &_face_membership) {
  return policy::tag_face_membership_op<Index>{&_face_membership};
}

template <typename Index>
auto tag(tf::face_membership<Index> &_face_membership) {
  return policy::tag_face_membership_op<Index>{&_face_membership};
}

} // namespace tf
namespace std {
template <typename Index, typename Base>
struct tuple_size<tf::policy::tag_face_membership<Index, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Index, typename Base>
struct tuple_element<I, tf::policy::tag_face_membership<Index, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
