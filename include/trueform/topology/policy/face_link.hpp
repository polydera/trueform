/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../../core/policy/unwrap.hpp"
#include "../face_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_face_link;
template <typename Range, typename Base>
auto has_face_link(const tag_face_link<Range, Base> *) -> std::true_type;

auto has_face_link(const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_face_link_policy = decltype(policy::has_face_link(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_face_link : Base {
  using Base::operator=;

  tag_face_link(tf::face_link_like<Range> _face_link_range, const Base &base)
      : Base{base}, _face_link_range{std::move(_face_link_range)} {}

  tag_face_link(tf::face_link_like<Range> _face_link_range, Base &&base)
      : Base{std::move(base)}, _face_link_range{std::move(_face_link_range)} {}

  auto face_link() const -> const tf::face_link_like<Range> & {
    return _face_link_range;
  }

  auto face_link() -> tf::face_link_like<Range> & { return _face_link_range; }

private:
  tf::face_link_like<Range> _face_link_range;

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
    return tag_face_link<Range, std::decay_t<T>>{val._face_link_range,
                                                 static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_face_link<Range, Base>> : static_size<Base> {};

template <typename Range, typename Base>
auto tag_face_link(tf::face_link_like<Range> &&_face_link_range, Base &&base) {
  if constexpr (has_face_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_face_link<Range, std::decay_t<decltype(b_base)>>{
                  std::move(_face_link_range), b_base});
  }
}

template <typename Index, typename Base>
auto tag_face_link(tf::face_link<Index> &_face_link, Base &&base) {
  return tag_face_link(tf::make_face_link_like(tf::make_range(_face_link)),
                       static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_face_link(const tf::face_link<Index> &_face_link, Base &&base) {
  return tag_face_link(tf::make_face_link_like(tf::make_range(_face_link)),
                       static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_face_link(tf::face_link<Index> &&_face_link, Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_face_link_op {
  Range face_link_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_face_link_op<Range> t) {
  return tf::tag_face_link(tf::make_face_link_like(t.face_link_range),
                           static_cast<U &&>(u));
}
} // namespace policy

template <typename Range> auto tag_face_link(Range &&_face_link_range) {
  return policy::tag_face_link_op<Range>{
      static_cast<Range &&>(_face_link_range)};
}

template <typename Index> auto tag_face_link(tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Index>
auto tag_face_link(const tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Index>
auto tag_face_link(tf::face_link<Index> &&_face_link) = delete;

template <typename Index> auto tag(tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Index> auto tag(const tf::face_link<Index> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Index> auto tag(tf::face_link<Index> &&_face_link) = delete;

template <typename Policy> auto tag(tf::face_link_like<Policy> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Policy>
auto tag(const tf::face_link_like<Policy> &_face_link) {
  return policy::tag_face_link_op<decltype(tf::make_range(_face_link))>{
      tf::make_range(_face_link)};
}

template <typename Policy> auto tag(tf::face_link_like<Policy> &&_face_link) {
  return tag(_face_link);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_face_link<Range, Base>> : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_face_link<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std

