/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../frame_like.hpp"
#include "./type.hpp"
#include "./unwrap.hpp"

namespace tf {
namespace policy {

template <std::size_t Dims, typename Policy, typename Base> struct tag_frame;
template <std::size_t Dims, typename Policy, typename Base>
auto has_frame(type, const tag_frame<Dims, Policy, Base> *) -> std::true_type;

auto has_frame(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_frame_policy = decltype(has_frame(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;
namespace policy {
template <std::size_t Dims, typename Policy, typename Base>
struct tag_frame : Base {
  using Base::operator=;
  tag_frame(const frame_like<Dims, Policy> &_frame, const Base &base)
      : Base{base}, _frame{_frame} {}

  tag_frame(frame_like<Dims, Policy> &&_frame, Base &&base)
      : Base{std::move(base)}, _frame{std::move(_frame)} {}

  template <typename Other>
  auto operator=(Other &&other)
      -> std::enable_if_t<has_frame_policy<Other> &&
                              std::is_assignable_v<frame_like<Dims, Policy> &,
                                                   decltype(other.frame())> &&
                              std::is_assignable_v<Base &, Other &&>,
                          tag_frame &> {
    Base::operator=(static_cast<Other &&>(other));
    _frame = other.frame();
    return *this;
  }

  /**
   * @brief Returns a const reference to the injected frame.
   */
  auto frame() const -> const frame_like<Dims, Policy> & { return _frame; }

  /**
   * @brief Returns a mutable reference to the injected frame.
   */
  auto frame() -> frame_like<Dims, Policy> & { return _frame; }

  auto transformation() const -> decltype(auto) {
    return _frame.transformation();
  }

  auto transformation() -> decltype(auto) { return _frame.transformation(); }

  auto inverse_transformation() const -> decltype(auto) {
    return _frame.inverse_transformation();
  }

  auto inverse_transformation() -> decltype(auto) {
    return _frame.inverse_transformation();
  }

private:
  frame_like<Dims, Policy> _frame;

  friend auto unwrap(const tag_frame &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_frame &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_frame &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_frame &val, T &&t) {
    return tag_frame<Dims, Policy, std::decay_t<T>>{val._frame,
                                                    static_cast<T &&>(t)};
  }
};
} // namespace policy

template <std::size_t Dims, typename Policy, typename Base>
struct static_size<policy::tag_frame<Dims, Policy, Base>> : static_size<Base> {
};

template <std::size_t Dims, typename T, typename Base>
auto tag_frame(const frame_like<Dims, T> &frame, Base &&base) {
  if constexpr (has_frame_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base,
                     policy::tag_frame<Dims, T, std::decay_t<decltype(b_base)>>{
                         frame, b_base});
  }
}

namespace policy {
template <std::size_t Dims, typename T> struct tag_frame_op {
  frame_like<Dims, T> frame;
};

template <typename U, std::size_t Dims, typename T>
auto operator|(U &&u, tag_frame_op<Dims, T> t) {
  return tf::tag_frame(t.frame, static_cast<U &&>(u));
}
} // namespace policy

template <std::size_t Dims, typename T>
auto tag_frame(frame_like<Dims, T> frame) {
  return policy::tag_frame_op<Dims, T>{std::move(frame)};
}

template <std::size_t Dims, typename T> auto tag(frame_like<Dims, T> frame) {
  return policy::tag_frame_op<Dims, T>{std::move(frame)};
}

} // namespace tf
namespace std {
template <std::size_t Dims, typename Policy, typename Base>
struct tuple_size<tf::policy::tag_frame<Dims, Policy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, std::size_t Dims, typename Policy, typename Base>
struct tuple_element<I, tf::policy::tag_frame<Dims, Policy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
