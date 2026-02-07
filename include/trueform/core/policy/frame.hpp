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
#include "../frame.hpp"
#include "../frame_like.hpp"
#include "../linalg/is_identity.hpp"
#include "./none.hpp"
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

/// @ingroup core_policies
/// @brief Tag a primitive with a coordinate frame.
///
/// Injects frame data for coordinate transformations.
/// If primitive already has a non-trivial frame, returns unchanged.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam T The frame policy type.
/// @tparam Base The primitive type.
/// @param frame The frame to inject.
/// @param base The primitive to tag.
/// @return The tagged primitive.
template <std::size_t Dims, typename Policy, typename Base>
auto tag_frame(const frame_like<Dims, Policy> &frame, Base &&base) {
  if constexpr (has_frame_policy<Base> || linalg::is_identity<Policy>) {
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  } else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_frame<Dims, Policy, std::decay_t<decltype(b_base)>>{
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

/// @ingroup core_policies
/// @brief Create frame tag operator for pipe syntax.
/// @overload
template <std::size_t Dims, typename T>
auto tag_frame(frame_like<Dims, T> frame) {
  if constexpr (linalg::is_identity<T>)
    return none;
  else
    return policy::tag_frame_op<Dims, T>{std::move(frame)};
}

/// @ingroup core_policies
/// @brief Tag with a frame (convenience wrapper).
///
/// @tparam Dims The coordinate dimensions.
/// @tparam T The frame policy type.
/// @param frame The frame to tag with.
/// @return Tag operator for pipe syntax.
template <std::size_t Dims, typename T> auto tag(frame_like<Dims, T> frame) {
  if constexpr (linalg::is_identity<T>)
    return none;
  else
    return policy::tag_frame_op<Dims, T>{std::move(frame)};
}

/// @ingroup core_policies
/// @brief Tag with a transformation (wraps in frame).
/// @overload
template <std::size_t Dims, typename T>
auto tag(transformation_like<Dims, T> transformation) {
  if constexpr (linalg::is_identity<T>)
    return none;
  else
    return tf::tag(tf::make_frame(transformation));
}

/// @ingroup core_policies
/// @brief Remove frame policy from a tagged primitive.
///
/// Recursively searches for and removes the frame policy layer while
/// preserving all other policies. If no frame policy exists, returns
/// the input unchanged.
///
/// @tparam T The primitive type.
/// @param t The primitive to untag.
/// @return The primitive with frame policy removed.
template <typename T>
auto untag_frame(T &&t) -> decltype(auto) {
  if constexpr (has_frame_policy<T>) {
    auto &&b_base = unwrap(t);
    if constexpr (!has_frame_policy<std::decay_t<decltype(b_base)>>) {
      // I have frame, base does not → I AM the frame → skip me
      if constexpr (std::is_rvalue_reference_v<T &&>)
        return static_cast<std::decay_t<decltype(b_base)>>(b_base);
      else
        return b_base;
    } else {
      // Frame is deeper → recurse and re-wrap
      return wrap_like(t, untag_frame(b_base));
    }
  } else {
    // No frame anywhere → return as-is
    if constexpr (std::is_rvalue_reference_v<T &&>)
      return static_cast<T>(t);
    else
      return static_cast<T &&>(t);
  }
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
