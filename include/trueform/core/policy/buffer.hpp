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
#include "../buffer.hpp"
#include "../local_buffer.hpp"
#include "../none.hpp"
#include "../polygon.hpp"
#include "../static_size.hpp"
#include "./none.hpp" // IWYU pragma: export
#include "./type.hpp"
#include "./unwrap.hpp"
#include <type_traits>
#include <utility>

namespace tf {

namespace policy {
template <typename T, typename Base> struct tag_buffer;

template <typename T, typename Base>
auto has_buffer(type, const tag_buffer<T, Base> *) -> std::true_type;

auto has_buffer(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_buffer_policy = decltype(has_buffer(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {

template <typename T, typename Base> struct tag_buffer : Base {
  using Base::operator=;

  tag_buffer(tf::buffer<T> &buf, const Base &base)
      : Base{base}, _buffer{&buf} {}
  tag_buffer(tf::buffer<T> &buf, Base &&base)
      : Base{std::move(base)}, _buffer{&buf} {}

  auto buffer() const -> tf::buffer<T> & { return *_buffer; }

private:
  tf::buffer<T> *_buffer;

  friend auto unwrap(const tag_buffer &val) -> const Base & {
    return static_cast<const Base &>(val);
  }
  friend auto unwrap(tag_buffer &val) -> Base & {
    return static_cast<Base &>(val);
  }
  friend auto unwrap(tag_buffer &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename U> friend auto wrap_like(const tag_buffer &val, U &&u) {
    return tag_buffer<T, std::decay_t<U>>{*val._buffer, static_cast<U &&>(u)};
  }
};

} // namespace policy

template <typename T, typename Base>
struct static_size<policy::tag_buffer<T, Base>> : static_size<Base> {};

/// @ingroup core_policies
/// @brief Tag a polygon with a scratch buffer.
///
/// Injects a buffer for algorithms that need temporary workspace.
///
/// @tparam T The buffer element type.
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy The polygon's policy type.
/// @param buf The buffer to inject.
/// @param poly The polygon to tag.
/// @return The tagged polygon.
template <typename T, std::size_t Dims, typename Policy>
auto tag_buffer(tf::buffer<T> &buf, const polygon<Dims, Policy> &poly) {
  return wrap_map(poly, [&buf](const auto &core_p) {
    using core_t = std::decay_t<decltype(core_p)>;
    return policy::tag_buffer<T, core_t>{buf, core_p};
  });
}

/// @ingroup core_policies
/// @brief Tag a polygon with a local buffer.
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto tag_buffer(tf::local_buffer<T> &buf, const polygon<Dims, Policy> &poly) {
  return tag_buffer(*buf, poly);
}

template <std::size_t Dims, typename Policy>
auto tag_buffer(tf::none_t, const polygon<Dims, Policy> &poly)
    -> const polygon<Dims, Policy> & {
  return poly;
}

namespace policy {
template <typename T> struct tag_buffer_op {
  tf::buffer<T> *buffer;
};

template <typename U, typename T> auto operator|(U &&u, tag_buffer_op<T> op) {
  return tf::tag_buffer(*op.buffer, static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup core_policies
/// @brief Create buffer tag operator for pipe syntax.
///
/// @tparam T The buffer element type.
/// @param buf The buffer to tag with.
/// @return Tag operator for use with pipe (|).
template <typename T> auto tag(tf::buffer<T> &buf) {
  return policy::tag_buffer_op<T>{&buf};
}

/// @ingroup core_policies
/// @brief Create local buffer tag operator for pipe syntax.
/// @overload
template <typename T> auto tag(tf::local_buffer<T> &buf) {
  return policy::tag_buffer_op<T>{&*buf};
}
} // namespace tf

namespace std {
template <typename T, typename Base>
struct tuple_size<tf::policy::tag_buffer<T, Base>> : tuple_size<Base> {};

template <std::size_t I, typename T, typename Base>
struct tuple_element<I, tf::policy::tag_buffer<T, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
