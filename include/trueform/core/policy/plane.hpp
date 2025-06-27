/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "../plane.hpp"
#include "./normal.hpp"
#include <utility>

namespace tf {
namespace policy {
template <std::size_t Dims, typename Policy, typename Base>
struct tag_plane : Base {
  /**
   * @brief Constructs an instance by copying the plane and the base.
   *
   * @param _plane The plane to inject.
   * @param base   The base object to extend.
   */
  tag_plane(const tf::plane_like<Dims, Policy> &_plane, const Base &base)
      : Base{base}, _plane{_plane} {}

  /**
   * @brief Constructs an instance by moving the plane and the base.
   *
   * @param _plane The plane to inject (moved).
   * @param base   The base object to extend (moved).
   */
  tag_plane(tf::plane_like<Dims, Policy> &&_plane, Base &&base)
      : Base{std::move(base)}, _plane{std::move(_plane)} {}

  /**
   * @brief Returns a const reference to the injected plane.
   */
  auto plane() const -> const tf::plane_like<Dims, Policy> & { return _plane; }

  /**
   * @brief Returns a const reference to the normal vector of the injected
   * plane.
   */
  auto normal() const -> const typename Policy::normal_type & {
    return _plane.normal;
  }

private:
  tf::plane_like<Dims, Policy> _plane;

  friend auto unwrap(const tag_plane &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_plane &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_plane &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_plane &val, T &&t) {
    return tag_plane<Dims, Policy, std::decay_t<T>>{val._plane,
                                                    static_cast<T &&>(t)};
  }
};

template <std::size_t Dims, typename Policy, typename Base>
auto has_normal(type, const tag_plane<Dims, Policy, Base> *) -> std::true_type;

template <std::size_t Dims, typename Policy, typename Base>
auto has_plane(type, const tag_plane<Dims, Policy, Base> *) -> std::true_type;

auto has_plane(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_plane_policy = decltype(has_plane(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <std::size_t Dims, typename Policy, typename Base>
struct static_size<policy::tag_plane<Dims, Policy, Base>> : static_size<Base> {
};

namespace policy {
template <std::size_t Dims, typename T, typename Base>
auto tag_plane_impl(const plane_like<Dims, T> &plane, Base &&base) {
  auto &b_base = unwrap(base);
  if constexpr (has_normal_policy<Base>) {
    if constexpr (!has_normal_policy<std::decay_t<decltype(b_base)>>) {
      return tag_plane<Dims, T, std::decay_t<decltype(b_base)>>{plane, b_base};
    } else
      return wrap_like(base, policy::tag_plane_impl(plane, b_base));
  } else {
    return wrap_like(base, tag_plane<Dims, T, std::decay_t<decltype(b_base)>>{
                               plane, b_base});
  }
}
} // namespace policy
template <std::size_t Dims, typename T, typename Base>
auto tag_plane(const plane_like<Dims, T> &plane, Base &&base) {
  if constexpr (has_plane_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    return policy::tag_plane_impl(plane, static_cast<Base &&>(base));
  }
}

namespace policy {
template <std::size_t Dims, typename T> struct tag_plane_op {
  plane_like<Dims, T> plane;
};

template <typename U, std::size_t Dims, typename T>
auto operator|(U &&u, tag_plane_op<Dims, T> t) {
  return tf::tag_plane(t.plane, static_cast<U &&>(u));
}
} // namespace policy

template <std::size_t Dims, typename T>
auto tag_plane(plane_like<Dims, T> plane) {
  return policy::tag_plane_op<Dims, T>{std::move(plane)};
}

template <std::size_t V, typename Policy>
auto tag_plane(const polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_plane_policy<Policy>) {
    return poly;
  } else if constexpr (has_normal_policy<Policy>) {
    return tf::tag_plane(
        tf::make_plane_like(poly.normal(), tf::dot(poly[0], poly.normal())),
        poly);
  } else {
    return tf::tag_plane(tf::make_plane(poly[0], poly[1], poly[2]), poly);
  }
}

template <std::size_t V, typename Policy>
auto tag_plane(polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_plane_policy<Policy>) {
    return poly;
  } else if constexpr (has_normal_policy<Policy>) {
    return tf::tag_plane(
        tf::make_plane_like(poly.normal(), tf::dot(poly[0], poly.normal())),
        poly);
  } else {
    return tf::tag_plane(tf::make_plane(poly[0], poly[1], poly[2]), poly);
  }
}
namespace policy {
struct tag_plane_self_op {};

template <typename U> auto operator|(U &&u, tag_plane_self_op) {
  return tf::tag_plane(static_cast<U &&>(u));
}
} // namespace policy
inline auto tag_plane() { return policy::tag_plane_self_op{}; }
} // namespace tf
namespace std {
template <std::size_t Dims, typename Policy, typename Base>
struct tuple_size<tf::policy::tag_plane<Dims, Policy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, std::size_t Dims, typename Policy, typename Base>
struct tuple_element<I, tf::policy::tag_plane<Dims, Policy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
