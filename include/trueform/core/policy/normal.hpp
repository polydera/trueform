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
#include "../normal.hpp"
#include "../polygon.hpp"
#include "../unit_vector_like.hpp"
#include "./type.hpp"
#include "./unwrap.hpp"
#include <utility>

namespace tf {
namespace policy {

template <std::size_t Dims, typename Policy, typename Base> struct tag_normal;
template <std::size_t Dims, typename Policy, typename Base>
auto has_normal(type, const tag_normal<Dims, Policy, Base> *) -> std::true_type;

auto has_normal(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_normal_policy = decltype(has_normal(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;
namespace policy {
template <std::size_t Dims, typename Policy, typename Base>
struct tag_normal : Base {
  using Base::operator=;
  tag_normal(const unit_vector_like<Dims, Policy> &_normal, const Base &base)
      : Base{base}, _normal{_normal} {}

  tag_normal(unit_vector_like<Dims, Policy> &&_normal, Base &&base)
      : Base{std::move(base)}, _normal{std::move(_normal)} {}

  template <typename Other>
  auto operator=(Other &&other) -> std::enable_if_t<
      has_normal_policy<Other> &&
          std::is_assignable_v<unit_vector_like<Dims, Policy> &,
                               decltype(other.normal())> &&
          std::is_assignable_v<Base &, Other &&>,
      tag_normal &> {
    Base::operator=(static_cast<Other &&>(other));
    _normal = other.normal();
    return *this;
  }

  /**
   * @brief Returns a const reference to the injected normal.
   */
  auto normal() const -> const unit_vector_like<Dims, Policy> & {
    return _normal;
  }

  /**
   * @brief Returns a mutable reference to the injected normal.
   */
  auto normal() -> unit_vector_like<Dims, Policy> & { return _normal; }

private:
  unit_vector_like<Dims, Policy> _normal;

  friend auto unwrap(const tag_normal &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_normal &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_normal &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_normal &val, T &&t) {
    return tag_normal<Dims, Policy, std::decay_t<T>>{val._normal,
                                                     static_cast<T &&>(t)};
  }
};
} // namespace policy

template <std::size_t Dims, typename Policy, typename Base>
struct static_size<policy::tag_normal<Dims, Policy, Base>> : static_size<Base> {
};

/// @ingroup core_policies
/// @brief Inject a unit normal into a primitive.
///
/// Returns a wrapper with `.normal()` accessor while preserving all
/// original functionality through inheritance.
///
/// @tparam Dims Dimensionality of the normal vector.
/// @tparam T    Underlying scalar type of the normal vector.
/// @tparam Base Type of the base object (will be decayed).
/// @param normal The unit normal vector to inject.
/// @param base   The base object to augment.
/// @return A composed object with `.normal()` accessor.
template <std::size_t Dims, typename T, typename Base>
auto tag_normal(const unit_vector_like<Dims, T> &normal, Base &&base) {
  if constexpr (has_normal_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_normal<Dims, T, std::decay_t<decltype(b_base)>>{
                  normal, b_base});
  }
}

namespace policy {
struct normal_tagger {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return tf::tag_normal(*iters.first, *iters.second);
  }
};

template <std::size_t Dims, typename T> struct tag_normal_op {
  unit_vector_like<Dims, T> normal;
};

template <typename U, std::size_t Dims, typename T>
auto operator|(U &&u, tag_normal_op<Dims, T> t) {
  return tf::tag_normal(t.normal, static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup core_policies
/// @brief Create normal tag operator for pipe syntax.
/// @overload
template <std::size_t Dims, typename T>
auto tag_normal(unit_vector_like<Dims, T> normal) {
  return policy::tag_normal_op<Dims, T>{std::move(normal)};
}

/// @ingroup core_policies
/// @brief Compute and tag normal from polygon vertices.
///
/// If polygon already has a normal, returns as-is.
/// Otherwise computes normal from first three vertices.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy The polygon's policy type.
/// @param poly The polygon to tag.
/// @return The polygon with normal data.
template <std::size_t Dims, typename Policy>
auto tag_normal(const polygon<Dims, Policy> &poly) -> decltype(auto) {
  if constexpr (has_normal_policy<Policy>) {
    return poly;
  } else {
    return tf::tag_normal(tf::make_normal(poly[0], poly[1], poly[2]), poly);
  }
}

/// @ingroup core_policies
/// @brief Compute and tag normal from polygon (mutable version).
/// @overload
template <std::size_t Dims, typename Policy>
auto tag_normal(polygon<Dims, Policy> &poly) -> decltype(auto) {
  if constexpr (has_normal_policy<Policy>) {
    return poly;
  } else {
    return tf::tag_normal(tf::make_normal(poly[0], poly[1], poly[2]), poly);
  }
}

/// @ingroup core_properties
/// @brief Extract or compute normal from polygon.
///
/// Returns the tagged normal if present, otherwise computes
/// from first three vertices.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy The polygon's policy type.
/// @param p The polygon.
/// @return The unit normal vector.
template <std::size_t Dims, typename Policy>
auto make_normal(const polygon<Dims, Policy> &p)
    -> tf::unit_vector<tf::coordinate_type<Policy>, Dims> {
  if constexpr (tf::has_normal_policy<Policy>) {
    return p.normal();
  } else
    return tf::make_normal(p[0], p[1], p[2]);
}

namespace policy {
struct tag_normal_self_op {};

template <typename U> auto operator|(U &&u, tag_normal_self_op) {
return tf::tag_normal(static_cast<U &&>(u));
}
} // namespace policy
/// @ingroup core_policies
/// @brief Create self-tagging normal operator for pipe syntax.
///
/// Used as `polygon | tag_normal()` to compute and tag normal.
///
/// @return Tag operator for use with pipe (|).
inline auto tag_normal() { return policy::tag_normal_self_op{}; }

} // namespace tf
namespace std {
template <std::size_t Dims, typename Policy, typename Base>
struct tuple_size<tf::policy::tag_normal<Dims, Policy, Base>>
  : tuple_size<Base> {};

template <std::size_t I, std::size_t Dims, typename Policy, typename Base>
struct tuple_element<I, tf::policy::tag_normal<Dims, Policy, Base>> {
using type = typename std::iterator_traits<
    decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
