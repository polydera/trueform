/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

/**
 * @ingroup injectors
 * @brief Constructs an `tag_normal` by injecting a normal into a base
 * type.
 *
 * This helper function provides a convenient way to inject a unit normal into
 * an object, returning a composed type with added `.normal()` accessors.
 *
 * @tparam T    Underlying scalar type of the normal vector.
 * @tparam Dims Dimensionality of the normal vector.
 * @tparam Base Type of the base object (will be decayed).
 * @param normal The normal vector to inject.
 * @param base   The base object to augment.
 * @return A composed object with normal support.
 */
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

template <std::size_t Dims, typename T>
auto tag_normal(unit_vector_like<Dims, T> normal) {
  return policy::tag_normal_op<Dims, T>{std::move(normal)};
}

template <std::size_t V, typename Policy>
auto tag_normal(const polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_normal_policy<Policy>) {
    return poly;
  } else {
    return tf::tag_normal(tf::make_normal(poly[0], poly[1], poly[2]), poly);
  }
}

template <std::size_t V, typename Policy>
auto tag_normal(polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_normal_policy<Policy>) {
    return poly;
  } else {
    return tf::tag_normal(tf::make_normal(poly[0], poly[1], poly[2]), poly);
  }
}

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
