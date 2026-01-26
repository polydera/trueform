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
#include "./normal.hpp"
#include "./unwrap.hpp"
#include "./zipped.hpp"
#include <utility>

namespace tf {
namespace policy {
template <typename Range, typename Base> struct tag_normals : Base {
  /**
   * @brief Constructs an instance.
   */
  tag_normals(const Range &_normals, const Base &base)
      : Base{base}, _normals{_normals} {}

  /**
   * @brief Constructs an instance.
   */
  tag_normals(Range &&_normals, Base &&base)
      : Base{std::move(base)}, _normals{std::move(_normals)} {}

  /**
   * @brief Returns a const reference to the normals.
   */
  auto normals() const -> const Range & { return _normals; }

  /**
   * @brief Returns a mutable reference to the normals.
   */
  auto normals() -> Range & { return _normals; }

private:
  Range _normals;

  friend auto unwrap(const tag_normals &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_normals &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_normals &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_normals &val, T &&t) {
    return tag_normals<Range, std::decay_t<T>>{val._normals,
                                               static_cast<T &&>(t)};
  }
};

/**
 * @ingroup core_policies
 * @brief Zip normals with a range for per-element normal access.
 *
 * Zips a normals range with a base range so each element has a `.normal()` accessor.
 *
 * @tparam Range The normals range type.
 * @tparam Base The range being augmented.
 */
template <typename Range, typename Base>
struct zip_normals : zipped<Range, policy::normal_tagger, Base> {
private:
  using base_t = zipped<Range, policy::normal_tagger, Base>;

public:
  /**
   * @brief Constructs an instance.
   */
  zip_normals(const Range &_normals, const Base &base)
      : base_t{_normals, base} {}

  zip_normals(Range &&_normals, Base &&base)
      : base_t{std::move(_normals), std::move(base)} {}

  /**
   * @brief Returns a const reference to the normals.
   */
  auto normals() const -> const Range & { return base_t::_zipped; }

  /**
   * @brief Returns a mutable reference to the normals.
   */
  auto normals() -> Range & { return base_t::_zipped; }

  friend auto unwrap(const zip_normals &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(zip_normals &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(zip_normals &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const zip_normals &val, T &&t) {
    return zip_normals<Range, std::decay_t<T>>{val._zipped,
                                               static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::zip_normals<Range, Base>> : static_size<Base> {};

template <typename Range, typename Base>
struct static_size<policy::tag_normals<Range, Base>> : static_size<Base> {};

namespace policy {
template <typename Range, typename Base>
auto has_normals(type, const tag_normals<Range, Base> *) -> std::true_type;

template <typename Range, typename Base>
auto has_normals(type, const zip_normals<Range, Base> *) -> std::true_type;

auto has_normals(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_normals_policy = decltype(has_normals(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

/// @ingroup core_policies
/// @brief Inject a range of normals into a range.
///
/// Adds `.normals()` accessor to access the normals range.
template <typename Range, typename Base>
auto tag_normals(Range &&normals, Base &&base) {
  if constexpr (has_normals_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base, policy::tag_normals<std::decay_t<Range>,
                                               std::decay_t<decltype(b_base)>>{
                               static_cast<Range &&>(normals), b_base});
  }
}

template <typename Range, typename Base>
auto zip_normals(Range &&normals, Base &&base) {
  if constexpr (has_normals_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base, policy::zip_normals<std::decay_t<Range>,
                                               std::decay_t<decltype(b_base)>>{
                               static_cast<Range &&>(normals), b_base});
  }
}

namespace policy {
struct normals_zipper {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return tf::zip_normals(*iters.first, *iters.second);
  }
};

template <typename Range> struct tag_normals_op {
  Range normals;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_normals_op<Range> t) {
  return tf::tag_normals(t.normals, static_cast<U &&>(u));
}

template <typename Range> struct zip_normals_op {
  Range normals;
};

template <typename U, typename Range>
auto operator|(U &&u, zip_normals_op<Range> t) {
  return tf::zip_normals(t.normals, static_cast<U &&>(u));
}
} // namespace policy
template <typename Range> auto tag_normals(Range &&normals) {
  return policy::tag_normals_op<std::decay_t<Range>>{
      static_cast<Range &&>(normals)};
}

template <typename Range> auto zip_normals(Range &&normals) {
  return policy::zip_normals_op<std::decay_t<Range>>{
      static_cast<Range &&>(normals)};
}

} // namespace tf
namespace std {
template <typename Range, typename Policy>
struct tuple_size<tf::policy::tag_normals<Range, Policy>> : tuple_size<Policy> {
};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::tag_normals<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
template <typename Range, typename Policy>
struct tuple_size<tf::policy::zip_normals<Range, Policy>> : tuple_size<Policy> {
};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::zip_normals<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
