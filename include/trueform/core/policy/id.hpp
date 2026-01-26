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
#include "../proxy_val.hpp"
#include "../static_size.hpp"
#include "./type.hpp"
#include "./unwrap.hpp"
#include <type_traits>
#include <utility>

namespace tf {

namespace policy {
template <typename Index, typename Base> struct tag_id;
template <typename Index, typename Base> struct tag_id_iter;

template <typename Range, typename Base>
auto has_id(type, const tag_id<Range, Base> *) -> std::true_type;
template <typename Range, typename Base>
auto has_id(type, const tag_id_iter<Range, Base> *) -> std::true_type;

auto has_id(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_id_policy = decltype(has_id(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;
namespace policy {
/**
 * @ingroup core_policies
 * @brief Type injector that tags an ID to a class.
 *
 * Provides `.id()` accessors for the injected identifier.
 *
 * @tparam Index The ID type.
 * @tparam Base The type being augmented.
 */
template <typename Index, typename Base> struct tag_id : Base {

  using Base::operator=;
  /**
   * @brief Constructs an instance.
   */
  tag_id(const Index &_id, const Base &base) : Base{base}, _id{_id} {}

  /**
   * @brief Constructs an instance.
   */
  tag_id(Index &&_id, Base &&base)
      : Base{std::move(base)}, _id{std::move(_id)} {}

  template <typename Other>
  auto operator=(Other &&other) -> std::enable_if_t<
      has_id_policy<Other> &&
          std::is_assignable_v<Index &, decltype(other.id())> &&
          std::is_assignable_v<Base &, Other &&>,
      tag_id &> {
    Base::operator=(static_cast<Other &&>(other));
    id() = other.id();
    return *this;
  }

  /**
   * @brief Returns a const reference to the id.
   */
  auto id() const -> const Index & { return _id; }

  /**
   * @brief Returns a mutable reference to the id.
   */
  auto id() -> Index & { return _id; }

private:
  Index _id;

  friend auto unwrap(const tag_id &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_id &val) -> Base & { return static_cast<Base &>(val); }

  friend auto unwrap(tag_id &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_id &val, T &&t) {
    return tag_id<Index, std::decay_t<T>>{val.id(), static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_id<Range, Base>> : static_size<Base> {};

/// @ingroup core_policies
/// @brief Inject an ID into a primitive.
///
/// Returns a wrapper with `.id()` accessor while preserving all
/// original functionality through inheritance.
template <typename Index, typename Base> auto tag_id(Index &&id, Base &&base) {
  if constexpr (has_id_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_id<std::decay_t<Index>, std::decay_t<decltype(b_base)>>{
            static_cast<Index &&>(id), b_base});
  }
}

namespace policy {

template <typename Iterator, typename Base> struct tag_id_iter : Base {

  using Base::operator=;
  /**
   * @brief Constructs an instance.
   */
  tag_id_iter(const tf::proxy_val<Iterator> &_id, const Base &base)
      : Base{base}, _id{_id} {}

  /**
   * @brief Constructs an instance.
   */
  tag_id_iter(tf::proxy_val<Iterator> &&_id, Base &&base)
      : Base{std::move(base)}, _id{std::move(_id)} {}

  template <typename Other>
  auto operator=(Other &&other) -> std::enable_if_t<
      has_id_policy<Other> &&
          std::is_assignable_v<decltype(*std::declval<Iterator>()),
                               decltype(other.id())> &&
          std::is_assignable_v<Base &, Other &&>,
      tag_id_iter &> {
    Base::operator=(static_cast<Other &&>(other));
    id() = other.id();
    return *this;
  }

  /**
   * @brief Returns a const reference to the id.
   */
  auto id() const -> decltype(auto) { return *_id; }

  /**
   * @brief Returns a mutable reference to the id.
   */
  auto id() -> decltype(auto) { return *_id; }

private:
  tf::proxy_val<Iterator> _id;

  friend auto unwrap(const tag_id_iter &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_id_iter &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_id_iter &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_id_iter &val, T &&t) {
    return tag_id_iter<Iterator, std::decay_t<T>>{val._id,
                                                  static_cast<T &&>(t)};
  }
};
} // namespace policy
template <typename Range, typename Base>
struct static_size<policy::tag_id_iter<Range, Base>> : static_size<Base> {};

/// @ingroup core_policies
/// @brief Inject an iterator-backed ID into a primitive.
template <typename Iterator, typename Base>
auto tag_id_iter(Iterator &&id, Base &&base) {
  if constexpr (has_id_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base, policy::tag_id_iter<std::decay_t<Iterator>,
                                               std::decay_t<decltype(b_base)>>{
                               static_cast<Iterator &&>(id), b_base});
  }
}

namespace policy {
struct id_iter_tagger {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return tf::tag_id_iter(iters.first, *iters.second);
  }
};

template <typename T> struct tag_id_op {
  T id;
};

template <typename U, typename T> auto operator|(U &&u, tag_id_op<T> t) {
  return tf::tag_id(t.id, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index> auto tag_id(Index &&id) {
  return policy::tag_id_op<std::decay_t<Index>>{static_cast<Index &&>(id)};
}
} // namespace tf
namespace std {
template <typename Index, typename Policy>
struct tuple_size<tf::policy::tag_id<Index, Policy>> : tuple_size<Policy> {};

template <std::size_t I, typename Index, typename Policy>
struct tuple_element<I, tf::policy::tag_id<Index, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
