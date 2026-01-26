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
#include "../view.hpp"
#include "../zip_range.hpp"
#include "./state.hpp"
#include "./unwrap.hpp"
#include "./zipped.hpp"
#include <utility>

namespace tf {
namespace policy {
template <typename Range, typename Base> struct tag_states : Base {
  /**
   * @brief Constructs an instance.
   */
  tag_states(const Range &_states, const Base &base)
      : Base{base}, _states{_states} {}

  /**
   * @brief Constructs an instance.
   */
  tag_states(Range &&_states, Base &&base)
      : Base{std::move(base)}, _states{std::move(_states)} {}

  /**
   * @brief Returns a const reference to the states.
   */
  auto states() const -> const Range & { return _states; }

  /**
   * @brief Returns a mutable reference to the states.
   */
  auto states() -> Range & { return _states; }

private:
  Range _states;

  friend auto unwrap(const tag_states &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_states &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_states &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_states &val, T &&t) {
    return tag_states<Range, std::decay_t<T>>{val._states,
                                              static_cast<T &&>(t)};
  }
};

/**
 * @ingroup core_policies
 * @brief Zip states with a range for per-element state access.
 *
 * Zips a states range with a base range so each element has a `.state()` accessor.
 *
 * @tparam Range The states range type.
 * @tparam Base The range being augmented.
 */
template <typename Range, typename Base>
struct zip_states : zipped<Range, policy::state_ptr_tagger, Base> {
private:
  using base_t = zipped<Range, policy::state_ptr_tagger, Base>;

public:
  /**
   * @brief Constructs an instance.
   */
  zip_states(const Range &_states, const Base &base) : base_t{_states, base} {}

  zip_states(Range &&_states, Base &&base)
      : base_t{std::move(_states), std::move(base)} {}

  /**
   * @brief Returns a const reference to the states.
   */
  auto states() const -> const Range & { return base_t::_zipped; }

  /**
   * @brief Returns a mutable reference to the states.
   */
  auto states() -> Range & { return base_t::_zipped; }

  friend auto unwrap(const zip_states &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(zip_states &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(zip_states &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const zip_states &val, T &&t) {
    return zip_states<Range, std::decay_t<T>>{val._zipped,
                                              static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::zip_states<Range, Base>> : static_size<Base> {};

template <typename Range, typename Base>
struct static_size<policy::tag_states<Range, Base>> : static_size<Base> {};

namespace policy {
template <typename Range, typename Base>
auto has_states(type, const tag_states<Range, Base> *) -> std::true_type;

template <typename Range, typename Base>
auto has_states(type, const zip_states<Range, Base> *) -> std::true_type;

auto has_states(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_states_policy = decltype(has_states(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

/// @ingroup core_policies
/// @brief Inject a range of states into a range.
///
/// Adds `.states()` accessor to access the states range.
namespace policy {
template <typename Range, typename Base>
auto tag_states_impl(Range &&states, Base &&base) {
  if constexpr (has_states_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_states<std::decay_t<Range>, std::decay_t<decltype(b_base)>>{
            static_cast<Range &&>(states), b_base});
  }
}

template <typename Range, typename Base>
auto zip_states_impl(Range &&states, Base &&base) {
  if constexpr (has_states_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::zip_states<std::decay_t<Range>, std::decay_t<decltype(b_base)>>{
            static_cast<Range &&>(states), b_base});
  }
}

struct states_zipper {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return zip_states_impl(*iters.first, *iters.second);
  }
};

template <typename T> struct tag_states_op {
  T states;
};

template <typename U, typename T> auto operator|(U &&u, tag_states_op<T> t) {
  return tag_states_impl(t.states, static_cast<U &&>(u));
}

template <typename T> struct zip_states_op {
  T states;
};

template <typename U, typename T> auto operator|(U &&u, zip_states_op<T> t) {
  return zip_states_impl(t.states, static_cast<U &&>(u));
}
} // namespace policy
template <typename Range, typename... Ranges>
auto tag_states(Range &&state, Ranges &&...states) {
  auto r = tf::core::make_zip_range(
      tf::make_view(static_cast<Range &&>(state)),
      tf::make_view(static_cast<Ranges &&>(states))...);
  return policy::tag_states_op<decltype(r)>{std::move(r)};
}

template <typename Range, typename... Ranges>
auto zip_states(Range &&state, Ranges &&...states) {
  auto r = tf::core::make_zip_range(
      tf::make_view(static_cast<Range &&>(state)),
      tf::make_view(static_cast<Ranges &&>(states))...);
  return policy::zip_states_op<decltype(r)>{std::move(r)};
}
} // namespace tf

namespace std {
template <typename Range, typename Policy>
struct tuple_size<tf::policy::tag_states<Range, Policy>> : tuple_size<Policy> {
};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::tag_states<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
template <typename Range, typename Policy>
struct tuple_size<tf::policy::zip_states<Range, Policy>> : tuple_size<Policy> {
};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::zip_states<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
