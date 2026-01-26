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
#include "./id.hpp"
#include "./unwrap.hpp"
#include "./zipped.hpp"
#include <utility>

namespace tf {
namespace policy {
template <typename Range, typename Base> struct tag_ids : Base {
  /**
   * @brief Constructs an instance.
   */
  tag_ids(const Range &_ids, const Base &base) : Base{base}, _ids{_ids} {}

  /**
   * @brief Constructs an instance.
   */
  tag_ids(Range &&_ids, Base &&base)
      : Base{std::move(base)}, _ids{std::move(_ids)} {}

  /**
   * @brief Returns a const reference to the ids.
   */
  auto ids() const -> const Range & { return _ids; }

  /**
   * @brief Returns a mutable reference to the ids.
   */
  auto ids() -> Range & { return _ids; }

private:
  Range _ids;

  friend auto unwrap(const tag_ids &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_ids &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_ids &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_ids &val, T &&t) {
    return tag_ids<Range, std::decay_t<T>>{val._ids, static_cast<T &&>(t)};
  }
};

/**
 * @ingroup core_policies
 * @brief Zip IDs with a range for per-element ID access.
 *
 * Zips an ID range with a base range so each element has an `.id()` accessor.
 *
 * @tparam Range The ID range type.
 * @tparam Base The range being augmented.
 */
template <typename Range, typename Base>
struct zip_ids : zipped<Range, policy::id_iter_tagger, Base> {
private:
  using base_t = zipped<Range, policy::id_iter_tagger, Base>;

public:
  /**
   * @brief Constructs an instance.
   */
  zip_ids(const Range &_ids, const Base &base) : base_t{_ids, base} {}

  zip_ids(Range &&_ids, Base &&base)
      : base_t{std::move(_ids), std::move(base)} {}

  /**
   * @brief Returns a const reference to the ids.
   */
  auto ids() const -> const Range & { return base_t::_zipped; }

  /**
   * @brief Returns a mutable reference to the ids.
   */
  auto ids() -> Range & { return base_t::_zipped; }

  friend auto unwrap(const zip_ids &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(zip_ids &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(zip_ids &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const zip_ids &val, T &&t) {
    return zip_ids<Range, std::decay_t<T>>{val._zipped, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::zip_ids<Range, Base>> : static_size<Base> {};

template <typename Range, typename Base>
struct static_size<policy::tag_ids<Range, Base>> : static_size<Base> {};

namespace policy {
template <typename Range, typename Base>
auto has_ids(type, const tag_ids<Range, Base> *) -> std::true_type;

template <typename Range, typename Base>
auto has_ids(type, const zip_ids<Range, Base> *) -> std::true_type;

auto has_ids(type, const void *) -> std::false_type;

template <typename Range, typename Base>
auto has_ids_zip_impl(type, const zip_ids<Range, Base> *) -> std::true_type;

auto has_ids_zip_impl(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_ids_tag = decltype(has_ids(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <typename T>
inline constexpr bool has_ids_zip = decltype(has_ids_zip_impl(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;

/// @ingroup core_policies
/// @brief Inject a range of IDs into a range.
///
/// Adds `.ids()` accessor to access the ID range.
template <typename Range, typename Base>
auto tag_ids(Range &&ids, Base &&base) {
  if constexpr (has_ids_tag<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_ids<std::decay_t<Range>, std::decay_t<decltype(b_base)>>{
            static_cast<Range &&>(ids), b_base});
  }
}

template <typename Range, typename Base>
auto zip_ids(Range &&ids, Base &&base) {
  if constexpr (has_ids_tag<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::zip_ids<std::decay_t<Range>, std::decay_t<decltype(b_base)>>{
            static_cast<Range &&>(ids), b_base});
  }
}

namespace policy {
struct ids_zipper {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return tf::zip_ids(*iters.first, *iters.second);
  }
};

template <typename T> struct tag_ids_op {
  T ids;
};

template <typename U, typename T> auto operator|(U &&u, tag_ids_op<T> t) {
  return tf::tag_ids(std::move(t.ids), static_cast<U &&>(u));
}

template <typename T> struct zip_ids_op {
  T ids;
};

template <typename U, typename T> auto operator|(U &&u, zip_ids_op<T> t) {
  return tf::zip_ids(std::move(t.ids), static_cast<U &&>(u));
}
} // namespace policy
template <typename Range> auto tag_ids(Range &&ids) {
  return policy::tag_ids_op<std::decay_t<Range>>{static_cast<Range &&>(ids)};
}

template <typename Range> auto zip_ids(Range &&ids) {
  return policy::zip_ids_op<std::decay_t<Range>>{static_cast<Range &&>(ids)};
}
} // namespace tf

namespace std {
template <typename Range, typename Policy>
struct tuple_size<tf::policy::tag_ids<Range, Policy>> : tuple_size<Policy> {};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::tag_ids<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
template <typename Range, typename Policy>
struct tuple_size<tf::policy::zip_ids<Range, Policy>> : tuple_size<Policy> {};

template <std::size_t I, typename Range, typename Policy>
struct tuple_element<I, tf::policy::zip_ids<Range, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
