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

#include "../assignable_range.hpp"
#include "../coordinate_type.hpp"
#include "../coordinate_dims.hpp"
#include "../static_size.hpp"
#include "../views/indirect_range.hpp"
namespace tf::core {
template <typename Policy>
class seg : public tf::core::assignable_range<2, Policy> {
private:
  using base_t = tf::core::assignable_range<2, Policy>;

public:
  seg(const Policy &policy) : base_t{policy} {}
  seg(Policy &&policy) : base_t{std::move(policy)} {}
  seg() = default;
  seg(const seg &) = default;
  seg(seg &&) = default;
  auto operator=(const seg &) -> seg & = default;
  auto operator=(seg &&) -> seg & = default;
  using coordinate_type = tf::coordinate_type<typename Policy::value_type>;
  using coordinate_dims = tf::coordinate_dims<typename Policy::value_type>;
  using base_t::base_t;
  using base_t::operator=;

  /**
   * @brief Indexes into the points of the segment
   *
   * @param i Index
   * @return Point at index `i`
   */
  auto operator[](std::size_t i) -> decltype(auto) {
    return base_t::operator[](i);
  }
  /**
   * @brief Indexes into the points of the segment
   *
   * @param i Index
   * @return Point at index `i`
   */
  auto operator[](std::size_t n) const -> decltype(auto) {
    return base_t::operator[](n);
  }

  /**
   * @brief Returns the iterator to the begining of
   * the point range
   *
   * @return An iterator
   */
  auto begin() const { return base_t::begin(); }
  /**
   * @brief Returns the iterator to the begining of
   * the point range
   *
   * @return An iterator
   */
  auto begin() { return base_t::begin(); }
  /**
   * @brief Returns the iterator to the ending of
   * the point range
   *
   * @return An iterator
   */
  auto end() const { return base_t::end(); }
  /**
   * @brief Returns the iterator to the ending of
   * the point range
   *
   * @return An iterator
   */
  auto end() { return base_t::end(); }
  /**
   * @brief Returns the number of points
   *
   * @return Number of points
   */
  constexpr auto size() const { return 2; }
};

template <typename Base> auto is_seg_impl(const seg<Base> *) -> std::true_type;
auto is_seg_impl(const void *) -> std::false_type;

template <typename T>
inline constexpr bool is_seg =
    decltype(is_seg_impl(static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <typename Policy> auto make_seg(Policy &&policy) {
  if constexpr (is_seg<Policy>)
    return policy;
  else
    return core::seg<std::decay_t<Policy>>{static_cast<Policy &&>(policy)};
}

template <typename Range0, typename Range1>
auto make_seg(Range0 &&ids, Range1 &&data) {
  return views::make_indirect_range(make_seg(views::make_indirect_range_base(
      static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data))));
}
} // namespace tf::core

namespace tf {
template <typename Policy>
struct static_size<tf::core::seg<Policy>>
    : std::integral_constant<std::size_t, 2> {};
} // namespace tf
namespace std {
template <typename Policy>
struct tuple_size<tf::core::seg<Policy>> : tuple_size<Policy> {};
template <std::size_t I, typename Policy>
struct tuple_element<I, tf::core::seg<Policy>> : tuple_element<I, Policy> {};
} // namespace std
