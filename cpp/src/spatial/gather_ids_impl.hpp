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

#include "primitive_dispatch.hpp"
#include "trueform/cpp/core/detail/non_deduced.hpp"
#include "trueform/cpp/spatial/gather_ids.hpp"
#include "trueform/cpp/spatial/gather_ids_within_distance.hpp"

#include "trueform/core/aabb_from.hpp"
#include "trueform/core/distance.hpp"
#include "trueform/core/intersects.hpp"
#include "trueform/core/sphere.hpp"
#include "trueform/spatial/gather_ids.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace detail {

template <typename T, typename = void>
struct gather_has_aabb_from : std::false_type {};

template <typename T>
struct gather_has_aabb_from<
    T, std::void_t<decltype(tf::aabb_from(std::declval<const T &>()))>>
    : std::true_type {};

template <typename Real> auto require_gather_distance(Real distance) -> Real {
  if (!(distance >= Real{0}))
    throw std::invalid_argument(
        "gather_ids_within_distance: distance must be non-negative");
  return distance * distance;
}

inline auto checked_gather_count(std::size_t count) -> int {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("gather_ids: result exceeds ndarray shape range");
  return static_cast<int>(count);
}

template <typename Index>
auto gather_ids_array(tf::buffer<Index> &&ids) -> nd_array<Index> {
  const auto count = checked_gather_count(ids.size());
  return nd_array<Index>::from_buffer(std::move(ids), {count});
}

template <typename OutputIndex, typename Index0, typename Index1>
auto gather_pair_array(const std::vector<std::pair<Index0, Index1>> &pairs)
    -> nd_array<OutputIndex> {
  const auto count = checked_gather_count(pairs.size());
  tf::buffer<OutputIndex> output;
  output.allocate(pairs.size() * 2);
  auto *value = output.data();
  for (const auto &pair : pairs) {
    *value++ = static_cast<OutputIndex>(pair.first);
    *value++ = static_cast<OutputIndex>(pair.second);
  }
  return nd_array<OutputIndex>::from_buffer(std::move(output), {count, 2});
}

/// The id a carrier names its primitives with. A cloud names points, and its
/// ids are the cloud's own.
template <typename Form> struct gather_form_index {
  using type = typename Form::index_type;
};
template <typename Real, std::size_t Dims>
struct gather_form_index<cpp::point_cloud<Real, Dims>> {
  using type = std::int32_t;
};

template <typename Form>
using gather_form_index_t = typename gather_form_index<Form>::type;

template <bool WithinDistance, typename Form, typename Real, std::size_t Dims>
auto form_primitive_gather(const Form &form, const primitive<Real, Dims> &query,
                           Real distance = Real{})
    -> nd_array<gather_form_index_t<Form>> {
  using Index = gather_form_index_t<Form>;
  require_spatial(query.kind());
  if (query.is_batch())
    throw std::invalid_argument("gather_ids: primitive query must be scalar");

  const auto threshold2 = [&] {
    if constexpr (WithinDistance)
      return require_gather_distance(distance);
    else
      return Real{};
  }();

  tf::buffer<Index> output;
  const auto values = query.data();
  const auto *data = values.raw_data();
  const auto kind = query.kind();
  const auto polygon_vertices = query.polygon_vertex_count();

  const auto native_form = form.form();
  visit_spatial_primitive<Dims>(
      [&](const auto &primitive_query) {
        using query_type = std::decay_t<decltype(primitive_query)>;
        if constexpr (gather_has_aabb_from<query_type>::value) {
          const auto query_aabb = tf::aabb_from(primitive_query);
          const auto aabb_predicate = [&](const auto &aabb) {
            if constexpr (WithinDistance)
              return tf::distance2(aabb, query_aabb) <= threshold2;
            else
              return tf::intersects(aabb, query_aabb);
          };
          const auto primitive_predicate = [&](const auto &value) {
            if constexpr (WithinDistance)
              return tf::distance2(value, primitive_query) <= threshold2;
            else
              return tf::intersects(value, primitive_query);
          };
          tf::gather_ids(native_form, aabb_predicate, primitive_predicate,
                         std::back_inserter(output));
        } else {
          const auto aabb_predicate = [&](const auto &aabb) {
            if constexpr (WithinDistance) {
              return tf::distance2(
                         tf::make_sphere(aabb.center(),
                                         aabb.diagonal().length() / 2),
                         primitive_query) <= threshold2;
            } else {
              return tf::intersects(primitive_query, aabb);
            }
          };
          const auto primitive_predicate = [&](const auto &value) {
            if constexpr (WithinDistance)
              return tf::distance2(value, primitive_query) <= threshold2;
            else
              return tf::intersects(value, primitive_query);
          };
          tf::gather_ids(native_form, aabb_predicate, primitive_predicate,
                         std::back_inserter(output));
        }
      },
      data, kind, polygon_vertices);
  return gather_ids_array(std::move(output));
}

template <bool WithinDistance, typename OutputIndex, typename Form0,
          typename Form1, typename Real>
auto form_pair_gather(const Form0 &a, const Form1 &b, Real distance = Real{})
    -> nd_array<OutputIndex> {
  using Index0 = gather_form_index_t<Form0>;
  using Index1 = gather_form_index_t<Form1>;
  const auto threshold2 = [&] {
    if constexpr (WithinDistance)
      return require_gather_distance(distance);
    else
      return Real{};
  }();

  std::vector<std::pair<Index0, Index1>> pairs;
  {
    const auto form_a = a.form();
    const auto form_b = b.form();
    const auto aabb_predicate = [&](const auto &aabb_a, const auto &aabb_b) {
      if constexpr (WithinDistance)
        return tf::distance2(aabb_a, aabb_b) <= threshold2;
      else
        return tf::intersects(aabb_a, aabb_b);
    };
    const auto primitive_predicate = [&](const auto &primitive_a,
                                         const auto &primitive_b) {
      if constexpr (WithinDistance)
        return tf::distance2(primitive_a, primitive_b) <= threshold2;
      else
        return tf::intersects(primitive_a, primitive_b);
    };
    tf::gather_ids(form_a, form_b, aabb_predicate, primitive_predicate,
                   std::back_inserter(pairs));
  }
  return gather_pair_array<OutputIndex>(pairs);
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto gather_ids(const mesh<Index, Real, Dims, Ngon> &form,
                const primitive<Real, Dims> &query) -> nd_array<Index> {
  return detail::form_primitive_gather<false>(form, query);
}

template <typename Index, typename Real, std::size_t Dims>
auto gather_ids(const edge_mesh<Index, Real, Dims> &form,
                const primitive<Real, Dims> &query) -> nd_array<Index> {
  return detail::form_primitive_gather<false>(form, query);
}

template <typename Real, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &form,
                const primitive<Real, Dims> &query) -> nd_array<std::int32_t> {
  return detail::form_primitive_gather<false>(form, query);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto gather_ids_within_distance(
    const mesh<Index, Real, Dims, Ngon> &form,
    const primitive<Real, Dims> &query,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_primitive_gather<true>(form, query, distance);
}

template <typename Index, typename Real, std::size_t Dims>
auto gather_ids_within_distance(
    const edge_mesh<Index, Real, Dims> &form,
    const primitive<Real, Dims> &query,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_primitive_gather<true>(form, query, distance);
}

template <typename Real, std::size_t Dims>
auto gather_ids_within_distance(
    const point_cloud<Real, Dims> &form, const primitive<Real, Dims> &query,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<std::int32_t> {
  return detail::form_primitive_gather<true>(form, query, distance);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0, std::size_t Ngon1>
auto gather_ids(const mesh<Index0, Real, Dims, Ngon0> &a,
                const mesh<Index1, Real, Dims, Ngon1> &b)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<false, common_index_t<Index0, Index1>>(
      a, b, Real{});
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0, std::size_t Ngon1>
auto gather_ids_within_distance(
    const mesh<Index0, Real, Dims, Ngon0> &a,
    const mesh<Index1, Real, Dims, Ngon1> &b,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<true, common_index_t<Index0, Index1>>(
      a, b, distance);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0>
auto gather_ids(const mesh<Index0, Real, Dims, Ngon0> &a,
                const edge_mesh<Index1, Real, Dims> &b)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<false, common_index_t<Index0, Index1>>(
      a, b, Real{});
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0>
auto gather_ids_within_distance(
    const mesh<Index0, Real, Dims, Ngon0> &a,
    const edge_mesh<Index1, Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<true, common_index_t<Index0, Index1>>(
      a, b, distance);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon0>
auto gather_ids(const mesh<Index, Real, Dims, Ngon0> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<Index> {
  return detail::form_pair_gather<false, Index>(a, b, Real{});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon0>
auto gather_ids_within_distance(
    const mesh<Index, Real, Dims, Ngon0> &a, const point_cloud<Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_pair_gather<true, Index>(a, b, distance);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon1>
auto gather_ids(const edge_mesh<Index0, Real, Dims> &a,
                const mesh<Index1, Real, Dims, Ngon1> &b)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<false, common_index_t<Index0, Index1>>(
      a, b, Real{});
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon1>
auto gather_ids_within_distance(
    const edge_mesh<Index0, Real, Dims> &a,
    const mesh<Index1, Real, Dims, Ngon1> &b,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<true, common_index_t<Index0, Index1>>(
      a, b, distance);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto gather_ids(const edge_mesh<Index0, Real, Dims> &a,
                const edge_mesh<Index1, Real, Dims> &b)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<false, common_index_t<Index0, Index1>>(
      a, b, Real{});
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto gather_ids_within_distance(
    const edge_mesh<Index0, Real, Dims> &a,
    const edge_mesh<Index1, Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<common_index_t<Index0, Index1>> {
  return detail::form_pair_gather<true, common_index_t<Index0, Index1>>(
      a, b, distance);
}

template <typename Index, typename Real, std::size_t Dims>
auto gather_ids(const edge_mesh<Index, Real, Dims> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<Index> {
  return detail::form_pair_gather<false, Index>(a, b, Real{});
}

template <typename Index, typename Real, std::size_t Dims>
auto gather_ids_within_distance(
    const edge_mesh<Index, Real, Dims> &a, const point_cloud<Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_pair_gather<true, Index>(a, b, distance);
}

template <typename Real, typename Index, std::size_t Dims, std::size_t Ngon1>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const mesh<Index, Real, Dims, Ngon1> &b) -> nd_array<Index> {
  return detail::form_pair_gather<false, Index>(a, b, Real{});
}

template <typename Real, typename Index, std::size_t Dims, std::size_t Ngon1>
auto gather_ids_within_distance(
    const point_cloud<Real, Dims> &a, const mesh<Index, Real, Dims, Ngon1> &b,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_pair_gather<true, Index>(a, b, distance);
}

template <typename Real, typename Index, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const edge_mesh<Index, Real, Dims> &b) -> nd_array<Index> {
  return detail::form_pair_gather<false, Index>(a, b, Real{});
}

template <typename Real, typename Index, std::size_t Dims>
auto gather_ids_within_distance(
    const point_cloud<Real, Dims> &a, const edge_mesh<Index, Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance) -> nd_array<Index> {
  return detail::form_pair_gather<true, Index>(a, b, distance);
}

template <typename Real, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<std::int32_t> {
  return detail::form_pair_gather<false, std::int32_t>(a, b, Real{});
}

template <typename Real, std::size_t Dims>
auto gather_ids_within_distance(
    const point_cloud<Real, Dims> &a, const point_cloud<Real, Dims> &b,
    typename detail::non_deduced<Real>::type distance)
    -> nd_array<std::int32_t> {
  return detail::form_pair_gather<true, std::int32_t>(a, b, distance);
}

} // namespace tf::cpp
