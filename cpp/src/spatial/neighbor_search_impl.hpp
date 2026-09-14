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
#include "trueform/cpp/spatial/neighbor_search.hpp"
#include "trueform/cpp/spatial/neighbor_search_knn.hpp"

#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/metric_point.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/nearest_neighbors.hpp"
#include "trueform/spatial/neighbor_search.hpp"
#include "trueform/spatial/tree_metric_info.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace detail {

/// The batch cutoff this query states for itself: one element is a tree
/// descent, so a short batch is still real work.
constexpr unsigned long neighbor_search_parallel_threshold = 100;

template <typename Real> auto require_radius(Real radius) -> void {
  if (radius < Real{0} || std::isnan(radius) ||
      (std::isinf(radius) && radius < Real{0}))
    throw std::invalid_argument(
        "neighbor_search: radius must be finite and nonnegative or positive "
        "infinity");
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto form_size(const cpp::mesh<Index, Real, Dims, Ngon> &form) -> std::size_t {
  return form.number_of_faces();
}

template <typename Real, std::size_t Dims>
auto form_size(const cpp::point_cloud<Real, Dims> &form) -> std::size_t {
  return form.number_of_points();
}

template <typename Index, typename Real, std::size_t Dims>
auto form_size(const cpp::edge_mesh<Index, Real, Dims> &form) -> std::size_t {
  return form.number_of_edges();
}

inline auto checked_product(int a, int b, int c = 1) -> std::size_t {
  if (a < 0 || b < 0 || c < 0)
    throw std::length_error("neighbor_search: negative result dimension");
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t value = 1;
  for (const auto factor : {a, b, c}) {
    const auto size = static_cast<std::size_t>(factor);
    if (size != 0 && value > limit / size)
      throw std::length_error(
          "neighbor_search: result shape exceeds int range");
    value *= size;
  }
  return value;
}

template <typename T>
auto filled_array(std::size_t size, tf::small_vector<int, 3> shape,
                  const T &value) -> nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(size);
  tf::parallel_fill(buffer, value);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real, std::size_t Dims, typename PointLike>
auto point_array(const PointLike &point) -> nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(Dims);
  for (std::size_t dimension = 0; dimension < Dims; ++dimension)
    buffer[dimension] = static_cast<Real>(point[dimension]);
  return nd_array<Real>::from_buffer(std::move(buffer),
                                     {static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims> auto zero_point() -> nd_array<Real> {
  return filled_array<Real>(Dims, {static_cast<int>(Dims)}, Real{0});
}

template <typename Real> auto miss_distance(Real radius) -> Real {
  return radius * radius;
}

/// An unstated radius is the type's maximum, so the search is unbounded.
template <typename Real> auto has_radius(Real radius) -> bool {
  return radius < std::numeric_limits<Real>::max();
}

template <typename Index, typename Real, std::size_t Dims>
auto make_neighbor_miss(Real radius) -> neighbor_result<Index, Real, Dims> {
  return {static_cast<Index>(-1), miss_distance(radius),
          zero_point<Real, Dims>()};
}

template <typename Index, typename Real, std::size_t Dims>
auto empty_knn_result(Real radius) -> neighbor_knn_result<Index, Real, Dims> {
  return {filled_array<Index>(0, {0}, static_cast<Index>(-1)),
          filled_array<Real>(0, {0, static_cast<int>(Dims)}, Real{0}),
          filled_array<Real>(0, {0}, miss_distance(radius))};
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto make_pair_miss(Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  return {static_cast<Index0>(-1), static_cast<Index1>(-1),
          miss_distance(radius), zero_point<Real, Dims>(),
          zero_point<Real, Dims>()};
}

template <typename Index, typename Real, std::size_t Dims>
auto make_batch_result(int count, Real radius)
    -> neighbor_batch_result<Index, Real, Dims> {
  const auto point_count = checked_product(count, static_cast<int>(Dims));
  return {
      filled_array<Index>(static_cast<std::size_t>(count), {count},
                          static_cast<Index>(-1)),
      filled_array<Real>(point_count, {count, static_cast<int>(Dims)}, Real{0}),
      filled_array<Real>(static_cast<std::size_t>(count), {count},
                         miss_distance(radius))};
}

template <typename Index, typename Real, std::size_t Dims>
auto make_knn_batch_result(int count, int k, Real radius)
    -> neighbor_knn_batch_result<Index, Real, Dims> {
  const auto result_count = checked_product(count, k);
  const auto point_count = checked_product(count, k, static_cast<int>(Dims));
  return {
      filled_array<Index>(result_count, {count, k}, static_cast<Index>(-1)),
      filled_array<Real>(point_count, {count, k, static_cast<int>(Dims)},
                         Real{0}),
      filled_array<Real>(result_count, {count, k}, miss_distance(radius)),
      filled_array<std::int32_t>(static_cast<std::size_t>(count), {count}, 0)};
}

template <typename Index, typename Real, std::size_t Dims, typename NativeForm>
auto native_neighbor_single(const NativeForm &form,
                            const primitive<Real, Dims> &query, Real radius)
    -> neighbor_result<Index, Real, Dims> {
  const auto values = query.data();
  const auto result = visit_spatial_primitive<Dims>(
      [&](const auto &value) {
        return tf::neighbor_search(form, value, radius);
      },
      values.raw_data(), query.kind(), query.polygon_vertex_count());
  if (!result)
    return make_neighbor_miss<Index, Real, Dims>(radius);
  return {static_cast<Index>(result.element), result.info.metric,
          point_array<Real, Dims>(result.info.point)};
}

template <typename Real, std::size_t Dims>
auto require_scalar_query(const primitive<Real, Dims> &query,
                          const char *operation) -> void {
  require_spatial(query.kind());
  if (query.is_batch())
    throw std::invalid_argument(std::string(operation) +
                                ": scalar query must not be a batch");
}

template <typename Real, std::size_t Dims>
auto require_batch_query(const primitive<Real, Dims> &queries,
                         const char *operation) -> void {
  require_spatial(queries.kind());
  if (!queries.is_batch())
    throw std::invalid_argument(std::string(operation) +
                                ": query must be a batch");
}

inline auto require_k(int k, const char *operation) -> void {
  if (k <= 0)
    throw std::invalid_argument(std::string(operation) +
                                ": k must be positive");
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto form_primitive_batch(const Form &form,
                          const primitive<Real, Dims> &queries, Real radius)
    -> neighbor_batch_result<Index, Real, Dims> {
  const auto count = queries.count();
  auto output = make_batch_result<Index, Real, Dims>(count, radius);
  if (count == 0)
    return output;

  const auto values = queries.data();
  const auto *data = values.raw_data();
  const auto kind = queries.kind();
  const auto polygon_vertices = queries.polygon_vertex_count();
  const auto stride = primitive_stride<Dims>(kind, polygon_vertices);
  auto *ids = output.element_ids.raw_data();
  auto *points = output.points.raw_data();
  auto *distances = output.distances.raw_data();

  const auto native_form = form.form();
  const auto compute = [&](int index) {
    const auto result = visit_spatial_primitive<Dims>(
        [&](const auto &value) {
          return tf::neighbor_search(native_form, value, radius);
        },
        data + static_cast<std::size_t>(index) * stride, kind,
        polygon_vertices);
    if (!result)
      return;
    ids[index] = static_cast<Index>(result.element);
    distances[index] = result.info.metric;
    for (std::size_t dimension = 0; dimension < Dims; ++dimension)
      points[static_cast<std::size_t>(index) * Dims + dimension] =
          result.info.point[dimension];
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(neighbor_search_parallel_threshold));
  return output;
}

template <typename Index, typename Real, std::size_t Dims, typename NativeForm>
auto native_neighbor_knn(const NativeForm &form,
                         const primitive<Real, Dims> &query, int k, Real radius)
    -> neighbor_knn_result<Index, Real, Dims> {
  using native_index = typename std::decay_t<decltype(form.tree())>::index_type;
  using native_result =
      tf::tree_metric_info<native_index, tf::metric_point<Real, Dims>>;
  std::vector<native_result> buffer(static_cast<std::size_t>(k));
  auto nearest = has_radius(radius)
                     ? tf::make_nearest_neighbors(
                           buffer.data(), static_cast<std::size_t>(k), radius)
                     : tf::make_nearest_neighbors(buffer.data(),
                                                  static_cast<std::size_t>(k));
  const auto values = query.data();
  visit_spatial_primitive<Dims>(
      [&](const auto &value) { tf::neighbor_search(form, value, nearest); },
      values.raw_data(), query.kind(), query.polygon_vertex_count());

  const auto count = static_cast<int>(nearest.size());
  auto ids = filled_array<Index>(static_cast<std::size_t>(count), {count},
                                 static_cast<Index>(-1));
  auto points =
      filled_array<Real>(checked_product(count, static_cast<int>(Dims)),
                         {count, static_cast<int>(Dims)}, Real{0});
  auto distances = filled_array<Real>(static_cast<std::size_t>(count), {count},
                                      miss_distance(radius));
  for (int index = 0; index < count; ++index) {
    ids[index] =
        static_cast<Index>(buffer[static_cast<std::size_t>(index)].element);
    distances[index] = buffer[static_cast<std::size_t>(index)].info.metric;
    for (std::size_t dimension = 0; dimension < Dims; ++dimension)
      points[static_cast<std::size_t>(index) * Dims + dimension] =
          buffer[static_cast<std::size_t>(index)].info.point[dimension];
  }
  return {std::move(ids), std::move(points), std::move(distances)};
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto form_primitive_knn(const Form &form, const primitive<Real, Dims> &query,
                        int k, Real radius)
    -> neighbor_knn_result<Index, Real, Dims> {
  checked_product(k, static_cast<int>(Dims));
  return native_neighbor_knn<Index, Real, Dims>(form.form(), query, k, radius);
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto form_primitive_knn_batch(const Form &form,
                              const primitive<Real, Dims> &queries, int k,
                              Real radius)
    -> neighbor_knn_batch_result<Index, Real, Dims> {
  const auto count = queries.count();
  auto output = make_knn_batch_result<Index, Real, Dims>(count, k, radius);
  if (count == 0)
    return output;

  const auto values = queries.data();
  const auto *data = values.raw_data();
  const auto kind = queries.kind();
  const auto polygon_vertices = queries.polygon_vertex_count();
  const auto stride = primitive_stride<Dims>(kind, polygon_vertices);
  auto *ids = output.element_ids.raw_data();
  auto *points = output.points.raw_data();
  auto *distances = output.distances.raw_data();
  auto *counts = output.counts.raw_data();

  {
    const auto native_form = form.form();
    using native_index =
        typename std::decay_t<decltype(native_form.tree())>::index_type;
    using native_result =
        tf::tree_metric_info<native_index, tf::metric_point<Real, Dims>>;
    const auto compute = [&](int query_index,
                             std::vector<native_result> &buffer) {
      buffer.resize(static_cast<std::size_t>(k));
      auto nearest =
          has_radius(radius)
              ? tf::make_nearest_neighbors(buffer.data(),
                                           static_cast<std::size_t>(k), radius)
              : tf::make_nearest_neighbors(buffer.data(),
                                           static_cast<std::size_t>(k));
      visit_spatial_primitive<Dims>(
          [&](const auto &value) {
            tf::neighbor_search(native_form, value, nearest);
          },
          data + static_cast<std::size_t>(query_index) * stride, kind,
          polygon_vertices);

      const auto found = static_cast<int>(nearest.size());
      counts[query_index] = found;
      const auto base =
          static_cast<std::size_t>(query_index) * static_cast<std::size_t>(k);
      for (int index = 0; index < found; ++index) {
        const auto &item = buffer[static_cast<std::size_t>(index)];
        ids[base + static_cast<std::size_t>(index)] =
            static_cast<Index>(item.element);
        distances[base + static_cast<std::size_t>(index)] = item.info.metric;
        for (std::size_t dimension = 0; dimension < Dims; ++dimension)
          points[(base + static_cast<std::size_t>(index)) * Dims + dimension] =
              item.info.point[dimension];
      }
    };
    tf::parallel_for_each(tf::make_sequence_range(count), compute,
                          std::vector<native_result>{},
                          tf::checked(neighbor_search_parallel_threshold));
  }
  return output;
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          typename Form0, typename Form1>
auto form_form_neighbor(const Form0 &a, const Form1 &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  const auto native_a = a.form();
  const auto native_b = b.form();
  // The owners may have emptied since their sizes were read.
  if (native_a.empty() || native_b.empty())
    return make_pair_miss<Index0, Real, Index1, Dims>(radius);
  const auto result = tf::neighbor_search(native_a, native_b, radius);
  if (!result)
    return make_pair_miss<Index0, Real, Index1, Dims>(radius);
  return neighbor_pair_result<Index0, Index1, Real, Dims>{
      static_cast<Index0>(result.elements.first),
      static_cast<Index1>(result.elements.second), result.info.metric,
      point_array<Real, Dims>(result.info.first),
      point_array<Real, Dims>(result.info.second)};
}

/// What every entry of one shape states before it searches, once per shape
/// rather than once per carrier.
template <typename Index, typename Real, std::size_t Dims, typename Form>
auto neighbor_single(const Form &form, const primitive<Real, Dims> &query,
                     Real radius) -> neighbor_result<Index, Real, Dims> {
  require_radius(radius);
  require_scalar_query(query, "neighbor_search");
  if (form_size(form) == 0)
    return make_neighbor_miss<Index, Real, Dims>(radius);
  return native_neighbor_single<Index, Real, Dims>(form.form(), query, radius);
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto neighbor_batch(const Form &form, const primitive<Real, Dims> &queries,
                    Real radius) -> neighbor_batch_result<Index, Real, Dims> {
  require_radius(radius);
  require_batch_query(queries, "neighbor_search_batch");
  if (form_size(form) == 0)
    return make_batch_result<Index, Real, Dims>(queries.count(), radius);
  return form_primitive_batch<Index, Real, Dims>(form, queries, radius);
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto neighbor_knn(const Form &form, const primitive<Real, Dims> &query, int k,
                  Real radius) -> neighbor_knn_result<Index, Real, Dims> {
  require_radius(radius);
  require_scalar_query(query, "neighbor_search_knn");
  require_k(k, "neighbor_search_knn");
  if (form_size(form) == 0)
    return empty_knn_result<Index, Real, Dims>(radius);
  return form_primitive_knn<Index, Real, Dims>(form, query, k, radius);
}

template <typename Index, typename Real, std::size_t Dims, typename Form>
auto neighbor_knn_batch(const Form &form, const primitive<Real, Dims> &queries,
                        int k, Real radius)
    -> neighbor_knn_batch_result<Index, Real, Dims> {
  require_radius(radius);
  require_batch_query(queries, "neighbor_search_knn_batch");
  require_k(k, "neighbor_search_knn_batch");
  if (form_size(form) == 0)
    return make_knn_batch_result<Index, Real, Dims>(queries.count(), k, radius);
  return form_primitive_knn_batch<Index, Real, Dims>(form, queries, k, radius);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          typename Form0, typename Form1>
auto neighbor_pair(const Form0 &a, const Form1 &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  require_radius(radius);
  if (form_size(a) == 0 || form_size(b) == 0)
    return make_pair_miss<Index0, Real, Index1, Dims>(radius);
  return form_form_neighbor<Index0, Real, Index1, Dims>(a, b, radius);
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search(const mesh<Index, Real, Dims, Ngon> &form,
                     const primitive<Real, Dims> &query, Real radius)
    -> neighbor_result<Index, Real, Dims> {
  return detail::neighbor_single<Index, Real, Dims>(form, query, radius);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_batch(const mesh<Index, Real, Dims, Ngon> &form,
                           const primitive<Real, Dims> &queries, Real radius)
    -> neighbor_batch_result<Index, Real, Dims> {
  return detail::neighbor_batch<Index, Real, Dims>(form, queries, radius);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_knn(const mesh<Index, Real, Dims, Ngon> &form,
                         const primitive<Real, Dims> &query, int k, Real radius)
    -> neighbor_knn_result<Index, Real, Dims> {
  return detail::neighbor_knn<Index, Real, Dims>(form, query, k, radius);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_knn_batch(const mesh<Index, Real, Dims, Ngon> &form,
                               const primitive<Real, Dims> &queries, int k,
                               Real radius)
    -> neighbor_knn_batch_result<Index, Real, Dims> {
  return detail::neighbor_knn_batch<Index, Real, Dims>(form, queries, k,
                                                       radius);
}

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index, Real, Dims> &form,
                     const primitive<Real, Dims> &query, Real radius)
    -> neighbor_result<Index, Real, Dims> {
  return detail::neighbor_single<Index, Real, Dims>(form, query, radius);
}

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_batch(const edge_mesh<Index, Real, Dims> &form,
                           const primitive<Real, Dims> &queries, Real radius)
    -> neighbor_batch_result<Index, Real, Dims> {
  return detail::neighbor_batch<Index, Real, Dims>(form, queries, radius);
}

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_knn(const edge_mesh<Index, Real, Dims> &form,
                         const primitive<Real, Dims> &query, int k, Real radius)
    -> neighbor_knn_result<Index, Real, Dims> {
  return detail::neighbor_knn<Index, Real, Dims>(form, query, k, radius);
}

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_knn_batch(const edge_mesh<Index, Real, Dims> &form,
                               const primitive<Real, Dims> &queries, int k,
                               Real radius)
    -> neighbor_knn_batch_result<Index, Real, Dims> {
  return detail::neighbor_knn_batch<Index, Real, Dims>(form, queries, k,
                                                       radius);
}

template <typename Real, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &form,
                     const primitive<Real, Dims> &query, Real radius)
    -> neighbor_result<std::int32_t, Real, Dims> {
  return detail::neighbor_single<std::int32_t, Real, Dims>(form, query, radius);
}

template <typename Real, std::size_t Dims>
auto neighbor_search_batch(const point_cloud<Real, Dims> &form,
                           const primitive<Real, Dims> &queries, Real radius)
    -> neighbor_batch_result<std::int32_t, Real, Dims> {
  return detail::neighbor_batch<std::int32_t, Real, Dims>(form, queries,
                                                          radius);
}

template <typename Real, std::size_t Dims>
auto neighbor_search_knn(const point_cloud<Real, Dims> &form,
                         const primitive<Real, Dims> &query, int k, Real radius)
    -> neighbor_knn_result<std::int32_t, Real, Dims> {
  return detail::neighbor_knn<std::int32_t, Real, Dims>(form, query, k, radius);
}

template <typename Real, std::size_t Dims>
auto neighbor_search_knn_batch(const point_cloud<Real, Dims> &form,
                               const primitive<Real, Dims> &queries, int k,
                               Real radius)
    -> neighbor_knn_batch_result<std::int32_t, Real, Dims> {
  return detail::neighbor_knn_batch<std::int32_t, Real, Dims>(form, queries, k,
                                                              radius);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0, std::size_t Ngon1>
auto neighbor_search(const mesh<Index0, Real, Dims, Ngon0> &a,
                     const mesh<Index1, Real, Dims, Ngon1> &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  return detail::neighbor_pair<Index0, Real, Index1, Dims>(a, b, radius);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0>
auto neighbor_search(const mesh<Index0, Real, Dims, Ngon0> &a,
                     const edge_mesh<Index1, Real, Dims> &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  return detail::neighbor_pair<Index0, Real, Index1, Dims>(a, b, radius);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon0>
auto neighbor_search(const mesh<Index, Real, Dims, Ngon0> &a,
                     const point_cloud<Real, Dims> &b, Real radius)
    -> neighbor_pair_result<Index, std::int32_t, Real, Dims> {
  return detail::neighbor_pair<Index, Real, std::int32_t, Dims>(a, b, radius);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon1>
auto neighbor_search(const edge_mesh<Index0, Real, Dims> &a,
                     const mesh<Index1, Real, Dims, Ngon1> &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  return detail::neighbor_pair<Index0, Real, Index1, Dims>(a, b, radius);
}

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index0, Real, Dims> &a,
                     const edge_mesh<Index1, Real, Dims> &b, Real radius)
    -> neighbor_pair_result<Index0, Index1, Real, Dims> {
  return detail::neighbor_pair<Index0, Real, Index1, Dims>(a, b, radius);
}

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index, Real, Dims> &a,
                     const point_cloud<Real, Dims> &b, Real radius)
    -> neighbor_pair_result<Index, std::int32_t, Real, Dims> {
  return detail::neighbor_pair<Index, Real, std::int32_t, Dims>(a, b, radius);
}

template <typename Real, typename Index, std::size_t Dims, std::size_t Ngon1>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const mesh<Index, Real, Dims, Ngon1> &b, Real radius)
    -> neighbor_pair_result<std::int32_t, Index, Real, Dims> {
  return detail::neighbor_pair<std::int32_t, Real, Index, Dims>(a, b, radius);
}

template <typename Real, typename Index, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const edge_mesh<Index, Real, Dims> &b, Real radius)
    -> neighbor_pair_result<std::int32_t, Index, Real, Dims> {
  return detail::neighbor_pair<std::int32_t, Real, Index, Dims>(a, b, radius);
}

template <typename Real, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const point_cloud<Real, Dims> &b, Real radius)
    -> neighbor_pair_result<std::int32_t, std::int32_t, Real, Dims> {
  return detail::neighbor_pair<std::int32_t, Real, std::int32_t, Dims>(a, b,
                                                                       radius);
}

} // namespace tf::cpp
