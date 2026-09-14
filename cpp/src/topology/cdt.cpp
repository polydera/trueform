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
#include "trueform/cpp/topology/cdt.hpp"

#include "trueform/core/algorithm/compose_index_maps.hpp"
#include "trueform/core/algorithm/mask_to_index_map.hpp"
#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/views/mapped_range.hpp"
#include "trueform/exact/resolve_int_type.hpp"
#include "trueform/reindex/polygons.hpp"
#include "trueform/topology/constrained_delaunay_triangulator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace {

template <typename Real>
auto require_points(const nd_array<Real> &points) -> std::int32_t {
  if (!points.is_valid() || points.ndim() != 2 || points.shape_at(1) != 2)
    throw std::invalid_argument("make_cdt: points must have shape [N, 2]");
  return static_cast<std::int32_t>(points.shape_at(0));
}

auto require_edges(const nd_array<std::int32_t> &edges,
                   std::int32_t point_count) -> std::int32_t {
  if (!edges.is_valid() || edges.ndim() != 2 || edges.shape_at(1) != 2)
    throw std::invalid_argument("make_cdt: edges must have shape [N, 2]");
  for (const auto point : edges)
    if (point < 0 || point >= point_count)
      throw std::out_of_range("make_cdt: edge point index out of range");
  return static_cast<std::int32_t>(edges.shape_at(0));
}

auto require_mask(const nd_array<std::int8_t> &mask, std::int32_t edge_count)
    -> void {
  if (!mask.is_valid() || mask.ndim() != 1 || mask.shape_at(0) != edge_count)
    throw std::invalid_argument(
        "make_cdt: edge mask must have shape [number of edges]");
}

template <typename T>
auto empty_array(tf::small_vector<int, 3> shape) -> nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(0);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

auto empty_point_map(std::int32_t point_count) -> index_map<> {
  tf::index_map_buffer<std::int32_t> map;
  map.f().allocate(static_cast<std::size_t>(point_count));
  map.kept_ids().allocate(0);
  tf::parallel_fill(map.f(), point_count);
  return index_map<>::from_index_map_buffer(std::move(map));
}

template <typename Real> auto empty_result() -> cdt_result<Real> {
  return {empty_array<std::int32_t>({0, 3}), empty_array<Real>({0, 2})};
}

template <typename Real>
auto empty_result_with_map(std::int32_t point_count)
    -> cdt_result_with_map<Real> {
  return {empty_array<std::int32_t>({0, 3}), empty_array<Real>({0, 2}),
          empty_point_map(point_count)};
}

template <typename Real, typename Polygons>
auto pack_result(Polygons &&polygons) -> cdt_result<Real> {
  const auto face_count = static_cast<int>(polygons.faces().size());
  const auto point_count = static_cast<int>(polygons.points().size());
  return {
      nd_array<std::int32_t>::from_buffer(
          std::move(polygons.faces_buffer().data_buffer()), {face_count, 3}),
      nd_array<Real>::from_buffer(
          std::move(polygons.points_buffer().data_buffer()), {point_count, 2})};
}

template <typename Real, typename Polygons>
auto pack_result_with_map(Polygons &&polygons,
                          tf::index_map_buffer<std::int32_t> &&point_index_map)
    -> cdt_result_with_map<Real> {
  const auto face_count = static_cast<int>(polygons.faces().size());
  const auto point_count = static_cast<int>(polygons.points().size());
  return {
      nd_array<std::int32_t>::from_buffer(
          std::move(polygons.faces_buffer().data_buffer()), {face_count, 3}),
      nd_array<Real>::from_buffer(
          std::move(polygons.points_buffer().data_buffer()), {point_count, 2}),
      index_map<>::from_index_map_buffer(std::move(point_index_map))};
}

template <typename Real, typename Cdt> auto extract_interior(Cdt &cdt) {
  auto faces = cdt.make_faces();
  auto polygons = tf::make_polygons(faces, cdt.converted_points());
  auto interior = tf::make_mapped_range(
      cdt.region_labels(), [](auto label) { return label % 2 == 1; });
  auto face_map = tf::mask_to_index_map<std::int32_t>(interior);

  tf::buffer<bool> point_mask;
  point_mask.allocate(polygons.points().size());
  tf::parallel_fill(point_mask, false);
  for (const auto face_id : face_map.kept_ids())
    for (const auto point_id : polygons.faces()[face_id])
      point_mask[static_cast<std::size_t>(point_id)] = true;

  auto point_map = tf::mask_to_index_map<std::int32_t>(point_mask);
  auto output = tf::reindexed(polygons, face_map, point_map);
  return std::make_pair(std::move(output), std::move(point_map));
}

template <typename Real, typename Points, typename Edges>
auto build_constrained(const Points &points, const Edges &edges,
                       bool split_constraints) {
  using Int = tf::exact::resolve_int_type<tf::none_t, Real>;
  tf::constrained_delaunay_triangulator<std::int32_t, Real, Int> cdt;
  const auto built = cdt.build(points, edges, split_constraints);
  return std::make_pair(std::move(cdt), built);
}

template <typename Real, typename Points, typename Edges, typename Mask>
auto build_constrained(const Points &points, const Edges &edges,
                       const Mask &mask, bool split_constraints) {
  using Int = tf::exact::resolve_int_type<tf::none_t, Real>;
  tf::constrained_delaunay_triangulator<std::int32_t, Real, Int> cdt;
  const auto built = cdt.build(points, edges, mask, split_constraints);
  return std::make_pair(std::move(cdt), built);
}

} // namespace

template <typename Real>
auto make_cdt(const nd_array<Real> &points) -> cdt_result<Real> {
  require_points(points);
  using Int = tf::exact::resolve_int_type<tf::none_t, Real>;
  tf::constrained_delaunay_triangulator<std::int32_t, Real, Int> cdt;
  tf::buffer<std::array<std::int32_t, 2>> no_edges;
  if (!cdt.build(tf::make_points<2>(points.make_range()),
                 tf::make_edges(tf::make_range(no_edges))))
    return empty_result<Real>();
  auto faces = cdt.make_faces();
  return pack_result<Real>(tf::make_polygons_buffer(
      tf::make_polygons(faces, cdt.converted_points())));
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points)
    -> cdt_result_with_map<Real> {
  const auto point_count = require_points(points);
  using Int = tf::exact::resolve_int_type<tf::none_t, Real>;
  tf::constrained_delaunay_triangulator<std::int32_t, Real, Int> cdt;
  tf::buffer<std::array<std::int32_t, 2>> no_edges;
  if (!cdt.build(tf::make_points<2>(points.make_range()),
                 tf::make_edges(tf::make_range(no_edges))))
    return empty_result_with_map<Real>(point_count);
  auto faces = cdt.make_faces();
  auto polygons = tf::make_polygons_buffer(
      tf::make_polygons(faces, cdt.converted_points()));
  return pack_result_with_map<Real>(std::move(polygons),
                                    std::move(cdt.index_map()));
}

template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              bool split_constraints) -> cdt_result<Real> {
  const auto point_count = require_points(points);
  require_edges(edges, point_count);
  auto [cdt, built] = build_constrained<Real>(
      tf::make_points<2>(points.make_range()),
      tf::make_edges(tf::make_blocked_range<2>(edges.make_range())),
      split_constraints);
  if (!built)
    return empty_result<Real>();
  auto [polygons, point_map] = extract_interior<Real>(cdt);
  return pack_result<Real>(std::move(polygons));
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        bool split_constraints) -> cdt_result_with_map<Real> {
  const auto point_count = require_points(points);
  require_edges(edges, point_count);
  auto [cdt, built] = build_constrained<Real>(
      tf::make_points<2>(points.make_range()),
      tf::make_edges(tf::make_blocked_range<2>(edges.make_range())),
      split_constraints);
  if (!built)
    return empty_result_with_map<Real>(point_count);
  auto [polygons, point_map] = extract_interior<Real>(cdt);
  auto composed = tf::compose_index_maps(cdt.index_map(), point_map);
  return pack_result_with_map<Real>(std::move(polygons), std::move(composed));
}

template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              const nd_array<std::int8_t> &edge_mask, bool split_constraints)
    -> cdt_result<Real> {
  const auto point_count = require_points(points);
  const auto edge_count = require_edges(edges, point_count);
  require_mask(edge_mask, edge_count);
  auto [cdt, built] = build_constrained<Real>(
      tf::make_points<2>(points.make_range()),
      tf::make_edges(tf::make_blocked_range<2>(edges.make_range())),
      edge_mask.make_range(), split_constraints);
  if (!built)
    return empty_result<Real>();
  auto [polygons, point_map] = extract_interior<Real>(cdt);
  return pack_result<Real>(std::move(polygons));
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        const nd_array<std::int8_t> &edge_mask,
                        bool split_constraints) -> cdt_result_with_map<Real> {
  const auto point_count = require_points(points);
  const auto edge_count = require_edges(edges, point_count);
  require_mask(edge_mask, edge_count);
  auto [cdt, built] = build_constrained<Real>(
      tf::make_points<2>(points.make_range()),
      tf::make_edges(tf::make_blocked_range<2>(edges.make_range())),
      edge_mask.make_range(), split_constraints);
  if (!built)
    return empty_result_with_map<Real>(point_count);
  auto [polygons, point_map] = extract_interior<Real>(cdt);
  auto composed = tf::compose_index_maps(cdt.index_map(), point_map);
  return pack_result_with_map<Real>(std::move(polygons), std::move(composed));
}

#define TF_CPP_INSTANTIATE_CDT(Real)                                           \
  template auto make_cdt(const nd_array<Real> &) -> cdt_result<Real>;          \
  template auto make_cdt_with_maps(const nd_array<Real> &)                     \
      -> cdt_result_with_map<Real>;                                            \
  template auto make_cdt(const nd_array<Real> &,                               \
                         const nd_array<std::int32_t> &, bool)                 \
      -> cdt_result<Real>;                                                     \
  template auto make_cdt_with_maps(const nd_array<Real> &,                     \
                                   const nd_array<std::int32_t> &, bool)       \
      -> cdt_result_with_map<Real>;                                            \
  template auto make_cdt(                                                      \
      const nd_array<Real> &, const nd_array<std::int32_t> &,                  \
      const nd_array<std::int8_t> &, bool) -> cdt_result<Real>;                \
  template auto make_cdt_with_maps(                                            \
      const nd_array<Real> &, const nd_array<std::int32_t> &,                  \
      const nd_array<std::int8_t> &, bool) -> cdt_result_with_map<Real>

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_CDT)

#undef TF_CPP_INSTANTIATE_CDT

} // namespace tf::cpp
