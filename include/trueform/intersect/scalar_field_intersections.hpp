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
#include "../core/algorithm/block_reduce.hpp"
#include "../core/polygons.hpp"
#include "../core/views/enumerate.hpp"
#include "./types/simple_intersections.hpp"
#include <algorithm>

namespace tf {

/// @ingroup intersect_data
/// @brief Low-level scalar field intersection data.
///
/// Computes and stores points where a scalar field defined over mesh vertices
/// crosses threshold values. Use @ref tf::make_isocontours for high-level
/// curve extraction.
///
/// @tparam Index The index type.
/// @tparam RealT The coordinate type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class scalar_field_intersections
    : public intersect::simple_intersections<Index, RealT, Dims> {
  using base_t = intersect::simple_intersections<Index, RealT, Dims>;

public:
  template <typename Policy, typename Range>
  auto build(const tf::polygons<Policy> &polygons, const Range &scalar_field,
             typename Range::value_type cut_value = {}) {
    // stack storage outlives build_impl; a pointer into the closure would
    // pin the value in memory
    typename Range::value_type single[1] = {cut_value};
    const auto *base = single;
    return build_impl(
        polygons, scalar_field, [base, cut_value](auto min, auto max) {
          return cut_span<decltype(base)>{
              base, Index(min <= cut_value && cut_value < max), Index(0)};
        });
  }

  template <typename Policy, typename Range0, typename Range1>
  auto build_many(const tf::polygons<Policy> &polygons,
                  const Range0 &scalar_field, const Range1 &cut_values) {
    return build_impl(
        polygons, scalar_field, [&](auto min, auto max) {
          auto first =
              std::lower_bound(cut_values.begin(), cut_values.end(), min);
          auto last = first;
          while (last != cut_values.end() && *last < max)
            ++last;
          return cut_span<decltype(first)>{
              first, static_cast<Index>(last - first),
              static_cast<Index>(first - cut_values.begin())};
        });
  }

private:
  // 16 bytes so handlers return it in registers, not through memory
  template <typename Iterator> struct cut_span {
    Iterator first;
    Index n;
    Index cut;
  };

  template <typename Policy, typename Range, typename F>
  auto build_impl(const tf::polygons<Policy> &polygons,
                  const Range &scalar_field, const F &handler_f) {
    base_t::clear();
    tf::buffer<tf::intersect::simple_edge_point_id<Index>> edge_ids;
    std::tuple<tf::buffer<tf::intersect::simple_intersection<Index>>,
               tf::buffer<tf::intersect::simple_edge_point_id<Index>>,
               tf::buffer<tf::point<RealT, Dims>>>
        local_result;

    tf::blocked_reduce(
        tf::enumerate(polygons),
        std::tie(base_t::_intersections, edge_ids, base_t::_points),
        local_result,
        [&scalar_field, &handler_f](const auto &r, auto &local_result) {
          auto &intersections = std::get<0>(local_result);
          auto &edge_point_ids = std::get<1>(local_result);
          auto &points = std::get<2>(local_result);
          intersections.reserve(1000);
          edge_point_ids.reserve(1000);
          points.reserve(1000);
          for (const auto &enum_pair : r) {
            Index polygon_id = static_cast<Index>(get<0>(enum_pair));
            const auto &polygon = get<1>(enum_pair);
            const auto &face = polygon.indices();
            std::size_t size = polygon.size();
            std::size_t prev = size - 1;
            for (std::size_t i = 0; i < size; prev = i++) {
              Index v0 = prev;
              Index v1 = i;
              if (scalar_field[face[v1]] < scalar_field[face[v0]])
                std::swap(v0, v1);
              Index id0 = face[v0];
              Index id1 = face[v1];
              auto span = handler_f(scalar_field[id0], scalar_field[id1]);
              for (Index k = 0; k < span.n; ++k) {
                auto cut_value = span.first[k];
                Index cut = span.cut + k;
                // vertex hits take (id, id) identity: they merge across all
                // incident faces
                bool on_vertex = cut_value == scalar_field[id0];
                if (on_vertex) {
                  // a touch (both neighbors above) joins no chord
                  std::size_t other =
                      v0 == Index(prev) ? (std::size_t(v0) + size - 1) % size
                                        : (std::size_t(v0) + 1) % size;
                  if (scalar_field[face[other]] > cut_value)
                    continue;
                }
                tf::point<RealT, Dims> created_point = polygon[v0];
                if (!on_vertex) {
                  auto t = (cut_value - scalar_field[id0]) /
                           (scalar_field[id1] - scalar_field[id0]);
                  created_point =
                      created_point + t * (polygon[v1] - polygon[v0]);
                }
                Index pt_id = points.size();
                points.push_back(created_point);
                edge_point_ids.push_back(
                    {id0, on_vertex ? id0 : id1, cut, pt_id});
                intersections.push_back(
                    {polygon_id, cut,
                     on_vertex
                         ? tf::topo_id<Index>{v0, tf::topo_type::vertex}
                         : tf::topo_id<Index>{Index(prev),
                                              tf::topo_type::edge},
                     pt_id});
              }
            }
          }
        },
        [](const auto &local_result, auto &result) {
          auto &&[l_intersections, l_edge_point_ids, l_points] = local_result;
          auto &&[intersections, edge_point_ids, points] = result;
          Index pt_offset = points.size();
          auto intersections_old_size = intersections.size();
          intersections.reallocate(intersections.size() +
                                   l_intersections.size());
          auto intersections_it =
              intersections.begin() + intersections_old_size;
          for (auto e : l_intersections) {
            e.id += pt_offset;
            *intersections_it++ = e;
          }
          //
          auto edge_point_ids_old_size = edge_point_ids.size();
          edge_point_ids.reallocate(edge_point_ids.size() +
                                    l_edge_point_ids.size());
          auto edge_point_ids_it =
              edge_point_ids.begin() + edge_point_ids_old_size;
          for (const auto &e : l_edge_point_ids) {
            *edge_point_ids_it = e;
            (*edge_point_ids_it++).point_id += pt_offset;
          }
          //
          auto points_old_size = points.size();
          points.reallocate(points.size() + l_points.size());
          auto points_it = points.begin() + points_old_size;
          std::copy(l_points.begin(), l_points.end(), points_it);
        });

    base_t::finalize(std::move(edge_ids));
  }
};
} // namespace tf
