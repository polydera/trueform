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
    return build_impl(polygons, scalar_field, [cut_value](auto min, auto max) {
      return std::make_pair(min < cut_value && max > cut_value, cut_value);
    });
  }

  template <typename Policy, typename Range0, typename Range1>
  auto build_many(const tf::polygons<Policy> &polygons,
                  const Range0 &scalar_field, const Range1 &cut_values) {
    return build_impl(polygons, scalar_field, [&](auto min, auto max) {
      return handle_cut(min, max, cut_values);
    });
  }

private:
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
          auto &&[intersections, edge_point_ids, points] = local_result;
          intersections.reserve(1000);
          edge_point_ids.reserve(1000);
          points.reserve(1000);
          for (const auto &[polygon_id, polygon] : r) {
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
              auto [should_cut, cut_value] =
                  handler_f(scalar_field[id0], scalar_field[id1]);
              if (!should_cut)
                continue;

              auto edge = polygon[v1] - polygon[v0];

              auto t = (cut_value - scalar_field[id0]) /
                       (scalar_field[id1] - scalar_field[id0]);
              auto created_point = polygon[v0] + t * edge;
              Index pt_id = points.size();
              points.push_back(created_point);
              edge_point_ids.push_back({Index(id0), Index(id1), Index(pt_id)});
              intersections.push_back(
                  {Index(polygon_id),
                   tf::intersect::intersection_target<Index>{
                       Index(prev), tf::topo_type::edge},
                   pt_id});
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

  template <typename T, typename Range>
  auto handle_cut(T min, T max, const Range &cut_values) {
    // Find the first cut value strictly greater than min
    auto it = std::upper_bound(cut_values.begin(), cut_values.end(), min);

    // Check if that cut value is still below max
    if (it == cut_values.end() || *it >= max)
      return std::make_pair(false, RealT(0));

    return std::make_pair(true, RealT(*it));
  }
};
} // namespace tf
