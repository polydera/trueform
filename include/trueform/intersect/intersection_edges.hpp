/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "./forms_intersections.hpp"
#include "./scalar_field_intersections.hpp"

namespace tf {
template <typename Index, typename RealT, std::size_t Dims, typename Policy,
          typename Range>
auto make_intersection_edges(
    const tf::scalar_field_intersections<Index, RealT, Dims> &sfi,
    const tf::polygons<Policy> &polygons, const Range &scalar_field,
    typename Range::value_type cut_value = {}) {
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(
      sfi.intersections(), buffer.data_buffer(),
      [&](const auto &r, auto &buff) {
        // scalar field intersections either have 1
        // or 2 elements.
        if (r.size() != 2)
          return;
        auto i0 = r[0];
        auto i1 = r[1];
        Index size = polygons[i0.polygon].size();
        auto next_id = tf::circular_increment(i1.target.id, size);
        if (i0.target.id == next_id) {
          std::swap(i0, i1);
          next_id = tf::circular_increment(i1.target.id, size);
        }
        if (scalar_field[polygons[i0.polygon].indices()[next_id]] < cut_value) {
          std::swap(i0, i1);
        }
        buff.push_back(i0.id);
        buff.push_back(i1.id);
      });

  return buffer;
}

template <typename Index, typename RealT, std::size_t Dims, typename Policy,
          typename Range, typename Iterator, std::size_t N>
auto make_intersection_edges(
    const tf::scalar_field_intersections<Index, RealT, Dims> &sfi,
    const tf::polygons<Policy> &polygons, const Range &scalar_field,
    const tf::range<Iterator, N> &cut_values) {
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(
      sfi.intersections(), buffer.data_buffer(),
      [&](const auto &r, auto &buff) {
        // scalar field intersections either have 1
        // or 2 elements.
        if (r.size() != 2)
          return;
        auto i0 = r[0];
        auto i1 = r[1];
        const auto &face = polygons.faces()[i0.polygon];
        auto s0 = scalar_field[face[i0.target.id]];
        auto s1 = scalar_field[face[i1.target.id]];
        if (s0 > s1)
          std::swap(s0, s1);
        auto cut_value =
            *std::upper_bound(cut_values.begin(), cut_values.end(), s0);
        Index size = polygons[i0.polygon].size();
        auto next_id = tf::circular_increment(i1.target.id, size);
        if (i0.target.id == next_id) {
          std::swap(i0, i1);
          next_id = tf::circular_increment(i1.target.id, size);
        }
        if (scalar_field[polygons[i0.polygon].indices()[next_id]] < cut_value) {
          std::swap(i0, i1);
        }
        buff.push_back(i0.id);
        buff.push_back(i1.id);
      });

  return buffer;
}

template <typename Index, typename RealT, std::size_t Dims>
auto make_intersection_edges(
    const tf::intersect::simple_intersections<Index, RealT, Dims>
        &intersections) {
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(intersections.intersections(), buffer.data_buffer(),
                       [&](const auto &r, auto &buff) {
                         // scalar field intersections either have 1
                         // or 2 elements.
                         if (r.size() != 2)
                           return;
                         buff.push_back(r[0].id);
                         buff.push_back(r[1].id);
                       });
  return buffer;
}

template <typename Index, typename RealT, std::size_t Dims>
auto make_intersection_edges(
    const tf::forms_intersections<Index, RealT, Dims> &intersections) {
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(intersections.intersections(), buffer.data_buffer(),
                       [&](const auto &r, auto &buff) {
                         auto it = r.begin();
                         auto end = r.end();
                         while (it != end) {
                           auto next =
                               std::find_if(it + 1, end, [it](const auto &x) {
                                 return x.polygon_other != it->polygon_other;
                               });
                           // each polygon has 1 or 2 intersections with another
                           // polygon
                           if (next - it == 2) {
                             buff.push_back(it->id);
                             buff.push_back((it + 1)->id);
                           }
                           it = next;
                         }
                       });
  return buffer;
}
} // namespace tf
