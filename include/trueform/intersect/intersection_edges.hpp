/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/dot.hpp"
#include "../core/views/slide_range.hpp"
#include "./scalar_field_intersections.hpp"
#include "./tagged_intersections.hpp"

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
        Index size = polygons[i0.object].size();
        auto next_id = tf::circular_increment(i1.target.id, size);
        if (i0.target.id == next_id) {
          std::swap(i0, i1);
          next_id = tf::circular_increment(i1.target.id, size);
        }
        if (scalar_field[polygons[i0.object].indices()[next_id]] < cut_value) {
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
        const auto &face = polygons.faces()[i0.object];
        auto s0 = scalar_field[face[i0.target.id]];
        auto s1 = scalar_field[face[i1.target.id]];
        if (s0 > s1)
          std::swap(s0, s1);
        auto cut_value =
            *std::upper_bound(cut_values.begin(), cut_values.end(), s0);
        Index size = polygons[i0.object].size();
        auto next_id = tf::circular_increment(i1.target.id, size);
        if (i0.target.id == next_id) {
          std::swap(i0, i1);
          next_id = tf::circular_increment(i1.target.id, size);
        }
        if (scalar_field[polygons[i0.object].indices()[next_id]] < cut_value) {
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
    const tf::tagged_intersections<Index, RealT, Dims> &intersections) {
  tf::blocked_buffer<Index, 2> buffer;
  const auto &intersection_points = intersections.intersection_points();
  struct node_t {
    Index id;
    RealT t;
  };
  auto handle_complex_edges_f = [&intersection_points](
                                    auto begin, auto end, auto &buffer,
                                    tf::buffer<node_t> &_work_buffer) {
    using Iterator = decltype(begin);
    std::array<std::pair<RealT, Iterator>,
               tf::static_size_v<decltype(intersection_points.front())>>
        min;
    min.fill({std::numeric_limits<RealT>::max(), begin});
    std::array<std::pair<RealT, Iterator>,
               tf::static_size_v<decltype(intersection_points.front())>>
        max;
    max.fill({std::numeric_limits<RealT>::min(), begin});

    _work_buffer.clear();
    auto it = begin;
    while (it != end) {
      auto pt = intersection_points[it->id];
      _work_buffer.push_back({it->id, RealT(0)});
      for (std::size_t i = 0; i < min.size(); ++i) {
        min[i] = std::min(min[i], std::make_pair(pt[i], it));
        max[i] = std::max(max[i], std::make_pair(pt[i], it));
      }
      ++it;
    }
    auto res = std::make_pair(max[0].first - min[0].first, std::size_t(0));
    for (std::size_t i = 1; i < min.size(); ++i) {
      res = std::max(res, std::make_pair(max[i].first - min[i].first, i));
    }
    auto origin = intersection_points[min[res.second].second->id];
    auto dir = intersection_points[max[res.second].second->id] - origin;

    for (auto &e : _work_buffer)
      e.t = tf::dot(intersection_points[e.id] - origin, dir);
    std::sort(_work_buffer.begin(), _work_buffer.end(),
              [](const auto &x, const auto &y) {
                return std::make_pair(x.t, x.id) < std::make_pair(y.t, y.id);
              });
    for (auto [a, b] : tf::make_slide_range<2>(_work_buffer)) {
      buffer.push_back(a.id);
      buffer.push_back(b.id);
    }
  };
  tf::generic_generate(
      intersections.intersections(), buffer.data_buffer(),
      tf::buffer<node_t>{}, [&](const auto &r, auto &buff, auto &_work_buffer) {
        auto it = r.begin();
        auto end = r.end();
        while (it != end) {
          auto next = std::find_if(it + 1, end, [it](const auto &x) {
            return x.object_other != it->object_other;
          });
          auto n = next - it;
          if (n == 2) {
            buff.push_back(it->id);
            buff.push_back((it + 1)->id);
          } else if (n > 2)
            handle_complex_edges_f(it, next, buff, _work_buffer);
          it = next;
        }
      });
  return buffer;
}
} // namespace tf
