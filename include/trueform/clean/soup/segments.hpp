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
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/remove_if_and_make_map.hpp"
#include "../../core/buffer.hpp"
#include "../../core/segments.hpp"
#include "../../core/segments_buffer.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/zip.hpp"
#include "../index_map/points.hpp"

namespace tf::clean {
template <typename Index, typename RealT, std::size_t Dims>
class segment_soup : public segments_buffer<Index, RealT, Dims> {
  using base_t = segments_buffer<Index, RealT, Dims>;

public:
  template <typename Policy>
  auto build(const tf::segments<Policy> &segments, RealT tolerance,
             bool should_remove_uncontained_points = true) {
    clear();
    make_initial_points(segments, tolerance);
    make_initial_edges();
    if (should_remove_uncontained_points)
      remove_uncontained_points();
  }

  template <typename Policy>
  auto build(const tf::segments<Policy> &segments,
             bool should_remove_uncontained_points = true) {
    build(segments, RealT{0}, should_remove_uncontained_points);
  }

  auto clear() {
    base_t::clear();
    _im.f().clear();
    _im.kept_ids().clear();
    _contained_points.clear();
  }

private:
  auto remove_uncontained_points() {
    _contained_points.allocate(base_t::points().size());
    tf::parallel_fill(_contained_points, false);
    tf::parallel_for_each(
        base_t::edges(),
        [&](const auto &edge) {
          _contained_points[edge[0]] = true;
          _contained_points[edge[1]] = true;
        },
        tf::checked);
    auto &map = _im.f();
    map.allocate(base_t::points().size());
    auto r = tf::zip(_contained_points, base_t::points());
    auto n_kept = tf::remove_if_and_make_map(
                      r, [](auto &&pair) { return !std::get<0>(pair); }, map) -
                  r.begin();
    base_t::points_buffer().erase(base_t::points_buffer().begin() + n_kept,
                                  base_t::points_buffer().end());
    tf::parallel_for_each(
        base_t::edges(),
        [&](auto &&edge) {
          edge[0] = map[edge[0]];
          edge[1] = map[edge[1]];
        },
        tf::checked);
  }

  template <typename Policy>
  auto make_initial_points(const tf::segments<Policy> &segments,
                           RealT tolerance) {
    base_t::points_buffer().allocate(segments.size() * 2);
    auto points = base_t::points();
    tf::parallel_for_each(tf::zip(segments, tf::make_blocked_range<2>(points)),
                       [](auto pair) {
                         auto &&[_in, _out] = pair;
                         _out[0] = _in[0];
                         _out[1] = _in[1];
                       });
    make_initial_points_impl(tolerance);
  }

  auto make_initial_points_impl() {
    auto points = base_t::points();
    tf::make_unique_and_index_map(points, _im);
    base_t::points_buffer().data_buffer().erase_till_end(
        base_t::points_buffer().data_buffer().begin() +
        _im.kept_ids().size() * Dims);
  }

  auto make_initial_points_impl(RealT tolerance) {
    if (tolerance <= 0)
      return make_initial_points_impl();
    auto points = base_t::points();
    tf::make_clean_index_map(points, tolerance, _im);
    tf::points_buffer<RealT, Dims> tmp_buffer;
    tmp_buffer.allocate(_im.kept_ids().size());
    tf::parallel_copy(tf::make_indirect_range(_im.kept_ids(), points),
                      tmp_buffer);
    base_t::points_buffer() = std::move(tmp_buffer);
  }

  auto make_initial_edges() {
    base_t::edges_buffer().data_buffer().allocate(_im.f().size());
    tf::parallel_for_each(
        tf::zip(tf::make_blocked_range<2>(_im.f()), base_t::edges_buffer()),
        [](auto pair) {
          auto &&[_in, _out] = pair;
          _out[0] = _in[0];
          _out[1] = _in[1];
          if (_out[0] > _out[1])
            std::swap(_out[0], _out[1]);
          // Mark edges with equal indices as invalid
          if (_out[0] == _out[1]) {
            _out[0] = std::numeric_limits<Index>::max();
            _out[1] = std::numeric_limits<Index>::max();
          }
        });
    tbb::parallel_sort(base_t::edges_buffer());
    auto n_unique = (std::unique(base_t::edges_buffer().begin(),
                                 base_t::edges_buffer().end()) -
                     base_t::edges_buffer().begin()) *
                    2;
    base_t::edges_buffer().data_buffer().erase(
        base_t::edges_buffer().data_buffer().begin() + n_unique,
        base_t::edges_buffer().data_buffer().end());
    // if the last surviving edge is the (invalid, invalid) sentinel, drop it
    if (auto &edges = base_t::edges_buffer(); edges.size()) {
      const auto &last = edges.back();
      if (last[0] == last[1]) {
        edges.data_buffer().erase(edges.data_buffer().end() - 2,
                                  edges.data_buffer().end());
      }
    }
  }

  tf::index_map_buffer<Index> _im;
  tf::buffer<bool> _contained_points;
};
} // namespace tf::clean
