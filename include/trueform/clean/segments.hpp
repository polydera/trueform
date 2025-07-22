/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/make_unique_index_map.hpp"
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/views/blocked_range.hpp"
#include "../core/views/zip.hpp"

namespace tf {
template <typename Index, typename RealT, std::size_t Dims>
class cleaned_segments : public segments_buffer<Index, RealT, Dims> {
  using base_t = segments_buffer<Index, RealT, Dims>;

public:
  template <typename Policy> auto build(const tf::segments<Policy> &segments) {
    clear();
    make_initial_points(segments);
    make_initial_edges();
  }

  auto clear() {
    base_t::clear();
    _im.f().clear();
    _im.kept_ids().clear();
  }

private:
  template <typename Policy>
  auto make_initial_points(const tf::segments<Policy> &segments) {
    base_t::raw_points_buffer().allocate(segments.size() * Dims * 2);
    auto points = tf::make_points<Dims>(base_t::raw_points_buffer());
    tf::parallel_apply(tf::zip(segments, tf::make_blocked_range<2>(points)),
                       [](auto pair) {
                         auto &&[_in, _out] = pair;
                         _out[0] = _in[0];
                         _out[1] = _in[1];
                       });

    tf::make_unique_and_index_map(points, _im);
    base_t::raw_points_buffer().erase_till_end(
        base_t::raw_points_buffer().begin() + _im.kept_ids().size() * Dims);
  }

  auto make_initial_edges() {
    base_t::raw_edges_buffer().allocate(_im.f().size());
    tf::parallel_apply(
        tf::zip(tf::make_blocked_range<2>(_im.f()),
                tf::make_blocked_range<2>(base_t::raw_edges_buffer())),
        [](auto pair) {
          auto &&[_in, _out] = pair;
          _out[0] = _in[0];
          _out[1] = _in[1];
          if (_out[0] > _out[1])
            std::swap(_out[0], _out[1]);
        });
    auto edges_as_points = tf::make_points<2>(base_t::raw_edges_buffer());
    tbb::parallel_sort(edges_as_points);
    auto n_unique =
        (std::unique(edges_as_points.begin(), edges_as_points.end()) -
         edges_as_points.begin()) *
        2;
    base_t::raw_edges_buffer().erase(base_t::raw_edges_buffer().begin() +
                                         n_unique,
                                     base_t::raw_edges_buffer().end());
  }

  tf::index_map_buffer<Index> _im;
};
} // namespace tf
