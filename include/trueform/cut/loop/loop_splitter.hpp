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

#include "./splitting_paths.hpp"

namespace tf::loop {
template <typename Index, typename RealType> class loop_splitter {
public:
  template <typename Range0, typename Range1, typename Range2>
  auto build(const Range0 &base_loop, const Range1 &edges,
             const Range2 &points) {
    clear();
    _spaths.build(base_loop, edges, points);
    divide_base_loop_with_crossing_paths(base_loop);
  }

  auto clear() {
    _spaths.clear();
    _vertices.clear();
    _offsets.clear();
  }

private:
  template <typename Range>
  auto divide_base_loop_with_crossing_paths(const Range &base_loop) {
    const auto &crossings = _spaths.crossing_paths();
    if (!crossings.size())
      return;
    const auto &descriptors = _spaths.crossing_path_descriptors();
    std::array<Index, 2> left_over{base_loop.front(), base_loop.back()};
    // so that we can start with the "left_over", making
    // the algorithm easier
    auto get = [&](Index i) {
      if (i == -1)
        return std::make_tuple(tf::make_range(left_over.cbegin(), 2), Index(0),
                               Index(base_loop.size() - 1));
      else
        return std::make_tuple(crossings[i], descriptors[i].start,
                               descriptors[i].end);
    };
    Index n_crossings = crossings.size();
    Index last = n_crossings - 1;
    for (Index i = -1; i < Index(crossings.size()); ++i) {
      _offsets.push_back(_vertices.size());
      auto [path, start, end] = get(i);
      // nested crossings
      if (i != last && descriptors[i + 1].end <= end) {
        Index current = start;
        Index next = i + 1;
        while (current != end) {
          std::copy(base_loop.begin() + current,
                    base_loop.begin() + descriptors[next].start,
                    std::back_inserter(_vertices));
          std::copy(crossings[next].begin(), crossings[next].end() - 1,
                    std::back_inserter(_vertices));
          current = descriptors[next].end;
          next = std::find_if(
                     descriptors.begin() + next + 1, descriptors.end(),
                     [&](const auto &d) { return !(d.start < current); }) -
                 descriptors.begin();
          if (next == n_crossings || descriptors[next].start >= current) {
            break;
          }
        }
        std::copy(base_loop.begin() + current, base_loop.begin() + end,
                  std::back_inserter(_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_vertices));
      } else {
        std::copy(base_loop.begin() + start, base_loop.begin() + end,
                  std::back_inserter(_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_vertices));
      }
    }
    if (_vertices.size())
      _offsets.push_back(_vertices.size());
  }

private:
  tf::loop::splitting_paths<Index, RealType> _spaths;
  tf::buffer<Index> _vertices;
  tf::buffer<Index> _offsets;
};
} // namespace tf::loop
