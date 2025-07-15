/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/area.hpp"
#include "./path_extractor.hpp"

namespace tf::loop {
template <typename Index, typename RealType> class splitting_paths {
public:
  template <typename Range0, typename Range1, typename Range2>
  auto build(const Range0 &base_loop, const Range1 &edges,
             const Range2 &points) {
    clear();
    _extractor.build(base_loop, edges, points.size());
    build_path_categories(points);
  }

  auto clear() {
    _extractor.clear();
    _crossing_paths.clear();
    _non_crossing_paths.clear();
    _loops.clear();
    _cuts.clear();
    _descriptors.clear();
  }

private:
  template <typename Range> auto build_path_categories(const Range &points) {
    for (auto &&[i, path] : tf::enumerate(_extractor.paths())) {
      auto start = path.front();
      auto end = path.back();
      auto start_id = _extractor.id_on_loop(start);
      auto end_id = _extractor.id_on_loop(end);
      auto area = tf::signed_area(tf::make_polygon(path, points));
      if (_extractor.is_on_loop(start) && _extractor.is_on_loop(end)) {
        _crossing_paths.push_back(i);
        // make canonical orientation
        if (end_id < start_id) {
          std::swap(start_id, end_id);
          std::reverse(path.begin(), path.end());
          area *= -1;
        } else if (start_id == end_id && area > 0) {
          std::reverse(path.begin(), path.end());
          area *= -1;
        }

      } else if (start == end) {
        _loops.push_back(i);
        if (area > 0) {
          std::reverse(path.begin(), path.end());
          area *= -1;
        }
      } else if (_extractor.is_cut_endpoint(start) &&
                 _extractor.is_cut_endpoint(end))
        _cuts.push_back(i);
      else
        _non_crossing_paths.push_back(i);
      _descriptors.push_back({start_id, end_id, area});
    }
  }
  auto order_crossing_paths() {
    /*
     * We order the paths as they rotate around the base loop.
     * i.e. by smallest starting and biggest ending point.
     *
     * If multiple paths begin and end in the same point, they
     * must be ordered by decreasing area. This way the next
     * crossing is always contained in the previous one
     * (for those begining and ending in the same point).
     */
    std::sort(_crossing_paths.begin(), _crossing_paths.end(),
              [this](Index i0, Index i1) {
                return std::make_tuple(_descriptors[i0].start_id,
                                       -_descriptors[i0].end_id,
                                       _descriptors[i0].signed_area) <
                       std::make_tuple(_descriptors[i1].start_id,
                                       -_descriptors[i1].end_id,
                                       _descriptors[i1].signed_area);
              });
  }
  struct path_descriptor {
    Index start;
    Index end;
    RealType signed_area;
  };
  tf::loop::path_extractor<Index> _extractor;
  tf::buffer<path_descriptor> _descriptors;
  // path type ids
  tf::buffer<Index> _crossing_paths;
  tf::buffer<Index> _non_crossing_paths;
  tf::buffer<Index> _loops;
  tf::buffer<Index> _cuts;
};

} // namespace tf::loop
