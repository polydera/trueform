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
#include "../core/polygon.hpp"
#include "../exact/meta.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/signed_area.hpp"
#include "./face_paths_extractor.hpp"

namespace tf {

/// @ingroup topology_planar
/// @brief Classifies the paths an edge set draws across a face loop.
///
/// @tparam Int The lattice the exact predicates run on. `build` accepts any
///   coordinate type, so the caller states it.
template <typename Index, typename Int> class face_splitting_paths {
  using T2 = typename tf::exact::meta<Int>::T2;

public:
  struct path_descriptor {
    Index start, end;
    T2 signed_area;
  };

  template <typename Range, typename Policy0, typename Policy1>
  auto build(const Range &base_loop, const tf::edges<Policy0> &edges,
             const tf::points<Policy1> &points) {
    using coord_t = tf::coordinate_type<Policy1>;
    if constexpr (std::is_integral_v<coord_t>) {
      build_impl(base_loop, edges, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      build_impl(base_loop, edges, int_pts);
    }
  }

  auto clear() {
    _extractor.clear();
    _crossing_paths.clear();
    _non_crossing_paths.clear();
    _loops.clear();
    _cuts.clear();
    _descriptors.clear();
  }

  auto crossing_paths() const {
    return tf::make_indirect_range(_crossing_paths, _extractor.paths());
  }

  auto crossing_path_descriptors() const {
    return tf::make_indirect_range(_crossing_paths, _descriptors);
  }
  auto crossing_path_descriptors() {
    return tf::make_indirect_range(_crossing_paths, _descriptors);
  }

  auto loop_paths() const {
    return tf::make_indirect_range(_loops, _extractor.paths());
  }

  auto non_crossing_paths() const {
    return tf::make_indirect_range(_non_crossing_paths, _extractor.paths());
  }

  auto cut_paths() const {
    return tf::make_indirect_range(_cuts, _extractor.paths());
  }

  auto paths() const { return _extractor.paths(); }
  auto paths() { return _extractor.paths(); }

private:
  template <typename Range, typename Policy0, typename Policy1>
  auto build_impl(const Range &base_loop, const tf::edges<Policy0> &edges,
                  const tf::points<Policy1> &points) {
    clear();
    _extractor.build(base_loop, edges, points.size());
    classify(points);
    order_crossing_paths();
  }

  template <typename Policy> auto classify(const tf::points<Policy> &points) {
    for (auto &&[i, path] : tf::enumerate(_extractor.paths())) {
      auto start = path.front();
      auto end = path.back();
      auto start_id = _extractor.id_on_loop(start);
      auto end_id = _extractor.id_on_loop(end);
      auto area = tf::exact::signed_area_2x(tf::make_polygon(path, points));

      if (start == end) {
        _loops.push_back(i);
        if (area > 0) {
          std::reverse(path.begin(), path.end());
          area = -area;
        }

      } else if (_extractor.is_on_loop(start) && _extractor.is_on_loop(end)) {
        _crossing_paths.push_back(i);
        // canonical orientation: start_id < end_id
        if (end_id < start_id) {
          std::swap(start_id, end_id);
          std::reverse(path.begin(), path.end());
          area = -area;
        }

      } else if ((_extractor.is_on_loop(start) ||
                  _extractor.is_cut_endpoint(start)) &&
                 (_extractor.is_on_loop(end) ||
                  _extractor.is_cut_endpoint(end))) {
        _cuts.push_back(i);

      } else {
        _non_crossing_paths.push_back(i);
      }
      _descriptors.push_back({start_id, end_id, area});
    }
  }

  auto order_crossing_paths() {
    std::sort(_crossing_paths.begin(), _crossing_paths.end(),
              [this](Index i0, Index i1) {
                auto &d0 = _descriptors[i0];
                auto &d1 = _descriptors[i1];
                if (d0.start != d1.start)
                  return d0.start < d1.start;
                if (d0.end != d1.end)
                  return d0.end > d1.end;               // biggest end first
                return d0.signed_area < d1.signed_area; // most negative first
              });
  }

  tf::face_paths_extractor<Index> _extractor;
  tf::buffer<path_descriptor> _descriptors;
  tf::buffer<Index> _crossing_paths;
  tf::buffer<Index> _non_crossing_paths;
  tf::buffer<Index> _loops;
  tf::buffer<Index> _cuts;
};
} // namespace tf
