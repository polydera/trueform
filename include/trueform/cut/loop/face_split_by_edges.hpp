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
#include "../../core/faces.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/search.hpp"
#include "../../topology/face_hole_relations.hpp"
#include "../../topology/planar_graph_regions.hpp"
#include "./splitting_paths.hpp"

namespace tf::loop {
template <typename Index, typename RealType> class face_split_by_edges {
public:
  template <typename Range0, typename Policy0, typename Policy1>
  auto build(const Range0 &face, const tf::edges<Policy0> &edges,
             const tf::points<Policy1> &points) {
    clear();
    _spaths.build(face, edges, points);
    process_paths(face, points);
    assign_categories();
    _fhr.build(tf::make_faces(faces()), face_areas(), tf::make_faces(holes()),
               points);
  }

  auto faces() const { return tf::make_indirect_range(_faces, all_loops()); }

  auto face_areas() const {
    return tf::make_indirect_range(_faces, _signed_areas);
  }

  auto holes() const { return tf::make_indirect_range(_holes, all_loops()); }

  auto hole_areas() const {
    return tf::make_indirect_range(_holes, _signed_areas);
  }

  auto holes_for_faces() const {
    return tf::make_offset_block_range(_fhr.offsets_buffer(),
                                       _fhr.data_buffer());
  }

  auto clear() {
    _pgr.clear();
    _spaths.clear();
    _fhr.clear();
    _tree.clear();
    _base_loops_vertices.clear();
    _base_loops_offsets.clear();
    _work_edges.clear();
    _non_crossing_in_base_loop.clear();
    _faces.clear();
    _holes.clear();
    _signed_areas.clear();
    _vertices.clear();
    _offsets.clear();
  }

private:
  template <typename Range>
  auto divide_base_loop_with_crossing_paths(const Range &base_loop) {
    const auto &crossings = _spaths.crossing_paths();
    if (!crossings.size()) {
      _base_loops_offsets.push_back(0);
      _base_loops_vertices.allocate(base_loop.size());
      std::copy(base_loop.begin(), base_loop.end(),
                _base_loops_vertices.begin());
      _base_loops_offsets.push_back(_base_loops_vertices.size());
      return;
    }
    const auto &descriptors = _spaths.crossing_path_descriptors();
    std::array<const Index, 2> left_over{base_loop.front(), base_loop.back()};
    // so that we can start with the "left_over", making
    // the algorithm easier
    auto get = [&](Index i) {
      if (i == -1)
        return std::make_tuple(tf::make_range(left_over.data(), 2), Index(0),
                               Index(base_loop.size() - 1));
      else
        return std::make_tuple(crossings[i], descriptors[i].start,
                               descriptors[i].end);
    };
    Index n_crossings = crossings.size();
    Index last = n_crossings - 1;
    for (Index i = -1; i < Index(crossings.size()); ++i) {
      _base_loops_offsets.push_back(_base_loops_vertices.size());
      auto [path, start, end] = get(i);
      // nested crossings
      if (i != last && descriptors[i + 1].end <= end) {
        Index current = start;
        Index next = i + 1;
        while (current != end) {
          std::copy(base_loop.begin() + current,
                    base_loop.begin() + descriptors[next].start,
                    std::back_inserter(_base_loops_vertices));
          std::copy(crossings[next].begin(), crossings[next].end() - 1,
                    std::back_inserter(_base_loops_vertices));
          current = descriptors[next].end;
          next = std::find_if(descriptors.begin() + next + 1, descriptors.end(),
                              [&, outer_end = end](const auto &d) {
                                return d.start >= current && d.end <= outer_end;
                              }) -
                 descriptors.begin();
          if (next == n_crossings || descriptors[next].start >= end) {
            break;
          }
        }
        std::copy(base_loop.begin() + current, base_loop.begin() + end,
                  std::back_inserter(_base_loops_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_base_loops_vertices));
      } else {
        std::copy(base_loop.begin() + start, base_loop.begin() + end,
                  std::back_inserter(_base_loops_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_base_loops_vertices));
      }
    }
    if (_base_loops_vertices.size())
      _base_loops_offsets.push_back(_base_loops_vertices.size());
  }

  auto base_loops() const {
    return tf::make_offset_block_range(_base_loops_offsets,
                                       _base_loops_vertices);
  }

  auto all_loops() const {
    return tf::make_offset_block_range(_offsets, _vertices);
  }

  template <typename Policy>
  auto assign_base_loop_to_non_crossings(const tf::points<Policy> &points) {
    auto polygons = tf::make_polygons(base_loops(), points);
    _tree.build(polygons, tf::config_tree(4, 4));
    _non_crossing_in_base_loop.allocate(_spaths.non_crossing_paths().size());
    for (auto &&[path, in_base_loop] :
         tf::zip(_spaths.non_crossing_paths(), _non_crossing_in_base_loop)) {
      in_base_loop = -1;
      auto point = points[path[1]];
      tf::search(_tree, tf::intersects_f(point),
                 [&, &in_base_loop = in_base_loop](Index b_id) {
                   if (tf::contains_coplanar_point(polygons[b_id], point)) {
                     in_base_loop = b_id;
                     return true;
                   } else
                     return false;
                 });
    }
  }

  template <typename Range>
  auto fill_for_base_loop(Index i, const Range &base_loop) {
    _work_edges.clear();
    for (auto [path, b_i] :
         tf::zip(_spaths.non_crossing_paths(), _non_crossing_in_base_loop)) {
      if (i == b_i) {
        for (auto [a, b] : tf::make_slide_range<2>(path)) {
          _work_edges.push_back(a);
          _work_edges.push_back(b);
          _work_edges.push_back(b);
          _work_edges.push_back(a);
        }
      }
    }
    if (!_work_edges.size())
      return false;
    Index size = base_loop.size();
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
      _work_edges.push_back(base_loop[prev]);
      _work_edges.push_back(base_loop[i]);
      _work_edges.push_back(base_loop[i]);
      _work_edges.push_back(base_loop[prev]);
    }
    return true;
  }

  template <typename Policy>
  auto copy_from_planar_regions(const tf::points<Policy> &points) {

    auto is_empty = [&](const auto &region) -> bool {
      const int n = int(region.size());
      if (n < 3)
        return true;
      if (region[1] != region[n - 1])
        return false;

      int i = 2;
      int j = n - 1;

      while (i < j && region[j] == region[i - 1] &&
             region[j - 1] == region[i]) {
        ++i;
        --j;
      }

      const int remaining = j - (i - 1) + 1;
      return remaining < 3;
    };

    Index area_offset = _signed_areas.size();
    Index min_area_id = -1;
    RealType min_area = std::numeric_limits<RealType>::max();
    // first accumulate areas
    for (const auto &region : _pgr) {
      auto sa =
          is_empty(region)
              ? RealType(0)
              : RealType(tf::signed_area(tf::make_polygon(region, points)));
      if (sa < 0 && sa < min_area) {
        min_area = sa;
        min_area_id = _signed_areas.size();
      }
      _signed_areas.push_back(sa);
    }
    // the smallest signed area is the unbounded
    // plane, which we remove
    Index count = area_offset;
    for (const auto &[region, sa] :
         tf::zip(_pgr, tf::drop(_signed_areas, area_offset))) {
      if (min_area_id == count++)
        continue;
      _signed_areas[area_offset++] = sa;
      _offsets.push_back(_vertices.size());
      std::copy(region.begin(), region.end(), std::back_inserter(_vertices));
    }
    if (min_area_id != -1)
      _signed_areas.pop_back();
  }

  template <typename Policy>
  auto process_non_crossings(const tf::points<Policy> &points) {
    assign_base_loop_to_non_crossings(points);
    for (auto [i, base_loop] : tf::enumerate(base_loops())) {
      /*std::cout << "base loop: " << i << "\n" << std::endl;*/
      /*for(auto e:base_loop)*/
      /*  std::cout << e << ", ";*/
      /*std::cout << std::endl;*/
      if (fill_for_base_loop(i, base_loop)) {
        /*std::cout << "  edges:" << std::endl;*/
        /*for(auto [a, b]:tf::make_blocked_range<2>(_work_edges))*/
        /*  std::cout << "    " << a << ", " << b << std::endl;*/
        _pgr.build(tf::make_edges(tf::make_blocked_range<2>(_work_edges)),
                   points);
        copy_from_planar_regions(points);
      } else { // no crossings in here
        _offsets.push_back(_vertices.size());
        std::copy(base_loop.begin(), base_loop.end(),
                  std::back_inserter(_vertices));
        _signed_areas.push_back(
            tf::signed_area(tf::make_polygon(base_loop, points)));
      }
    }
  }

  auto process_cuts() {
    for (const auto &cut : _spaths.cut_paths()) {
      _signed_areas.push_back(0);
      _offsets.push_back(_vertices.size());
      std::copy(cut.begin(), cut.end(), std::back_inserter(_vertices));
      if (cut.size() > 2)
        std::reverse_copy(cut.begin() + 1, cut.end() - 1,
                          std::back_inserter(_vertices));
    }
  }

  template <typename Policy>
  auto process_loop_paths(const tf::points<Policy> &points) {
    for (const auto &_loop : _spaths.loop_paths()) {
      auto loop = tf::make_range(_loop.begin(), _loop.size() - 1);
      auto sa = tf::signed_area(tf::make_polygon(loop, points));
      // Emit hole (original direction, negative area)
      _signed_areas.push_back(sa);
      _offsets.push_back(_vertices.size());
      std::copy(loop.begin(), loop.end(), std::back_inserter(_vertices));
      // Emit face (reversed direction, positive area)
      _signed_areas.push_back(-sa);
      _offsets.push_back(_vertices.size());
      std::reverse_copy(loop.begin(), loop.end(),
                        std::back_inserter(_vertices));
    }
  }

  template <typename Range, typename Policy1>
  auto process_paths(const Range &base_loop,
                     const tf::points<Policy1> &points) {
    divide_base_loop_with_crossing_paths(base_loop);
    if (_spaths.non_crossing_paths().size()) {
      process_non_crossings(points);
    } else {
      for (const auto &base_loop : base_loops()) {
        _offsets.push_back(_vertices.size());
        std::copy(base_loop.begin(), base_loop.end(),
                  std::back_inserter(_vertices));
        _signed_areas.push_back(
            tf::signed_area(tf::make_polygon(base_loop, points)));
      }
    }
    process_cuts();
    process_loop_paths(points);
    if (_vertices.size())
      _offsets.push_back(_vertices.size());
  }

  auto assign_categories() {
    for (auto [id, sa] : tf::enumerate(_signed_areas))
      if (sa > 0)
        _faces.push_back(id);
      else
        _holes.push_back(id);
  }

  tf::planar_graph_regions<Index, RealType> _pgr;
  tf::loop::splitting_paths<Index, RealType> _spaths;
  tf::face_hole_relations<Index, RealType> _fhr;
  tf::aabb_tree<Index, RealType, 2> _tree;
  tf::buffer<Index> _base_loops_vertices;
  tf::buffer<Index> _base_loops_offsets;
  tf::buffer<Index> _work_edges;
  tf::buffer<Index> _non_crossing_in_base_loop;
  tf::buffer<Index> _faces;
  tf::buffer<Index> _holes;
  tf::buffer<RealType> _signed_areas;
  tf::buffer<Index> _vertices;
  tf::buffer<Index> _offsets;
};
} // namespace tf::loop
