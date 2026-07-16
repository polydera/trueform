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
#include "../core/coordinate_type.hpp"
#include "../core/faces.hpp"
#include "../core/points.hpp"
#include "../core/views/blocked_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/meta.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/signed_area.hpp"
#include "./face_hole_relations.hpp"
#include "./face_splitting_paths.hpp"
#include "./planar_graph_regions.hpp"

namespace tf {

/// @ingroup topology_planar
/// @brief Subdivides a face by a set of edges using exact arithmetic.
///
/// Given a face boundary and interior edges, classifies the resulting
/// paths (crossings, loops, cuts, non-crossings) and extracts all sub-faces
/// and holes. Uses exact int32 arithmetic for all geometric predicates.
/// Float points are converted via pt_converter.
template <typename Index, typename Int = tf::exact::int32>
class face_split_by_edges {
private:
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

public:
  template <typename Range, typename Policy0, typename Policy1>
  auto build(const Range &face, const tf::edges<Policy0> &edges,
             const tf::points<Policy1> &points) {
    clear();
    using coord_t = tf::coordinate_type<Policy1>;
    if constexpr (std::is_integral_v<coord_t>) {
      build_impl(face, edges, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      build_impl(face, edges, int_pts);
    }
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

  auto face_splitting_paths() const
      -> const tf::face_splitting_paths<Index, Int> & {
    return _spaths;
  }

  /// The directed edge soup fed to planar_graph_regions, and the raw regions
  /// it traced (before the exterior is discarded).
  auto work_edges() const { return tf::make_range(_work_edges); }
  auto planar_regions() const -> const tf::planar_graph_regions<Index, Int> & {
    return _pgr;
  }

  auto clear() {
    _pgr.clear();
    _spaths.clear();
    _fhr.clear();
    _work_edges.clear();
    _faces.clear();
    _holes.clear();
    _signed_areas.clear();
    _vertices.clear();
    _offsets.clear();
  }

private:
  template <typename Range, typename Policy>
  auto area_of(const Range &loop, const tf::points<Policy> &points) {
    return tf::exact::signed_area_2x(tf::make_polygon(loop, points));
  }

  auto all_loops() const {
    return tf::make_offset_block_range(_offsets, _vertices);
  }

  /// Emit a single loop as a face into the output buffers.
  template <typename Range, typename Policy>
  auto emit_face(const Range &loop, const tf::points<Policy> &points) {
    auto a = area_of(loop, points);
    auto id = Index(_signed_areas.size());
    _signed_areas.push_back(a);
    _offsets.push_back(_vertices.size());
    std::copy(loop.begin(), loop.end(), std::back_inserter(_vertices));
    _faces.push_back(id);
  }

  /// Divide base loop by crossing paths and emit each sub-region as a face.
  /// Used when there are no non-crossing paths.
  template <typename Range, typename Policy>
  auto emit_crossing_faces(const Range &base_loop,
                           const tf::points<Policy> &points) {
    const auto &crossings = _spaths.crossing_paths();
    if (!crossings.size()) {
      emit_face(base_loop, points);
      return;
    }
    const auto &descriptors = _spaths.crossing_path_descriptors();
    std::array<const Index, 2> left_over{base_loop.front(), base_loop.back()};
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
                  std::back_inserter(_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_vertices));
      } else {
        std::copy(base_loop.begin() + start, base_loop.begin() + end,
                  std::back_inserter(_vertices));
        std::reverse_copy(path.begin() + 1, path.end(),
                          std::back_inserter(_vertices));
      }
      // Compute area for this sub-loop and register as face
      auto loop_begin = _vertices.begin() + _offsets.back();
      auto loop_range = tf::make_range(loop_begin, _vertices.end());
      auto a = area_of(loop_range, points);
      auto id = Index(_signed_areas.size());
      _signed_areas.push_back(a);
      _faces.push_back(id);
    }
  }

  template <typename Policy>
  auto copy_from_planar_regions(const tf::points<Policy> &points) {
    // find exterior: most negative signed_area_2x
    T2 min_area = 0;
    Index min_id = -1;
    Index count = 0;
    for (const auto &region : _pgr) {
      auto a = area_of(region, points);
      if (a < min_area) {
        min_area = a;
        min_id = count;
      }
      ++count;
    }
    // Emit the rest: positive regions are faces, everything else is a
    // hole. A NEGATIVE region that is not the exterior is the outer walk
    // of an interior chord cycle — a hole in its surrounding region (the
    // nesting pass places it); emitting it as a face would duplicate the
    // cycle's cell with inverted winding. A ZERO-AREA region is a slit
    // walk (an isolated chord tree walked out-and-back, or the sandwich
    // between duplicate chains) — the same shape cut paths emit: a
    // zero-area hole, so the triangulation imprints its edges. As a face
    // it would triangulate to nothing and lose the constraints.
    count = 0;
    for (const auto &region : _pgr) {
      if (min_id == count++) {
        continue;
      }
      auto a = area_of(region, points);
      auto id = Index(_signed_areas.size());
      _signed_areas.push_back(a);
      _offsets.push_back(_vertices.size());
      std::copy(region.begin(), region.end(), std::back_inserter(_vertices));
      if (a > 0)
        _faces.push_back(id);
      else
        _holes.push_back(id);
    }
  }

  /// Build work_edges from all crossing + non-crossing paths + base loop.
  template <typename Range>
  auto fill_edges_for_graph_regions(const Range &face) {
    _work_edges.clear();
    auto add_paths = [&](const auto &paths) {
      for (const auto &path : paths)
        for (auto [a, b] : tf::make_slide_range<2>(path)) {
          _work_edges.push_back(a);
          _work_edges.push_back(b);
          _work_edges.push_back(b);
          _work_edges.push_back(a);
        }
    };
    add_paths(_spaths.crossing_paths());
    add_paths(_spaths.non_crossing_paths());
    Index size = face.size();
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
      _work_edges.push_back(face[prev]);
      _work_edges.push_back(face[i]);
      _work_edges.push_back(face[i]);
      _work_edges.push_back(face[prev]);
    }
  }

  /// Run planar_graph_regions on all edges (crossings + non-crossings + base loop).
  template <typename Range, typename Policy>
  auto emit_as_graph_regions(const Range &face,
                             const tf::points<Policy> &points) {
    fill_edges_for_graph_regions(face);
    _pgr.build(tf::make_edges(tf::make_blocked_range<2>(_work_edges)), points);
    copy_from_planar_regions(points);
  }

  auto process_cuts() {
    for (const auto &cut : _spaths.cut_paths()) {
      auto id = Index(_signed_areas.size());
      _signed_areas.push_back(T2(0));
      _offsets.push_back(_vertices.size());
      std::copy(cut.begin(), cut.end(), std::back_inserter(_vertices));
      if (cut.size() > 2)
        std::reverse_copy(cut.begin() + 1, cut.end() - 1,
                          std::back_inserter(_vertices));
      _holes.push_back(id);
    }
  }

  template <typename Policy>
  auto process_loop_paths(const tf::points<Policy> &points) {
    for (const auto &_loop : _spaths.loop_paths()) {
      auto loop = tf::make_range(_loop.begin(), _loop.size() - 1);
      auto a = area_of(loop, points);
      // hole (original direction, negative area)
      auto hole_id = Index(_signed_areas.size());
      _signed_areas.push_back(a);
      _offsets.push_back(_vertices.size());
      std::copy(loop.begin(), loop.end(), std::back_inserter(_vertices));
      _holes.push_back(hole_id);
      // face (reversed direction, positive area)
      auto face_id = Index(_signed_areas.size());
      _signed_areas.push_back(-a);
      _offsets.push_back(_vertices.size());
      std::reverse_copy(loop.begin(), loop.end(),
                        std::back_inserter(_vertices));
      _faces.push_back(face_id);
    }
  }

  template <typename Range, typename Policy>
  auto process_paths(const Range &base_loop, const tf::points<Policy> &points) {
    if (_spaths.non_crossing_paths().size()) {
      emit_as_graph_regions(base_loop, points);
    } else {
      emit_crossing_faces(base_loop, points);
    }
    process_cuts();
    process_loop_paths(points);
    if (_vertices.size())
      _offsets.push_back(_vertices.size());
  }

  template <typename Range, typename Policy0, typename Policy1>
  auto build_impl(const Range &face, const tf::edges<Policy0> &edges,
                  const tf::points<Policy1> &points) {
    _spaths.build(face, edges, points);
    process_paths(face, points);
    _fhr.build(tf::make_faces(faces()), tf::make_faces(holes()), points);
  }

  tf::planar_graph_regions<Index, Int> _pgr;
  tf::face_splitting_paths<Index, Int> _spaths;
  tf::face_hole_relations<Index, Int> _fhr;
  tf::buffer<Index> _work_edges;
  tf::buffer<Index> _faces;
  tf::buffer<Index> _holes;
  tf::buffer<T2> _signed_areas;
  tf::buffer<Index> _vertices;
  tf::buffer<Index> _offsets;
};
} // namespace tf
