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

#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/contiguous_index_hash_map.hpp"
#include "../core/index_hash_map.hpp"
#include "../core/points_buffer.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/face_split_by_edges.hpp"
#include "../topology/hole_patcher.hpp"

namespace tf {

/// Splits a single face by its intersection edges.
///
/// Takes a base loop and edge definitions from intersection_graph,
/// remaps to contiguous indices, calls face_split_by_edges, and maps
/// output back to graph::vertex<Index>.
///
/// get_point must return already-projected point<Int, 2>.
///
/// Fast path: 0 edges emits base loop; 1 edge with both endpoints on
/// the loop splits into 2 sub-loops without invoking face_split_by_edges.
template <typename Index, typename Int = tf::exact::int32> class face_cutter {
  using vertex_t = intersect::graph::vertex<Index>;
  using source = intersect::graph::vertex_source;

public:
  /// get_point(graph::vertex<Index>) → point<Int, 2> (already projected)
  template <typename Loop, typename EdgeDefs, typename GetPoint>
  auto build(const Loop &loop, const EdgeDefs &edge_defs,
             const GetPoint &get_point, tf::buffer<Index> &offsets,
             tf::buffer<vertex_t> &vertices) -> Index {
    if (edge_defs.size() == 0) {
      emit_loop(loop, offsets, vertices);
      return 1;
    }

    if (edge_defs.size() == 1) {
      auto &&e = edge_defs[0];
      auto [p0, p1] = find_pair_in_loop(loop, e[0], e[1]);
      if (p0 >= 0 && p1 >= 0) {
        split_at(loop, p0, p1, offsets, vertices);
        return 2;
      }
    }

    return build_generic(loop, edge_defs, get_point, offsets, vertices);
  }

  /// Structure-emitting build: region boundary walks plus hole walks,
  /// no hole patching. Per emitted loop one entry in `lh_offsets`
  /// points into `lh_ids`, whose ids index the
  /// (`hole_offsets`, `hole_vertices`) pool. Fast paths emit hole-free
  /// loops. Returns the number of emitted loops.
  template <typename Loop, typename EdgeDefs, typename GetPoint>
  auto build_regions(const Loop &loop, const EdgeDefs &edge_defs,
                     const GetPoint &get_point, tf::buffer<Index> &offsets,
                     tf::buffer<vertex_t> &vertices,
                     tf::buffer<Index> &hole_offsets,
                     tf::buffer<vertex_t> &hole_vertices,
                     tf::buffer<Index> &lh_offsets, tf::buffer<Index> &lh_ids)
      -> Index {
    if (edge_defs.size() == 0) {
      emit_loop(loop, offsets, vertices);
      lh_offsets.push_back(static_cast<Index>(lh_ids.size()));
      return 1;
    }

    if (edge_defs.size() == 1) {
      auto &&e = edge_defs[0];
      auto [p0, p1] = find_pair_in_loop(loop, e[0], e[1]);
      if (p0 >= 0 && p1 >= 0) {
        split_at(loop, p0, p1, offsets, vertices);
        lh_offsets.push_back(static_cast<Index>(lh_ids.size()));
        lh_offsets.push_back(static_cast<Index>(lh_ids.size()));
        return 2;
      }
    }

    prepare_and_split(loop, edge_defs, get_point);

    if (_fs.faces().size() == 0) {
      write_face(_base_loop, offsets, vertices);
      lh_offsets.push_back(static_cast<Index>(lh_ids.size()));
      return 1;
    }

    const bool with_holes = _fs.holes().size() != 0;
    Index k = 0;
    for (const auto &face : _fs.faces()) {
      write_face(face, offsets, vertices);
      lh_offsets.push_back(static_cast<Index>(lh_ids.size()));
      if (with_holes)
        for (auto h : _fs.holes_for_faces()[std::size_t(k)]) {
          lh_ids.push_back(static_cast<Index>(hole_offsets.size()));
          write_face(_fs.holes()[h], hole_offsets, hole_vertices);
        }
      ++k;
    }
    return static_cast<Index>(_fs.faces().size());
  }

  auto splitter() const -> const tf::face_split_by_edges<Index, Int> & {
    return _fs;
  }
  auto index_hash_map() const -> const auto & { return _ihm; }

  auto clear() -> void {
    _ihm.clear();
    _fs.clear();
    _hp.clear();
    _edges.clear();
    _base_loop.clear();
    _points.clear();
  }

private:
  template <typename Loop>
  static auto find_pair_in_loop(const Loop &loop, Index id0, Index id1)
      -> std::pair<Index, Index> {
    Index p0 = -1, p1 = -1;
    for (Index i = 0; i < static_cast<Index>(loop.size()); ++i) {
      if (loop[i].source != source::created)
        continue;
      if (loop[i].id == id0)
        p0 = i;
      else if (loop[i].id == id1)
        p1 = i;
      if (p0 >= 0 && p1 >= 0)
        return {p0, p1};
    }
    return {p0, p1};
  }

  template <typename Loop>
  static auto emit_loop(const Loop &loop, tf::buffer<Index> &offsets,
                        tf::buffer<vertex_t> &vertices) {
    offsets.push_back(static_cast<Index>(vertices.size()));
    for (const auto &v : loop)
      vertices.push_back(v);
  }

  template <typename Loop>
  static auto split_at(const Loop &loop, Index pos0, Index pos1,
                       tf::buffer<Index> &offsets,
                       tf::buffer<vertex_t> &vertices) {
    if (pos0 > pos1)
      std::swap(pos0, pos1);
    auto n = static_cast<Index>(loop.size());

    offsets.push_back(static_cast<Index>(vertices.size()));
    for (Index i = pos0; i <= pos1; ++i)
      vertices.push_back(loop[i]);

    offsets.push_back(static_cast<Index>(vertices.size()));
    for (Index i = pos1; i < n; ++i)
      vertices.push_back(loop[i]);
    for (Index i = 0; i <= pos0; ++i)
      vertices.push_back(loop[i]);
  }

  template <typename Loop, typename EdgeDefs, typename GetPoint>
  auto prepare_and_split(const Loop &loop, const EdgeDefs &edge_defs,
                         const GetPoint &get_point) -> void {
    _ihm.clear();
    _base_loop.clear();
    _edges.clear();
    _points.clear();

    // Base loop vertices are inserted first; shared vertices
    // keep the loop's topo_id (sub_id) rather than the dummy
    // used for edge-only lookups below.
    tf::make_contiguous_index_hash_map(loop, _ihm, Index(0));
    for (const auto &e : edge_defs) {
      auto v0 = vertex_t{source::created, e[0], {0, tf::topo_type::none}};
      auto v1 = vertex_t{source::created, e[1], {0, tf::topo_type::none}};
      if (_ihm.f().find(v0) == _ihm.f().end()) {
        _ihm.kept_ids().push_back(v0);
        _ihm.f()[v0] = static_cast<Index>(_ihm.kept_ids().size() - 1);
      }
      if (_ihm.f().find(v1) == _ihm.f().end()) {
        _ihm.kept_ids().push_back(v1);
        _ihm.f()[v1] = static_cast<Index>(_ihm.kept_ids().size() - 1);
      }
    }

    _base_loop.reserve(loop.size());
    for (const auto &v : loop)
      _base_loop.push_back(_ihm.f()[v]);

    _edges.reserve(edge_defs.size() * 2);
    for (const auto &e : edge_defs) {
      _edges.push_back(
          _ihm.f()[vertex_t{source::created, e[0], {0, tf::topo_type::none}}]);
      _edges.push_back(
          _ihm.f()[vertex_t{source::created, e[1], {0, tf::topo_type::none}}]);
    }

    _points.allocate(_ihm.kept_ids().size());
    for (std::size_t i = 0; i < _ihm.kept_ids().size(); ++i)
      _points[i] = get_point(_ihm.kept_ids()[i]);

    _fs.build(_base_loop, tf::make_edges(tf::make_blocked_range<2>(_edges)),
              tf::make_points(_points));
  }

  template <typename Loop, typename EdgeDefs, typename GetPoint>
  auto build_generic(const Loop &loop, const EdgeDefs &edge_defs,
                     const GetPoint &get_point, tf::buffer<Index> &offsets,
                     tf::buffer<vertex_t> &vertices) -> Index {
    prepare_and_split(loop, edge_defs, get_point);

    if (_fs.faces().size() == 0) {
      write_face(_base_loop, offsets, vertices);
      return 1;
    }

    if (_fs.holes().size())
      return extract_with_holes(offsets, vertices);
    return extract_without_holes(offsets, vertices);
  }

  template <typename Range>
  auto write_face(const Range &face, tf::buffer<Index> &offsets,
                  tf::buffer<vertex_t> &vertices) {
    offsets.push_back(static_cast<Index>(vertices.size()));
    for (auto id : face)
      vertices.push_back(_ihm.kept_ids()[id]);
  }

  auto extract_without_holes(tf::buffer<Index> &offsets,
                             tf::buffer<vertex_t> &vertices) -> Index {
    for (const auto &face : _fs.faces())
      write_face(face, offsets, vertices);
    return static_cast<Index>(_fs.faces().size());
  }

  auto extract_with_holes(tf::buffer<Index> &offsets,
                          tf::buffer<vertex_t> &vertices) -> Index {
    for (const auto &[face, hole_ids] :
         tf::zip(_fs.faces(), _fs.holes_for_faces())) {
      if (hole_ids.size()) {
        _hp.build(
            face,
            tf::make_faces(tf::make_indirect_range(hole_ids, _fs.holes())),
            tf::make_points(_points));
        write_face(_hp.face(), offsets, vertices);
      } else {
        write_face(face, offsets, vertices);
      }
    }
    return static_cast<Index>(_fs.faces().size());
  }

  struct hash_t {
    auto operator()(const vertex_t &v) const {
      return hash(std::array<Index, 2>{Index(v.source), v.id});
    }
    tf::array_hash<Index, 2> hash;
  };

  tf::index_hash_map<vertex_t, Index, hash_t> _ihm;
  tf::face_split_by_edges<Index, Int> _fs;
  tf::hole_patcher<Index, Int> _hp;
  tf::buffer<Index> _edges;
  tf::buffer<Index> _base_loop;
  tf::points_buffer<Int, 2> _points;
};

} // namespace tf
