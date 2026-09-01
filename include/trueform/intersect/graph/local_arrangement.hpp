/*
 * Copyright (c) 2026 XLAB
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

#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../identity/find_points_on_carrier_ends.hpp"
#include "../identity/materialize_intersection_points.hpp"
#include "../polygon_intersections.hpp"
#include "./close_plane_classes.hpp"
#include "./detect_plane_crossings.hpp"
#include "./face_descriptor.hpp"
#include "./plane_census.hpp"
#include "./plane_def_respan.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_graph.hpp"
#include "./plane_identity_birth.hpp"
#include "./plane_identity_collapse.hpp"
#include "./plane_identity_names.hpp"
#include "./plane_tables.hpp"
#include "./plane_uncut_entrants.hpp"
#include "./state_uncut_entrants.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::intersect::graph {

/// THE LOCAL ARRANGEMENT — the intersections, the plane graph and the
/// definition tables the splits are applied to, owned by one build.
///
/// Created points are retained in the parameter forms their producing
/// tiers publish. IBP points remain vertex anchors or exact parameters on
/// original-edge carriers, accompanied by the existing dyadic transport
/// parameter. Rooted points remain a canonical graph root plus their
/// dyadic parameter; rootless classes retain their name-owned position.
/// No 3D created-point table is stored: a consumer that needs a lattice
/// position evaluates the corresponding statement through the
/// original-point callback. The one materialized table — the 3D
/// intersection points this build produced for its own consumption —
/// stays owned here for the plane tier to read; the owner releases it
/// once nothing downstream reads it.
///
/// The product is the REBUILT definition tables: post-split pieces are
/// definitions inheriting their parent's face, original edge, ordinal and
/// radial bit, canon-major again with fresh spans and a rebuilt plane
/// CSR, so nothing downstream re-applies a splits side table. One
/// materialization gate closes the build: every surviving identity
/// quantizes once in its own frame, same-position identities collapse per
/// root and across roots, and one dense remap states the result.
template <typename Index, typename RealType, typename Int>
class local_arrangement {
public:
  using intersections_t = tf::polygon_intersections<Index, RealType, Int>;
  using graph_t = plane_graph<Index, Int>;
  using descriptor_t = face_descriptor<Index>;
  using frame_t = typename graph_t::frame_t;
  using def_t = plane_edge_def<Index>;
  using statement_t = plane_name_statement<Index>;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using pt3_t = tf::exact::pt3<Int>;

  auto clear() -> void {
    _intersections.clear();
    _dyadic_parameters.clear();
    _graph.clear();
    _face_offsets.clear();
    _intersection_points.clear();
    _landings.clear();
    _statements.clear();
    _class_offsets.clear();
    _class_id.clear();
    _class_root.clear();
    _class_t.clear();
    _created_class.clear();
    _merges.clear();
    _tables.defs().clear();
    _tables.def_offsets().clear();
    _tables.edges().clear();
    _entrants.clear();
    _entrant_descriptors.clear();
    _entrant_planes.clear();
    _entrant_orientations.clear();
    _vertex_offsets.clear();
    _split_edge.clear();
    _split_offsets.clear();
    _split_data.clear();
    _tables.n_canon() = 0;
    _n_classes = 0;
    _census = plane_census{};
  }

  template <typename GetOriginalPoint, typename ApplyToFace,
            typename ApplyToForm, typename FaceOffsets>
  auto build(intersections_t &&intersections,
             const GetOriginalPoint &get_original_point,
             const ApplyToFace &apply_to_face, const ApplyToForm &apply_to_form,
             const FaceOffsets &face_offsets, bool resolve_self_contours,
             bool placed_originals) -> void {
    clear();
    tf::intersect::materialize_intersection_parameters<Index, Int>(
        intersections, _dyadic_parameters);
    const auto vertex_offsets = intersections.vertex_offsets();
    const auto get_flat_point = [&](Index flat) -> pt3_t {
      const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
      return get_original_point(int(tag),
                                flat - vertex_offsets[std::size_t(tag)]);
    };
    tf::intersect::materialize_intersection_points<Index, Int>(
        intersections, get_flat_point, _dyadic_parameters,
        _intersection_points);
    tf::intersect::find_points_on_carrier_ends<Index, Int>(
        intersections, get_flat_point, _intersection_points, _landings);

    _intersections = std::move(intersections);
    _face_offsets.allocate(face_offsets.size());
    std::copy(face_offsets.begin(), face_offsets.end(), _face_offsets.begin());
    const auto get_point = [&](std::int16_t tag, Index id) -> pt3_t {
      return tag < 0 ? _intersection_points[std::size_t(id)]
                     : get_original_point(int(tag), id);
    };
    // the mesh's own non-manifold edge is an input fact: the form's edge
    // link states it, the graph carries it onto the side's definitions
    const auto side_non_manifold = [&](Index tag, Index object,
                                       std::int16_t side) -> bool {
      bool answer = false;
      apply_to_form(tag, [&](const auto &form) {
        if constexpr (tf::has_manifold_edge_link_policy<decltype(form)>)
          answer = !form.manifold_edge_link()[object][std::size_t(side)]
                        .is_manifold();
      });
      return answer;
    };
    _graph.build(_intersections, apply_to_face, apply_to_form, get_point,
                 side_non_manifold);

    _census.defs_in = _graph.edge_defs().size();
    if (_graph.n_canon() == 0)
      return;
    _vertex_offsets.allocate(_intersections.vertex_offsets().size());
    std::copy(_intersections.vertex_offsets().begin(),
              _intersections.vertex_offsets().end(), _vertex_offsets.begin());

    detect_plane_crossings(_graph, _intersections, get_point,
                           get_original_point, apply_to_face,
                           tf::make_range(_vertex_offsets),
                           resolve_self_contours, _statements, _census);
    _n_classes = close_plane_classes<Index, Int>(
        _graph, get_point, _statements, _class_offsets, _class_id, _class_root,
        _class_t, _census);

    // One producer of coordinates: @ref
    // tf::intersect::graph::collapse_plane_identities quantizes every
    // class in its own frame, absorbs the coincidences (including a split
    // that rounds onto its root's endpoint) and states the result as the
    // dense class table plus the closed rewrite rows.
    _census.cross_root_merges = collapse_plane_identities(
        _graph, get_point,
        [&](Index cls) {
          return class_position(cls, get_point, get_original_point);
        },
        _vertex_offsets, tf::make_range(_landings), placed_originals,
        _statements, _class_offsets, _n_classes, n_points(), _class_id,
        _created_class, _merges);
    // a class the collapse minted to NAME a position states no root: it
    // answers a coordinate off its own name and never enters the split
    // walk, which is why `_n_classes` does not grow with it
    const auto n_classes_after =
        _class_offsets.size() == 0 ? std::size_t(0) : _class_offsets.size() - 1;
    for (auto cls = std::size_t(_n_classes); cls < n_classes_after; ++cls) {
      _class_root.push_back(Index(-1));
      _class_t.push_back(param_t(0));
    }

    // A split found on one instance is a fact of the canonical group, so
    // it subdivides EVERY instance of that group. Both halves are the
    // shared mechanism: @ref tf::intersect::graph::order_plane_splits
    // states the ordered survivors of every cut root, and @ref
    // tf::intersect::graph::respan_plane_defs merges the pieces into the
    // definition tables, the canonical spans and the plane CSR.
    //
    // The same merge carries the ENTRANCE: the facts the roots and the
    // rewrite rows state reach source faces the cut world never named,
    // and those faces state their own edges into it.
    //
    // @ref tf::intersect::graph::fuse_plane_defs closes the stage: the
    // rewrite rows reach EVERY definition and the tables are
    // canonicalized on the result, so one undirected final key is one
    // canonical group and no consumer resolves an endpoint through the
    // merge table.
    order_plane_splits<Index, Int>(
        _graph, get_point,
        [&](Index id) { return point_of(id, get_point, get_original_point); },
        _statements, _class_offsets, _class_id, _merges,
        tf::make_range(_vertex_offsets), _n_classes, _split_edge,
        _split_offsets, _split_data, _census.splits_out_of_span,
        _census.splits_on_endpoint);
    const auto graph_descriptors = _graph.descriptors();
    const auto entrance_face_offsets = tf::make_range(_face_offsets);
    discover_uncut_entrants(
        _graph, plane_retired_originals(_merges), tf::make_range(_split_edge),
        tf::make_range(_vertex_offsets), entrance_face_offsets, apply_to_form,
        [&graph_descriptors, &entrance_face_offsets](Index tag, Index object) {
          return plane_graph_names_face<Index>(graph_descriptors,
                                               entrance_face_offsets, tag,
                                               object);
        },
        _entrants);
    tf::buffer<def_t> entrant_parts;
    tf::buffer<Index> entrant_instance_offsets;
    tf::buffer<def_t> entrant_instance_defs;
    if (_entrants.size() != 0)
      state_uncut_entrants(
          _graph, get_point, apply_to_form, tf::make_range(_face_offsets),
          tf::make_range(_entrants), tf::make_range(_split_edge),
          _entrant_descriptors, _entrant_planes, _entrant_orientations,
          entrant_parts, entrant_instance_offsets, entrant_instance_defs);
    _tables.n_canon() = respan_plane_defs(
        _graph, _merges, tf::make_range(_vertex_offsets), _split_edge,
        _split_offsets, _split_data, entrant_parts, entrant_instance_offsets,
        entrant_instance_defs, _tables.defs(), _tables.def_offsets(),
        _tables.edges());
    if (_merges.size() != 0)
      _tables.n_canon() = fuse_plane_defs(
          _merges, tf::make_range(_vertex_offsets), _tables.defs(),
          _tables.def_offsets(), _tables.edges());
    _census.defs_out = _tables.defs().size();
  }

  auto intersections() const -> const intersections_t & {
    return _intersections;
  }
  auto graph() const -> const graph_t & { return _graph; }

  // ---- Point contracts -------------------------------------------------

  auto n_vertex_points() const -> Index {
    return _intersections.n_vertex_points();
  }
  auto n_points() const -> Index { return _intersections.n_points(); }
  auto n_created() const -> Index { return Index(_created_class.size()); }
  auto n_created_points() const -> Index { return n_points() + n_created(); }

  auto vertex_anchor(Index id) const -> const auto & {
    return _intersections.vertex_anchor(id);
  }
  auto home_edge(Index id) const -> const auto & {
    return _intersections.home_edge(id);
  }
  auto dyadic_parameter(Index id) const -> param_t {
    return _dyadic_parameters[std::size_t(id)];
  }

  template <typename GetOriginalPoint>
  auto intersection_point(Index id,
                          const GetOriginalPoint &get_original_point) const
      -> pt3_t {
    if (id < n_vertex_points()) {
      const auto &anchor = vertex_anchor(id);
      return get_original_point(int(anchor.tag), anchor.vid);
    }
    const auto offsets = vertex_offsets();
    const auto &edge = home_edge(id);
    auto original = [&](Index flat) -> pt3_t {
      const auto tag = tf::exact::tag_of_flat_vertex(offsets, flat);
      return get_original_point(int(tag), flat - offsets[std::size_t(tag)]);
    };
    const auto u = original(edge.u);
    const auto v = original(edge.v);
    const auto t = dyadic_parameter(id);
    pt3_t point;
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      point[coordinate] =
          tf::exact::dyadic_blend(u[coordinate], v[coordinate], t);
    return point;
  }

  template <typename GetOriginalPoint>
  auto point_of(Index id, const GetOriginalPoint &get_original_point) const
      -> pt3_t {
    if (id < n_points())
      return intersection_point(id, get_original_point);
    auto get_input_point = [&](std::int16_t tag, Index point) -> pt3_t {
      if (tag >= 0)
        return get_original_point(int(tag), point);
      return intersection_point(point, get_original_point);
    };
    return point_of(id, get_input_point, get_original_point);
  }

  /// THE one way a coordinate is read: an intersection point answers
  /// from its own table, everything this build minted blends its root's
  /// endpoints at its dyadic parameter, on demand.
  template <typename GetPoint, typename GetMeshPoint>
  auto point_of(Index id, const GetPoint &get_point,
                const GetMeshPoint &get_mesh_point) const -> pt3_t {
    if (id < n_points())
      return get_point(std::int16_t(-1), id);
    const auto cls = _created_class[std::size_t(id - n_points())];
    return class_position(cls, get_point, get_mesh_point);
  }

  template <typename GetPoint, typename GetMeshPoint>
  auto class_position(Index cls, const GetPoint &get_point,
                      const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return plane_class_position<Index, Int>(
        _graph, _class_root[std::size_t(cls)], _class_t[std::size_t(cls)],
        _statements[std::size_t(_class_offsets[std::size_t(cls)])].name,
        tf::make_range(_vertex_offsets), get_point, get_mesh_point);
  }

  auto vertex_offsets() const { return _intersections.vertex_offsets(); }
  auto face_offsets() const { return tf::make_range(_face_offsets); }

  /// The 3D intersection-point table materialized by this build. The plane
  /// tier reads it during its own build; the owner releases it afterwards.
  auto intersection_points() const {
    return tf::make_range(_intersection_points);
  }
  auto release_intersection_points() -> void {
    _intersection_points = tf::buffer<pt3_t>();
  }

  // ---- THE PRODUCT: what the triangulation tier consumes ----------------

  /// The post-split definition table, canon-major: every constraint of
  /// every plane, each one stating its face, its original edge, its
  /// base-loop ordinal and its radial bit. There is no splits side
  /// table — these ARE the splits, applied.
  auto edge_defs() const { return _tables.edge_defs(); }
  auto n_canon() const -> Index { return _tables.n_canon(); }
  auto canon_group(Index canon) const { return _tables.canon_group(canon); }
  /// A plane's definitions, in canonical order.
  auto plane_edges(Index plane) const { return _tables.plane_edges(plane); }
  auto tables() const -> const plane_tables<Index, Int> & { return _tables; }
  /// `{source, id, target}` — identities the gate absorbed, in the
  /// rewrite currency the pipeline already applies, so a consumer
  /// holding an old id stays merge-blind.
  auto merges() const { return tf::make_range(_merges); }

  // ---- the entrance: the source faces the cut world never named --------

  /// The flat source faces a weld or a split reaches outside the cut
  /// world. Their definitions are in the tables; these are the faces.
  auto entrants() const { return tf::make_range(_entrants); }
  /// One descriptor per entrant face, in that order: a definition's
  /// `face` past the graph's own descriptors is a position in here.
  auto entrant_descriptors() const {
    return tf::make_range(_entrant_descriptors);
  }
  /// One plane per entrant face, in the same order: the blocks the plane
  /// CSR carries past the graph's planes.
  auto entrant_planes() const { return tf::make_range(_entrant_planes); }
  /// One projected winding per entrant face, in the same order. The source
  /// face states this once when it enters; downstream plane consumers read it.
  auto entrant_orientations() const {
    return tf::make_range(_entrant_orientations);
  }

  /// The immutable roots this build split and each root's surviving
  /// identities. A later plane tier uses this published piece statement
  /// when a newly reached source face inherits an already-applied
  /// boundary split.
  auto split_roots() const { return tf::make_range(_split_edge); }
  auto split_survivors(std::size_t group) const {
    return tf::slice(tf::make_range(_split_data),
                     std::size_t(_split_offsets[group]),
                     std::size_t(_split_offsets[group + 1]));
  }

  // ---- Unified cut/entrant carrier space --------------------------------

  auto n_cut_planes() const -> Index { return _graph.n_planes(); }
  auto n_planes() const -> Index { return Index(_tables.edges().size()); }
  auto n_cut_faces() const -> Index {
    return Index(_graph.descriptors().size());
  }
  auto n_faces() const -> Index {
    return n_cut_faces() + Index(_entrant_descriptors.size());
  }

  auto frame(Index plane) const -> const frame_t & {
    if (plane < n_cut_planes())
      return _graph.frame(plane);
    return _entrant_planes[std::size_t(plane - n_cut_planes())];
  }
  auto member_count(Index plane) const -> Index {
    return plane < n_cut_planes() ? Index(_graph.plane_faces(plane).size())
                                  : Index(1);
  }
  auto member(Index plane, Index position) const -> Index {
    if (plane < n_cut_planes())
      return _graph.plane_faces(plane)[std::size_t(position)];
    return n_cut_faces() + (plane - n_cut_planes());
  }
  auto plane_of_face(Index face) const -> Index {
    if (face < n_cut_faces())
      return _graph.plane_of()[std::size_t(face)];
    return n_cut_planes() + (face - n_cut_faces());
  }
  auto descriptor(Index face) const -> const descriptor_t & {
    if (face < n_cut_faces())
      return _graph.descriptors()[std::size_t(face)];
    return _entrant_descriptors[std::size_t(face - n_cut_faces())];
  }
  auto descriptors() const {
    const auto cut = _graph.descriptors();
    const auto entrants = entrant_descriptors();
    const auto cut_size = n_cut_faces();
    return tf::make_mapped_range(
        tf::make_sequence_range(n_faces()),
        [cut, entrants, cut_size](Index face) -> const descriptor_t & {
          if (face < cut_size)
            return cut[std::size_t(face)];
          return entrants[std::size_t(face - cut_size)];
        });
  }
  auto face_orientation(Index face) const -> std::int8_t {
    if (face < n_cut_faces())
      return _graph.face_orientation()[std::size_t(face)];
    return _entrant_orientations[std::size_t(face - n_cut_faces())];
  }

  // ---- the census and the class tables an oracle reads ----

  auto census() const -> const plane_census & { return _census; }
  /// The statements, sorted by name: a name's run is its class.
  auto statements() const { return tf::make_range(_statements); }
  auto class_offsets() const { return tf::make_range(_class_offsets); }
  auto n_classes() const -> Index { return _n_classes; }
  auto class_id() const { return tf::make_range(_class_id); }

private:
  intersections_t _intersections;
  tf::buffer<param_t> _dyadic_parameters;
  graph_t _graph;
  tf::buffer<Index> _face_offsets;
  tf::buffer<pt3_t> _intersection_points;
  tf::buffer<std::array<Index, 2>> _landings;
  tf::buffer<statement_t> _statements;
  tf::buffer<Index> _class_offsets;
  tf::buffer<Index> _class_id;
  // a created point's complete statement: its name (in _statements) plus the
  // root it was born on and its dyadic parameter there. No coordinate
  // is stored — blending commutes with projection, so a consumer that
  // blends on demand reads exactly what a table would have held.
  tf::buffer<Index> _class_root;
  tf::buffer<param_t> _class_t;
  tf::buffer<Index> _created_class;
  tf::buffer<std::array<Index, 3>> _merges;
  plane_tables<Index, Int> _tables;
  // the entrance's own tables: one descriptor and one plane per source
  // face the cut world never named, in discovery order — the position
  // that a definition's stamp past the graph's descriptors, and its
  // block past the graph's planes, both name
  tf::buffer<Index> _entrants;
  tf::buffer<descriptor_t> _entrant_descriptors;
  tf::buffer<tf::exact::plane_frame<Int>> _entrant_planes;
  tf::buffer<std::int8_t> _entrant_orientations;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<Index> _split_edge;
  tf::buffer<Index> _split_offsets;
  tf::buffer<Index> _split_data;
  Index _n_classes = 0;
  plane_census _census;
};

} // namespace tf::intersect::graph
