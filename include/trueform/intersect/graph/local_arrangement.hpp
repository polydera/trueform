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

#include "../../arrangement/identity/identity_substrate.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../identity/find_points_on_carrier_ends.hpp"
#include "../identity/materialize_intersection_points.hpp"
#include "../polygon_intersections.hpp"
#include "./assert_carve_identities_are_accounted.hpp"
#include "./assert_split_survivors_are_ordered.hpp"
#include "./detect_plane_crossings.hpp"
#include "./face_descriptor.hpp"
#include "./final_identity_point.hpp"
#include "./find_moved_identity_planes.hpp"
#include "./find_touched_group_planes.hpp"
#include "./near_identity_rows.hpp"
#include "./plane_census.hpp"
#include "./plane_def_respan.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_graph.hpp"
#include "./plane_identity_birth.hpp"
#include "./plane_identity_collapse.hpp"
#include "./plane_identity_names.hpp"
#include "./plane_tables.hpp"
#include "./plane_uncut_entrants.hpp"
#include "./restate_plane_defs.hpp"
#include "./restate_split_survivors.hpp"
#include "./seat_entrant_plane_edges.hpp"
#include "./state_live_plane_members.hpp"
#include "./state_uncut_entrants.hpp"
#include "./touch_restated_plane_groups.hpp"

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
  using identities_t = tf::arrangement::identity_substrate<Index, Int>;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using pt3_t = tf::exact::pt3<Int>;

  auto clear() -> void {
    _intersections.clear();
    _dyadic_parameters.clear();
    _graph.clear();
    _face_offsets.clear();
    _intersection_points.clear();
    _landings.clear();
    _tables.clear();
    _entrants.clear();
    _entrant_descriptors.clear();
    _entrant_planes.clear();
    _entrant_orientations.clear();
    _entrant_plane_of.clear();
    _seated.clear();
    _live_member_offsets.clear();
    _live_members.clear();
    _carve_sources.clear();
    _carve_face_base = 0;
    _vertex_offsets.clear();
    _split_edge.clear();
    _split_offsets.clear();
    _split_data.clear();
    _near_rows.clear();
    _census = plane_census{};
  }

  /// Ask the gate for its near table at `window` (@ref
  /// collapse_plane_identities); the request stands until it is withdrawn.
  auto record_near_identities(Int window) -> void { _near_window = window; }
  auto near_window() const -> Int { return _near_window; }
  /// The near table the gate and every carve stated, deferred (@ref
  /// near_identity_rows): each member as it was stated, so a reader resolves
  /// a retired one through the identity substrate.
  auto near_rows() const -> const near_identity_rows<Index, pt3_t> & {
    return _near_rows;
  }
  /// A carve's block joining the table (@ref
  /// tf::arrangement::reseat::reseat_package).
  auto state_near_rows(const near_identity_rows<Index, pt3_t> &rows) -> void {
    _near_rows.append(rows);
  }

  /// The identities the build states enter `identities` as its local layer;
  /// every later read of one goes through it.
  template <typename GetOriginalPoint, typename ApplyToFace,
            typename ApplyToForm, typename FaceOffsets>
  auto build(identities_t &identities, intersections_t &&intersections,
             const GetOriginalPoint &get_original_point,
             const ApplyToFace &apply_to_face, const ApplyToForm &apply_to_form,
             const FaceOffsets &face_offsets, bool resolve_self_contours,
             bool placed_originals) -> void {
    const Index *no_faces = nullptr;
    build(identities, std::move(intersections), get_original_point,
          apply_to_face, apply_to_form, face_offsets, resolve_self_contours,
          placed_originals, tf::make_range(no_faces, no_faces));
  }

  /// The build over a re-seated world. `carved_faces` are the faces a
  /// re-seat cut out of a source face, as flat face keys, ascending: each
  /// enters the cut world whether or not a record names it, so every piece
  /// of a carved face is triangulated.
  template <typename GetOriginalPoint, typename ApplyToFace,
            typename ApplyToForm, typename FaceOffsets, typename CarvedFaces>
  auto build(identities_t &identities, intersections_t &&intersections,
             const GetOriginalPoint &get_original_point,
             const ApplyToFace &apply_to_face, const ApplyToForm &apply_to_form,
             const FaceOffsets &face_offsets, bool resolve_self_contours,
             bool placed_originals, const CarvedFaces &carved_faces) -> void {
    clear();
    identities.clear_local();
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
    identities.name_local_planes(_graph, _intersections, apply_to_face,
                                 apply_to_form, get_point, side_non_manifold);

    _census.defs_in = _graph.edge_defs().size();
    if (_graph.n_canon() == 0 && carved_faces.size() == 0)
      return;
    _vertex_offsets.allocate(_intersections.vertex_offsets().size());
    std::copy(_intersections.vertex_offsets().begin(),
              _intersections.vertex_offsets().end(), _vertex_offsets.begin());

    identities.state_local_names(
        _graph, _intersections, get_point, get_original_point, apply_to_face,
        tf::make_range(_vertex_offsets), resolve_self_contours, _census);
    identities.close_local_classes(_graph, get_point, _census);

    // One producer of coordinates: the gate quantizes every class in its
    // own frame, absorbs the coincidences (including a split that rounds
    // onto its root's endpoint) and states the result as the dense class
    // table plus the closed rewrite rows.
    _census.cross_root_merges = identities.collapse_local_identities(
        _graph, get_point,
        [&](Index cls) {
          return class_position(identities, cls, get_point, get_original_point);
        },
        _vertex_offsets, tf::make_range(_landings), placed_originals,
        n_points(), _near_window, _near_rows);

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
        [&](Index id) {
          return point_of(identities, id, get_point, get_original_point);
        },
        identities.local_statements(), identities.local_class_offsets(),
        identities.local_class_ids(), identities.local_merges(),
        tf::make_range(_vertex_offsets), identities.n_local_classes(),
        _split_edge, _split_offsets, _split_data, _census.splits_out_of_span,
        _census.splits_on_endpoint);
    const auto graph_descriptors = _graph.descriptors();
    const auto entrance_face_offsets = tf::make_range(_face_offsets);
    discover_uncut_entrants(
        _graph, plane_retired_originals(identities.local_merges()),
        tf::make_range(_split_edge), tf::make_range(_vertex_offsets),
        entrance_face_offsets, apply_to_form,
        [&graph_descriptors, &entrance_face_offsets](Index tag, Index object) {
          return plane_graph_names_face<Index>(graph_descriptors,
                                               entrance_face_offsets, tag,
                                               object);
        },
        [&graph_descriptors](Index face, Index side) {
          return std::array<Index, 2>{
              graph_descriptors[std::size_t(face)].object, side};
        },
        _entrants);
    discover_carved_entrants(graph_descriptors, entrance_face_offsets,
                             carved_faces, _entrants);
    tf::buffer<def_t> entrant_parts;
    tf::buffer<Index> entrant_instance_offsets;
    tf::buffer<def_t> entrant_instance_defs;
    if (_entrants.size() != 0) {
      state_uncut_entrants(
          _graph, apply_to_form, tf::make_range(_face_offsets),
          tf::make_range(_entrants), tf::make_range(_split_edge),
          _entrant_descriptors, entrant_parts, entrant_instance_offsets,
          entrant_instance_defs);
      identities.seat_local_entrant_planes(
          _graph, _entrant_descriptors, apply_to_form, get_point,
          _entrant_plane_of, _entrant_planes, _entrant_orientations, _seated);
    }
    respan_plane_defs(_graph, identities.local_merges(),
                      tf::make_range(_vertex_offsets), _split_edge,
                      _split_offsets, _split_data, entrant_parts,
                      entrant_instance_offsets, entrant_instance_defs,
                      _tables.defs(), _tables.def_offsets(), _tables.edges(),
                      n_cut_faces(), Index(_entrants.size()));
    if (_entrants.size() != 0)
      seat_entrant_plane_edges(_tables.edges(), n_cut_planes(), _seated);
    if (identities.local_merges().size() != 0)
      fuse_plane_defs(identities.local_merges(),
                      tf::make_range(_vertex_offsets), _tables.defs(),
                      _tables.def_offsets(), _tables.edges());
    _census.defs_out = _tables.defs().size();
  }

  /// A carve stated into this standing world: the mints, welds, entering
  /// and retiring faces join their tables, and the rows merge in
  /// key order
  /// (@ref tf::intersect::graph::restate_plane_defs). `sources` states per
  /// entering face its source face and, per side, the source side it keeps
  /// whole or `-1`; a carve stated after another appends to it, its faces
  /// following the faces the carves before it stated. `row_images`,
  /// `canon_images`, `touched_planes` and `moved_rows` are @ref
  /// restate_census's.
  template <typename GetOriginalPoint>
  auto restate_carve(
      identities_t &identities, const tf::buffer<pt3_t> &minted,
      const tf::buffer<std::array<Index, 3>> &welds,
      const tf::buffer<Index> &retired, const tf::buffer<Index> &restated,
      const tf::buffer<descriptor_t> &descriptors,
      const tf::buffer<std::array<Index, 4>> &sources,
      const tf::buffer<std::int8_t> &orientations,
      const tf::buffer<Index> &plane_of,
      const tf::buffer<tf::exact::plane_frame<Int>> &frames,
      const tf::buffer<tf::exact::canonical_plane<Int>> &names,
      const tf::buffer<def_t> &rows, const tf::buffer<Index> &planes,
      const GetOriginalPoint &get_original_point, tf::buffer<Index> &row_images,
      tf::buffer<Index> &canon_images, tf::buffer<Index> &touched_planes,
      tf::buffer<std::array<Index, 3>> &moved_rows) -> void {
    const auto moved = restate_identities(identities, minted, welds,
                                          get_original_point, moved_rows);
    tf::buffer<Index> dying;
    dying.allocate(retired.size() + restated.size());
    std::merge(retired.begin(), retired.end(), restated.begin(), restated.end(),
               dying.begin());
    const auto is_dying = [&dying](Index face) {
      return std::binary_search(dying.begin(), dying.end(), face);
    };

    const auto n_standing_faces = n_faces();
    const auto n_standing_planes = n_planes();
    const auto n_planes_after = n_standing_planes + Index(frames.size());
    tf::buffer<Index> live_offsets;
    tf::buffer<Index> live_members;
    state_live_plane_members(
        n_planes_after, n_standing_planes,
        [this](Index plane) { return member_count(plane); },
        [this](Index plane, Index k) { return member(plane, k); }, retired,
        n_standing_faces, plane_of, live_offsets, live_members);
    if (_carve_sources.size() == 0)
      _carve_face_base = n_standing_faces;
    if constexpr (tf::core::verification)
      tf::core::verify(_carve_face_base + Index(_carve_sources.size()) ==
                           n_standing_faces,
                       "a carve's faces follow the faces the carves before "
                       "it stated");
    tf::core::append(sources, _carve_sources);
    tf::core::append(descriptors, _entrant_descriptors);
    tf::core::append(orientations, _entrant_orientations);
    tf::core::append(plane_of, _entrant_plane_of);
    tf::core::append(frames, _entrant_planes);
    identities.seat_local_planes(names);
    _seated.clear();
    _live_member_offsets = std::move(live_offsets);
    _live_members = std::move(live_members);

    auto restated_tables = restate_touched_rows(
        identities, moved, planes, is_dying, rows, n_planes_after, row_images,
        canon_images, touched_planes);
    if constexpr (tf::core::verification)
      assert_carve_identities_are_accounted(
          restated_tables, n_created_points(identities) - Index(minted.size()),
          Index(minted.size()));
    _tables = std::move(restated_tables);
  }

  /// Census welds stated into this standing world: the welds join the closed
  /// rewrite table, and the planes holding an identity the closed table moved
  /// restate their rows (@ref
  /// tf::intersect::graph::restate_plane_defs). `row_images` states per
  /// standing row the row it now stands at, `-1` for none; `canon_images`
  /// per standing group the group its rows now stand in, `-1` for one that
  /// kept no row; `touched_planes` the planes holding a touched group,
  /// ascending — the dirty set this statement hands the diff build;
  /// `moved_rows` the closed rewrite rows the standing table did not hold. A
  /// census weld retires no original: the entrance a retirement owes is not
  /// stated here.
  template <typename GetOriginalPoint>
  auto restate_census(identities_t &identities,
                      const tf::buffer<std::array<Index, 3>> &welds,
                      const GetOriginalPoint &get_original_point,
                      tf::buffer<Index> &row_images,
                      tf::buffer<Index> &canon_images,
                      tf::buffer<Index> &touched_planes,
                      tf::buffer<std::array<Index, 3>> &moved_rows) -> void {
    if constexpr (tf::core::verification)
      for (const auto &weld : welds)
        tf::core::verify(weld[0] == Index(1),
                         "a census weld retires no original");
    const tf::buffer<pt3_t> no_mints;
    const auto moved = restate_identities(identities, no_mints, welds,
                                          get_original_point, moved_rows);

    const auto offsets = vertex_offsets();
    const auto planes =
        find_moved_identity_planes(_tables, offsets, moved, [this](Index face) {
          return plane_of_face(face);
        });
    const auto never_dying = [](Index) { return false; };
    const tf::buffer<def_t> no_rows;
    _tables = restate_touched_rows(identities, moved, planes, never_dying,
                                   no_rows, n_planes(), row_images,
                                   canon_images, touched_planes);
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
  /// The created identity space ends where `identities` ends it: an id past
  /// `n_points()` names one of its created classes.
  auto n_created_points(const identities_t &identities) const -> Index {
    return n_points() + identities.n_local_created();
  }

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
  auto point_of(const identities_t &identities, Index id,
                const GetOriginalPoint &get_original_point) const -> pt3_t {
    if (id < n_points())
      return intersection_point(id, get_original_point);
    auto get_input_point = [&](std::int16_t tag, Index point) -> pt3_t {
      if (tag >= 0)
        return get_original_point(int(tag), point);
      return intersection_point(point, get_original_point);
    };
    return point_of(identities, id, get_input_point, get_original_point);
  }

  /// THE one way a coordinate is read at this tier: an identity's statement,
  /// every endpoint it names read through the local layer by the one
  /// position law (@ref tf::intersect::graph::final_identity_point).
  /// `get_point` answers an intersection point, `get_mesh_point` an
  /// original.
  template <typename GetPoint, typename GetMeshPoint>
  auto point_of(const identities_t &identities, Index id,
                const GetPoint &get_point,
                const GetMeshPoint &get_mesh_point) const -> pt3_t {
    const evaluated_identity<Index> self{id, false, nullptr};
    return statement_point(
        identities, id,
        [&](std::int16_t tag, Index point) {
          return local_point(identities, tag, point, &self, get_point,
                             get_mesh_point);
        },
        get_point);
  }

  /// A class's statement read the same way: its root's endpoints, read
  /// through the local layer, blended at its dyadic parameter.
  template <typename GetPoint, typename GetMeshPoint>
  auto class_position(const identities_t &identities, Index cls,
                      const GetPoint &get_point,
                      const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return class_statement_point(
        identities, cls, [&](std::int16_t tag, Index point) {
          return local_point(identities, tag, point, nullptr, get_point,
                             get_mesh_point);
        });
  }

  /// An identity's statement with every endpoint it names read through
  /// `read(tag, id)`: an intersection point answers `get_point`, a created
  /// identity its direct form or its class's root blended at the class's
  /// dyadic parameter. The layer a reader evaluates at is its `read`.
  template <typename Read, typename GetPoint>
  auto statement_point(const identities_t &identities, Index id,
                       const Read &read, const GetPoint &get_point) const
      -> pt3_t {
    if (id < n_points())
      return get_point(std::int16_t(-1), id);
    const auto cls =
        identities.local_created_classes()[std::size_t(id - n_points())];
    if (cls < Index(0))
      return identities.local_direct_points()[std::size_t(-cls - Index(1))];
    return class_statement_point(identities, cls, read);
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

  // ---- the entrance: the source faces the cut world never named --------

  /// The flat source faces a weld or a split reaches outside the cut
  /// world. Their definitions are in the tables; these are the faces.
  auto entrants() const { return tf::make_range(_entrants); }
  /// One descriptor per entrant face, in that order: a definition's
  /// `face` past the graph's own descriptors is a position in here.
  auto entrant_descriptors() const {
    return tf::make_range(_entrant_descriptors);
  }
  /// The frames of the planes the entrants founded, past the graph's planes;
  /// an entrant whose name a cut plane holds is seated on that plane.
  auto entrant_planes() const { return tf::make_range(_entrant_planes); }
  /// One projected winding per entrant face, in entrant order, in the frame
  /// of the plane it is seated on; downstream plane consumers read it.
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
  /// A plane's members: the graph's own, then the entrants seated on it.
  auto member_count(Index plane) const -> Index {
    if (_live_member_offsets.size() != 0)
      return _live_member_offsets[std::size_t(plane) + 1] -
             _live_member_offsets[std::size_t(plane)];
    const auto own = plane < n_cut_planes()
                         ? Index(_graph.plane_faces(plane).size())
                         : Index(0);
    return _seated.size() == 0
               ? own
               : own + Index(_seated[std::size_t(plane)].size());
  }
  auto member(Index plane, Index position) const -> Index {
    if (_live_member_offsets.size() != 0)
      return _live_members[std::size_t(
          _live_member_offsets[std::size_t(plane)] + position)];
    const auto own = plane < n_cut_planes()
                         ? Index(_graph.plane_faces(plane).size())
                         : Index(0);
    if (position < own)
      return _graph.plane_faces(plane)[std::size_t(position)];
    return n_cut_faces() +
           _seated[std::size_t(plane)][std::size_t(position - own)];
  }
  auto plane_of_face(Index face) const -> Index {
    if (face < n_cut_faces())
      return _graph.plane_of()[std::size_t(face)];
    return _entrant_plane_of[std::size_t(face - n_cut_faces())];
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
  /// The source face and side a face's side stands on: a face the build
  /// named is its own source, a face a carve stated answers its source's
  /// side when it keeps that side whole and `-1` otherwise.
  auto source_side(Index face, Index side) const -> std::array<Index, 2> {
    if (_carve_sources.size() == 0 || face < _carve_face_base)
      return {descriptor(face).object, side};
    const auto &source = _carve_sources[std::size_t(face - _carve_face_base)];
    return {source[0], source[std::size_t(side) + 1]};
  }
  auto face_orientation(Index face) const -> std::int8_t {
    if (face < n_cut_faces())
      return _graph.face_orientation()[std::size_t(face)];
    return _entrant_orientations[std::size_t(face - n_cut_faces())];
  }

  auto census() const -> const plane_census & { return _census; }

private:
  template <typename Read>
  auto class_statement_point(const identities_t &identities, Index cls,
                             const Read &read) const -> pt3_t {
    return plane_class_position<Index, Int>(
        _graph, identities.local_class_roots()[std::size_t(cls)],
        identities.local_class_ts()[std::size_t(cls)],
        identities
            .local_statements()[std::size_t(
                identities.local_class_offsets()[std::size_t(cls)])]
            .name,
        tf::make_range(_vertex_offsets), read, [&read](int tag, Index point) {
          return read(std::int16_t(tag), point);
        });
  }

  /// One endpoint at the local layer: what `L` states it speaks, under the
  /// enclosing statements `mints`.
  template <typename GetPoint, typename GetMeshPoint>
  auto local_point(const identities_t &identities, std::int16_t tag, Index id,
                   const evaluated_identity<Index> *mints,
                   const GetPoint &get_point,
                   const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return final_identity_point<pt3_t>(
        identities.local_merges().size() != 0,
        [&identities](Index source, Index key) {
          return identities.local_target(source, key);
        },
        tf::make_range(_vertex_offsets), tag, id, mints,
        [&](std::int16_t at_tag, Index at,
            const evaluated_identity<Index> *self) -> pt3_t {
          if (at_tag >= 0)
            return get_mesh_point(int(at_tag), at);
          return statement_point(
              identities, at,
              [&](std::int16_t read_tag, Index point) {
                return local_point(identities, read_tag, point, self,
                                   get_point, get_mesh_point);
              },
              get_point);
        });
  }

  /// The rows of every group `moved`, the dying faces or the joining `rows`
  /// reach, restated into a new table for this world to adopt; `row_images`,
  /// `canon_images` and `touched_planes` are @ref restate_census's.
  template <typename IsDying>
  auto restate_touched_rows(const identities_t &identities,
                            const tf::buffer<std::array<Index, 2>> &moved,
                            const tf::buffer<Index> &planes,
                            const IsDying &is_dying,
                            const tf::buffer<def_t> &rows, Index n_planes_after,
                            tf::buffer<Index> &row_images,
                            tf::buffer<Index> &canon_images,
                            tf::buffer<Index> &touched_planes)
      -> plane_tables<Index, Int> {
    const auto offsets = vertex_offsets();
    const auto touched =
        touch_restated_plane_groups(_tables, identities.local_merges(), offsets,
                                    moved, planes, is_dying, rows);
    touched_planes = find_touched_group_planes(
        _tables, touched, [this](Index face) { return plane_of_face(face); });
    plane_tables<Index, Int> restated_tables;
    restate_plane_defs(
        _tables, identities.local_merges(), offsets, touched, is_dying, rows,
        [this](const def_t &def) { return plane_of_face(def.face); },
        n_planes_after, restated_tables, row_images);
    state_canon_images(row_images, restated_tables, canon_images);
    return restated_tables;
  }

  /// A statement's identities joined to this world's: each mint a direct
  /// created identity, the welds closed into the local layer, and the split
  /// survivors speaking the closed layer. Returns the keys of `moved_rows`,
  /// the closed rows the layer did not hold, ascending.
  template <typename GetOriginalPoint>
  auto restate_identities(identities_t &identities,
                          const tf::buffer<pt3_t> &minted,
                          const tf::buffer<std::array<Index, 3>> &welds,
                          const GetOriginalPoint &get_original_point,
                          tf::buffer<std::array<Index, 3>> &moved_rows)
      -> tf::buffer<std::array<Index, 2>> {
    identities.mint_local_points(minted);
    const auto offsets = vertex_offsets();
    identities.weld_local(welds, offsets[offsets.size() - 1],
                          n_created_points(identities), moved_rows);
    restate_split_survivors(_graph, identities.local_merges(),
                            tf::make_range(_vertex_offsets), _split_edge,
                            _split_offsets, _split_data);
    if constexpr (tf::core::verification)
      assert_split_survivors_are_ordered<Index, Int>(
          _graph, identities.local_merges(), _split_edge, _split_offsets,
          _split_data, [&](std::int16_t tag, Index id) -> pt3_t {
            return tag < 0 ? point_of(identities, id, get_original_point)
                           : get_original_point(int(tag), id);
          });
    tf::buffer<std::array<Index, 2>> moved;
    moved.allocate(moved_rows.size());
    tf::parallel_transform(
        tf::make_range(moved_rows), moved,
        [](const std::array<Index, 3> &row) {
          return std::array<Index, 2>{row[0], row[1]};
        },
        tf::checked);
    return moved;
  }

  /// Per standing group the group its first row with an image stands in —
  /// every live instance of a group re-emits onto one key — and `-1` for one
  /// that kept no row; read before `restated` replaces the standing tables.
  auto state_canon_images(const tf::buffer<Index> &row_images,
                          const plane_tables<Index, Int> &restated,
                          tf::buffer<Index> &canon_images) const -> void {
    canon_images.allocate(std::size_t(_tables.n_canon()));
    const auto &standing_offsets = _tables.def_offsets();
    const auto restated_defs = restated.edge_defs();
    tf::parallel_for_each(
        tf::make_sequence_range(_tables.n_canon()),
        [&](Index canon) {
          auto image = Index(-1);
          for (auto row = standing_offsets[std::size_t(canon)];
               image == Index(-1) &&
               row < standing_offsets[std::size_t(canon) + 1];
               ++row)
            image = row_images[std::size_t(row)];
          canon_images[std::size_t(canon)] =
              image == Index(-1) ? Index(-1)
                                 : restated_defs[std::size_t(image)].id;
        },
        tf::checked);
  }

  intersections_t _intersections;
  tf::buffer<param_t> _dyadic_parameters;
  graph_t _graph;
  tf::buffer<Index> _face_offsets;
  tf::buffer<pt3_t> _intersection_points;
  tf::buffer<std::array<Index, 2>> _landings;
  plane_tables<Index, Int> _tables;
  // the entrance's own tables: one descriptor and one plane per source
  // face the cut world never named, in discovery order — the position
  // that a definition's stamp past the graph's descriptors, and its
  // block past the graph's planes, both name
  tf::buffer<Index> _entrants;
  tf::buffer<descriptor_t> _entrant_descriptors;
  tf::buffer<tf::exact::plane_frame<Int>> _entrant_planes;
  tf::buffer<std::int8_t> _entrant_orientations;
  // per entrant: the plane its name seats it on
  tf::buffer<Index> _entrant_plane_of;
  // per plane, the entrants seated on it; empty while there are none
  tf::offset_block_buffer<Index, Index> _seated;
  // per plane its live members, once a carve retired some: it then answers
  // membership in place of the graph's own and the seated
  tf::buffer<Index> _live_member_offsets;
  tf::buffer<Index> _live_members;
  // per face a carve stated, past `_carve_face_base`: its source face and
  // the source side each of its sides keeps whole, `-1` for none
  tf::buffer<std::array<Index, 4>> _carve_sources;
  Index _carve_face_base = 0;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<Index> _split_edge;
  tf::buffer<Index> _split_offsets;
  tf::buffer<Index> _split_data;
  Int _near_window = Int(0);
  near_identity_rows<Index, pt3_t> _near_rows;
  plane_census _census;
};

} // namespace tf::intersect::graph
