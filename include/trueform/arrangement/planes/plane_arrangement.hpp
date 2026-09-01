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

#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"
#include "../../intersect/graph/local_arrangement.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/topo_id.hpp"
#include "./advance_plane_wave.hpp"
#include "./classify_plane_entrant_origins.hpp"
#include "./classify_plane_wave_entrants.hpp"
#include "./close_plane_lazy_round.hpp"
#include "./close_plane_round.hpp"
#include "./compact_plane_weld_triangles.hpp"
#include "./discover_plane_refinement_entrants.hpp"
#include "./discover_plane_wave_entrants.hpp"
#include "./make_plane_refinement_cut_mask.hpp"
#include "./make_plane_refinement_entrant_orientations.hpp"
#include "./make_plane_refinement_evidence.hpp"
#include "./make_plane_refinement_overlay.hpp"
#include "./make_plane_refinement_physical_splits.hpp"
#include "./make_plane_refinement_plan.hpp"
#include "./make_plane_weld_entrant_tables.hpp"
#include "./map_plane_refinement_splits.hpp"
#include "./materialize_plane_products.hpp"
#include "./plane_arrangement_arena.hpp"
#include "./retire_stalled_plane.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_refinement_plan.hpp"
#include "./plane_round_evidence.hpp"
#include "./plane_triangulation_types.hpp"
#include "./plane_wave_answered.hpp"
#include "./plane_world.hpp"
#include "./propagate_plane_refinement_rings.hpp"
#include "./publish_plane_final_pieces.hpp"
#include "./rewrite_plane_flat_triangles.hpp"
#include "./seat_plane_wave_entrants.hpp"
#include "./state_plane_round_frontier.hpp"
#include "./state_plane_weld_entrant_side.hpp"
#include "./triangulate_plane_round.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// One triangulation per plane carrier of a PREPARED world.
///
/// The local arrangement already applied every split, grouped the
/// definitions canon-major and published each plane's edge block, and the
/// plane graph already published the frames and the member windings. So
/// this holds no per-plane re-derivation at all: the plane's edge block IS
/// the constraint set, a definition IS its own provenance, a 2D point is
/// one read of its flat identity, and a preserve-mode CDT that does not
/// refuse ends the plane's work.
///
/// A refusal is the only thing that costs anything. The plane is rebuilt in
/// resolve mode to state its crossings and interior landings on the exact
/// pieces that carry them; those statements close into identities, order into
/// splits, and move the carriers they touch from the world tier into this
/// arrangement's own — where the split subdivides them and the frontier is
/// triangulated again. A wave that states nothing new ends the loop, and the
/// planes still refusing are published in `failed()`.
///
/// A weld — two standing identities the election joins, or two the projected
/// triangulation could not tell apart — is an identity substitution, and this
/// arrangement owns it: the rows naming a retired identity are swept, ported,
/// rewritten and closed under their new keys in the same wave. A weld moves no
/// constraint, so a plane whose rows only changed identity KEEPS the
/// triangulation it has and its corners are remapped through the same closed
/// table. A plane that REFUSED holds none to keep, and the substitution may
/// have retired the very coincidence it refused on, so it reads its rewritten
/// rows again: a carrier is triangulated again exactly when its constraint set
/// changed or it carries no product.
///
/// THE WAVE'S GRAIN LAWS bound what a recovery costs, and every phase of one
/// is written against them:
///
/// 1. A RECOVERY ROUND'S COST IS PROPORTIONAL TO THE DIRTY SET. No phase of a
///    wave is proportional to the world: a carrier no statement reached keeps
///    its product and is never read.
/// 2. PORTING ROUTES CHANGED GROUPS ONLY. A group whose statement CHANGES — a
///    split, a weld, a retirement reaching it — goes local and takes every
///    carrier of it in the same wave, so this arrangement never owns a piece
///    some plane still reads the world's version of. The frontier is therefore
///    the changed groups' carriers — a ring — and never the component they sit
///    in.
/// 3. AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS: a
///    carrier the wave ports for another group's sake keeps naming the world's
///    own rows for it, and the carrier the wave never touched reads exactly
///    those rows and publishes exactly that ticket.
/// 4. CARRIER LOOKUP IS THE IDENTITY'S OWN MEMBERSHIP, BUILT ON DEMAND. Which
///    carriers hold a retired identity is a question the world answers by
///    lookup — @ref tf::arrangement::states_flat_carriers — and the structure
///    that answers it belongs to the first wave that retires anything, so a
///    build that retires nothing never builds one.
///
/// This owns the state and wires the phases; every step of the build is a free
/// operation of `tf::arrangement` this hands its own tables, buffers and census
/// to.
template <typename Index, typename Int> class plane_arrangement {
public:
  using pt3_t = tf::exact::pt3<Int>;
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using name_t = plane_recovery_name<Index, param_t>;
  using coplanar_t = tf::arrangement::coplanar_descriptor<Index>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;

  /// A junction the triangulation found, named by the endpoint identities of
  /// the two exact pieces it joins.
  static constexpr Index crossing_name_kind = 5;
  static constexpr Index refinement_name_kind = 6;

  // ---- THE PRODUCT ----

  /// Every triangle of the arrangement, contiguous per plane and inside
  /// a plane contiguous per member. A corner is a FLAT identity:
  /// `< n_flat_points()` an original vertex in the tags' flat space,
  /// past it a point of the intersection space.
  auto triangles() const { return tf::make_range(_triangles); }
  /// Per triangle, per slot s (corner s -> corner s + 1): the canonical
  /// piece it belongs to, or -1 for a filler diagonal. Recorded only when
  /// `record_triangle_arrangement()` was asked before the build.
  auto slot_parents() const { return tf::make_range(_slot_parents); }
  /// Per triangle, per corner: where the corner sits on the emitting
  /// member's own polygon — its corner ordinal, the original side it lies
  /// on, or the interior.
  auto corner_subs() const { return tf::make_range(_corner_subs); }
  auto immutable_piece_extent() const -> Index {
    return _immutable_canon_extent;
  }
  /// The exclusive upper bound of the final-piece ticket address space. A
  /// PA-owned suffix piece can shadow an immutable prefix identity, so this is
  /// not the cardinality of live pieces.
  auto final_piece_ticket_extent() const -> Index {
    return _immutable_canon_extent + Index(_final_piece_definitions.size());
  }
  /// The complete definition span named by a non-filler slot ticket. The
  /// immutable prefix stays caller-owned; PA-owned suffix spans include every
  /// inactive immutable instance sharing the final piece.
  template <typename Immutable>
  auto piece_definitions(const Immutable &immutable, Index ticket) const {
    assert(immutable.n_canon() == _immutable_canon_extent);
    assert(ticket >= Index(0) && ticket < final_piece_ticket_extent());
    if (ticket < _immutable_canon_extent) {
      const auto span = immutable.canon_group(ticket);
      return tf::make_range(span.begin(), span.end());
    }
    const auto span =
        _final_piece_definitions[std::size_t(ticket - _immutable_canon_extent)];
    return tf::make_range(span.begin(), span.end());
  }
  /// The local block each plane's definitions resolve through, `-1` for a
  /// plane that still reads the world. Empty while every plane does.
  auto plane_tickets() const { return tf::make_range(_plane_ticket); }
  /// One local plane block's definitions, in key order, each read from the
  /// tier that owns it: AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR
  /// BOTH CARRIERS, so a block this arrangement holds still names the world's
  /// own rows for every group no wave changed.
  template <typename Immutable>
  auto current_plane_defs(const Immutable &immutable, Index block) const {
    const auto world_defs = immutable.edge_defs();
    const auto local_defs = tf::make_range(_local_tables.defs());
    return tf::make_mapped_range(
        tf::make_range(_local_tables.edges())[std::size_t(block)],
        [world_defs, local_defs](Index row) -> const def_t & {
          return row < Index(0) ? world_defs[std::size_t(-1 - row)]
                                : local_defs[std::size_t(row)];
        });
  }
  /// The piece TICKET each row of one local plane block states: the immutable
  /// prefix while the world is still the group's authority, the PA-owned
  /// suffix once it is not.
  template <typename Immutable>
  auto current_plane_tickets(const Immutable &immutable, Index block) const {
    const auto world_defs = immutable.edge_defs();
    const auto local_defs = tf::make_range(_local_tables.defs());
    const auto canon_base = _immutable_canon_extent;
    return tf::make_mapped_range(
        tf::make_range(_local_tables.edges())[std::size_t(block)],
        [world_defs, local_defs, canon_base](Index row) {
          return row < Index(0)
                     ? world_defs[std::size_t(-1 - row)].id
                     : canon_base + local_defs[std::size_t(row)].id;
        });
  }
  /// The definition-row indices one local plane block resolves through.
  auto current_plane_def_rows(Index block) const {
    return tf::make_range(_local_tables.edges())[std::size_t(block)];
  }
  auto plane_triangles(Index p) const {
    return tf::slice(tf::make_range(_triangles),
                     std::size_t(_plane_offsets[std::size_t(p)]),
                     std::size_t(_plane_offsets[std::size_t(p) + 1]));
  }
  auto plane_range(Index p) const -> std::array<Index, 2> {
    return {_plane_offsets[std::size_t(p)], _plane_offsets[std::size_t(p) + 1]};
  }
  /// A graph face's own emitted span: for a stack member, the subset it
  /// covers, in that member's own winding.
  auto face_range(Index face) const -> std::array<Index, 2> {
    return _face_range[std::size_t(face)];
  }
  /// Per triangle: -1 for a survivor, else its row in
  /// `coplanar_descriptors()`. Recorded only when
  /// `record_triangle_arrangement()` was asked before the build.
  auto coplanar_of() const { return tf::make_range(_coplanar_of); }
  auto coplanar_descriptors() const { return tf::make_range(_coplanar); }
  /// Per triangle: its region is covered by more than one member. Recorded
  /// only when `record_triangle_arrangement()` was asked before the build.
  auto stacked() const { return tf::make_range(_stacked); }
  /// THE COMPLETENESS SURFACE: the planes that emitted nothing, so they
  /// carry no product this tier can state, and every face on them holds an
  /// empty span. Empty means every carrier holds its product — a plane
  /// refuses here only when its triangulation refused every round of the
  /// recovery wave, and when a wave names a fact this tier's producers
  /// cannot publish every plane is named at once and the product is empty.
  /// A carrier that bounds no area holds its product by emitting nothing and
  /// is named here on the same terms as every other. Neither is a refinement
  /// the discovery declined — that carrier keeps the stock triangulation and
  /// is counted in `refinement_census()`.
  auto failed() const { return tf::make_range(_failed); }
  /// `{source, id, target}` — the identities this tier's waves absorbed, in
  /// the same rewrite currency the world publishes its own in, closed, so
  /// one binary search answers what an identity of this arrangement speaks.
  auto merges() const { return tf::make_range(_merges); }
  /// The faces refinement promoted past the world's own: face
  /// `n_base_faces() + i` is stated by row `i`, and its plane is
  /// `n_base_planes() + i`. The three tables are one statement — the
  /// descriptor, the carrier's exact frame, and the face's winding in it —
  /// so a consumer reads the promoted side of every carrier fact here
  /// rather than rederiving it.
  auto promoted_descriptors() const {
    return tf::make_range(_promoted_descriptors);
  }
  auto promoted_frames() const { return tf::make_range(_promoted_frames); }
  auto promoted_orientations() const {
    return tf::make_range(_promoted_orientations);
  }
  /// The world extents this build stood on. A face or plane below them is
  /// the world's own; the promotion appended everything at or past them.
  auto n_base_faces() const -> Index { return _base_faces; }
  auto n_base_planes() const -> Index { return _base_planes; }
  /// Per triangle: the plane-local 2-cell it came out of — the block a walk
  /// that never crosses a constraint reaches. Cells meet only at pieces, so
  /// `(plane, cell)` names the coarse carrier a component flood walks, and a
  /// stack's members share the cells they cover. Recorded only when
  /// `record_triangle_cells()` was asked before the build; a weld that
  /// compacted the store leaves it empty.
  auto triangle_cells() const { return tf::make_range(_triangle_cells); }
  auto n_planes() const -> Index {
    return _plane_offsets.size() == 0 ? Index(0)
                                      : Index(_plane_offsets.size()) - Index(1);
  }
  auto n_faces() const -> Index { return Index(_face_range.size()); }
  auto n_flat_points() const -> Index {
    return _vertex_offsets[_vertex_offsets.size() - 1];
  }
  /// The flat identity a corner speaks: the vertex offsets this build was
  /// given are the per-tag prefix, and a created id lifts past all of them.
  auto flat_of(std::int16_t tag, Index id) const -> Index {
    return tf::intersect::graph::flat_of_vertex(_vertex_offsets, tag, id);
  }
  auto census() const -> const plane_arrangement_census & { return _census; }
  auto refinement_census() const -> const plane_refinement_census & {
    return _refinement_census;
  }

  /// Ask emission to keep the cell each triangle came out of, which the
  /// triangulation still holds at that moment and drops with the next plane.
  /// The request stands until it is withdrawn, so a build reads it once and
  /// the emission nobody asked runs untouched.
  auto record_triangle_cells(bool record = true) -> void {
    _record_cells = record;
  }

  /// Ask emission to keep THE ARRANGEMENT FACTS A TRIANGLE CARRIES: the piece
  /// each constrained slot lies on (`slot_parents`), the survivor a coincident
  /// duplicate names (`coplanar_of` with `coplanar_descriptors`), and whether
  /// its region is stacked (`stacked`) — together with the piece space
  /// `piece_definitions()` publishes over them. All three are facts about how
  /// a triangle sits among MANY carriers; a world whose carrier is a single
  /// face states none of them. The request stands until it is withdrawn, so a
  /// build reads it once and a caller that never asks pays neither the
  /// constraint-owner walk, nor the store, nor the ticket pass that rebases
  /// them.
  auto record_triangle_arrangement(bool record = true) -> void {
    _record_arrangement = record;
  }

  auto clear() -> void {
    _triangles.clear();
    _slot_parents.clear();
    _corner_subs.clear();
    _plane_offsets.clear();
    _face_range.clear();
    _coplanar_of.clear();
    _coplanar.clear();
    _stacked.clear();
    _failed.clear();
    _triangle_cells.clear();
    _local_tables.defs().clear();
    _local_tables.def_offsets().clear();
    _local_tables.edges().clear();
    _local_tables.n_canon() = Index(0);
    _group_router.clear();
    _plane_ticket.clear();
    _merges.clear();
    _final_piece_definitions.clear();
    _created_class.clear();
    _direct_created_points.clear();
    _plane_steiner_sites.clear();
    _refinement_refused.clear();
    _promoted_descriptors.clear();
    _promoted_frames.clear();
    _promoted_orientations.clear();
    _class_t.clear();
    _class_name.clear();
    _base_created = 0;
    _base_faces = 0;
    _base_planes = 0;
    _immutable_canon_extent = 0;
    _refine_config = tf::cdt_refine_config{};
    _refined = false;
    _census = plane_arrangement_census{};
    _refinement_census = plane_refinement_census{};
  }

  /// THE CLOSED-WORLD BUILD: the world states its own points, so the
  /// created extent and the flat prefix come from the caller.
  ///
  /// The world is taken by mutable reference because a world may state its
  /// definition tier LAZILY, and the barrier that makes it real is this
  /// build's own event. A world born with its tier is untouched by that.
  /// `apply_to_form` is the SOURCE MESH, and stating it is what licenses the
  /// wave entrance: a split can reach a face the cut world never named only
  /// where the caller can hand this arrangement that face's own corners. A
  /// world that holds every face already — the mesh pole — states nothing
  /// here, and the entrance is not compiled.
  template <typename Policy, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets, typename ApplyToForm = tf::none_t>
  auto build(plane_world<Policy> &world, Index n_base_created,
             const GetBasePoint &get_base_point,
             const GetOriginalPoint &get_original_point,
             const VertexOffsets &vertex_offsets,
             const ApplyToForm &apply_to_form = tf::none) -> void {
    if (!build_core(world, n_base_created, get_base_point, get_original_point,
                    vertex_offsets, apply_to_form))
      return;
    publish_products(world);
  }

  /// THE REFINED BUILD: complete refinement's exact split, four-ring carrier
  /// and plane-local Steiner pre-step, then hand that prepared state to the
  /// ordinary stock path once. No stock triangulation or refusal recovery
  /// occurs inside the pre-step.
  template <typename Policy, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets>
  auto build_refined(plane_world<Policy> &world, Index n_base_created,
                     const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point,
                     const VertexOffsets &vertex_offsets,
                     const tf::cdt_refine_config &config) -> void {
    build_refined_closed_world(world, n_base_created, get_base_point,
                               get_original_point, vertex_offsets, config);
  }

  template <typename RealType, typename GetOriginalPoint,
            typename ApplyToForm = tf::none_t>
  auto build(const tf::intersect::graph::local_arrangement<Index, RealType, Int>
                 &arrangement,
             const GetOriginalPoint &get_original_point,
             const ApplyToForm &apply_to_form = tf::none) -> void {
    build_on_local_arrangement(
        arrangement, get_original_point,
        [&](auto &world, const auto &get_base_point) {
          build(world, world.n_created_points(), get_base_point,
                get_original_point, world.vertex_offsets(), apply_to_form);
        });
  }

  template <typename RealType, typename GetOriginalPoint, typename ApplyToForm>
  auto build_refined(
      const tf::intersect::graph::local_arrangement<Index, RealType, Int>
          &arrangement,
      const GetOriginalPoint &get_original_point,
      const ApplyToForm &apply_to_form, const tf::cdt_refine_config &config)
      -> void {
    build_on_local_arrangement(
        arrangement, get_original_point,
        [&](auto &world, const auto &get_base_point) {
          run_refined(world, get_original_point, apply_to_form, get_base_point,
                      config);
        });
  }

  template <typename RealType, typename GetOriginalPoint, typename ApplyToForm>
  auto build(const tf::intersect::graph::local_arrangement<Index, RealType, Int>
                 &arrangement,
             const GetOriginalPoint &get_original_point,
             const ApplyToForm &apply_to_form,
             const tf::cdt_refine_config &config) -> void {
    build_refined(arrangement, get_original_point, apply_to_form, config);
  }

  /// Planes the refinement PRODUCER declined to plan for. Not a failure:
  /// the stock kernel triangulates them in the same pass, so the carrier
  /// holds an unrefined product and `failed()` stays the one surface that
  /// states an absent one.
  auto refinement_refused_planes() const {
    return tf::make_range(_refinement_refused);
  }

  auto n_created() const -> Index { return Index(_created_class.size()); }

  /// Resolve one identity created by this tier from the exact authority that
  /// named it. `get_base_point` resolves the immutable prefix; PA-owned birth
  /// edges, rootless names and direct plane-local forms remain private details
  /// of this arrangement and are evaluated on demand without a point table.
  template <typename GetBasePoint, typename GetMeshPoint>
  auto resolve_created_point(Index id, const GetBasePoint &get_base_point,
                             const GetMeshPoint &get_mesh_point) const
      -> pt3_t {
    assert(id >= _base_created &&
           id < _base_created + Index(_created_class.size()));
    return final_point(std::int16_t(-1), id, get_base_point, get_mesh_point);
  }

private:
  auto append_refinement_refused(const tf::buffer<Index> &planes) -> void {
    tf::core::append(planes, _refinement_refused);
    if (_refinement_refused.size() == 0)
      return;
    tbb::parallel_sort(_refinement_refused.begin(), _refinement_refused.end());
    _refinement_refused.erase_till_end(std::unique(
        _refinement_refused.begin(), _refinement_refused.end()));
  }

  /// The world a local arrangement states, and the one exact reader of its
  /// identities: an original vertex answers from its mesh, an identity of the
  /// arrangement from the table the world materialized, and anything the
  /// arrangement itself created blends the name that placed it. Both public
  /// local-arrangement builds enter through here, so the reader is stated
  /// once.
  template <typename RealType, typename GetOriginalPoint, typename Build>
  auto build_on_local_arrangement(
      const tf::intersect::graph::local_arrangement<Index, RealType, Int>
          &arrangement,
      const GetOriginalPoint &get_original_point, const Build &build_with)
      -> void {
    auto world = make_plane_world(arrangement);
    const auto intersection_points = world.intersection_points();
    const auto get_input_point = [&](std::int16_t tag, Index id) -> pt3_t {
      return tag < 0 ? intersection_points[std::size_t(id)]
                     : get_original_point(int(tag), id);
    };
    const auto get_base_point = [&](std::int16_t tag, Index id) -> pt3_t {
      if (tag >= 0)
        return get_original_point(int(tag), id);
      return world.point_of(id, get_input_point, get_original_point);
    };
    build_with(world, get_base_point);
  }

  template <typename World, typename Splits, typename GetBasePoint,
            typename GetOriginalPoint>
  auto apply_refinement_wave(
      const World &world, const Splits &splits,
      tf::buffer<plane_flat_weld<Index>> &&welds,
      const GetBasePoint &get_base_point,
      const GetOriginalPoint &get_original_point) -> bool {
    if (splits.size() == 0 && welds.size() == 0)
      return true;
    plane_round_evidence<Index, Int> evidence;
    make_plane_refinement_evidence(
        splits, refinement_name_kind,
        param_t(1) << tf::exact::meta<Int>::param_bits, true,
        evidence.statements, evidence.topology);
    evidence.welds = std::move(welds);
    tf::buffer<Index> frontier;
    const auto result = advance_plane_wave(
        world, evidence, _local_tables, _plane_ticket, _group_router, _merges,
        _created_class, _class_t, _class_name, _vertex_offsets,
        n_flat_points(), _base_created,
        exact_point_of(get_base_point, get_original_point),
        name_point_of(get_base_point, get_original_point), _census, frontier);
    _census.created = _created_class.size();
    return result != plane_wave_result::unsupported;
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets, typename ApplyToForm>
  auto fall_back_to_stock(World &world, Index n_base_created,
                          const GetBasePoint &get_base_point,
                          const GetOriginalPoint &get_original_point,
                          const VertexOffsets &vertex_offsets,
                          const ApplyToForm &apply_to_form) -> void {
    if (build_core(world, n_base_created, get_base_point, get_original_point,
                   vertex_offsets, apply_to_form))
      publish_products(world);
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets>
  auto build_refined_closed_world(
      World &world, Index n_base_created,
      const GetBasePoint &get_base_point,
      const GetOriginalPoint &get_original_point,
      const VertexOffsets &vertex_offsets,
      const tf::cdt_refine_config &config) -> void {
    // refinement's splits are keyed by the canonical group and its discovery
    // reads a root's instances, so this arm needs the group space up front
    world.materialize();
    if (!initialize(world, n_base_created, vertex_offsets))
      return;
    refine_at_emission(config);
    auto initial = take_refinement_discovery(world, config, get_base_point,
                                             get_original_point);
    if (!apply_refinement_wave(world, initial.splits,
                               std::move(initial.welds), get_base_point,
                               get_original_point)) {
      fall_back_to_stock(world, n_base_created, get_base_point,
                         get_original_point, vertex_offsets, tf::none);
      return;
    }
    _refinement_census.refused_planes = _refinement_refused.size();
    _census.created = _created_class.size();
    if (!run_stock(world, get_base_point, get_original_point, tf::none))
      return;
    publish_products(world);
  }

  template <typename World, typename GetOriginalPoint, typename ApplyToForm,
            typename GetBasePoint>
  auto run_refined(World &world,
                   const GetOriginalPoint &get_original_point,
                   const ApplyToForm &apply_to_form,
                   const GetBasePoint &get_base_point,
                   const tf::cdt_refine_config &config) -> void {
    if (!initialize(world, world.n_created_points(), world.vertex_offsets()))
      return;
    refine_at_emission(config);

    auto initial = take_refinement_discovery(world, config, get_base_point,
                                             get_original_point);

    make_plane_refinement_physical_splits<Index, Int>(
        initial.splits,
        [&](const auto &split) { return world.canon_group(split.group); },
        world, apply_to_form, initial.physical_splits);
    const auto initial_physical_rows = initial.physical_splits.size();
    if (initial.physical_splits.size() != 0) {
      const auto n_tags = Index(world.face_offsets().size()) - Index(1);
      const auto cut_mask =
          make_plane_refinement_cut_mask(world, n_tags, apply_to_form);
      const auto rings = propagate_plane_refinement_rings<Index, Int>(
          cut_mask, apply_to_form, get_original_point,
          initial.physical_splits);
      discover_plane_refinement_entrants(
          initial.physical_splits, cut_mask, world.face_offsets(),
          apply_to_form, initial.promoted_faces);
      _refinement_census.rings_run = rings.rings_run;
      if (initial.physical_splits.size() !=
          initial_physical_rows + rings.accepted_physical_rows) {
        fall_back_to_stock(world, world.n_created_points(), get_base_point,
                           get_original_point, world.vertex_offsets(),
                           apply_to_form);
        return;
      }
    }
    _refinement_census.physical_delivery_rows =
        initial.physical_splits.size();
    _refinement_census.promoted_planes = initial.promoted_faces.size();

    tf::intersect::graph::plane_tables<Index, Int> raw_entrants;
    const auto expand_entrant_side = [&](const def_t &parent,
                                         tf::buffer<def_t> &output) {
      state_plane_weld_entrant_side<Index>(world, world.vertex_offsets(),
                                           parent, output);
    };
    const tf::buffer<std::array<Index, 3>> no_merges;
    make_plane_weld_entrant_tables<Index, Int>(
        tf::make_range(initial.promoted_faces), world.n_faces(), no_merges,
        world.vertex_offsets(), world.face_offsets(), get_base_point,
        apply_to_form, expand_entrant_side, _promoted_descriptors,
        _promoted_frames, raw_entrants);
    make_plane_refinement_entrant_orientations<Index>(
        _promoted_descriptors, _promoted_frames, get_original_point,
        apply_to_form, _promoted_orientations);

    tf::buffer<Index> entrant_origins;
    tf::buffer<char> fresh_origins;
    if (raw_entrants.n_canon() != Index(0)) {
      const tf::buffer<def_t> no_origin_defs;
      const tf::buffer<plane_birth_edge_record<Index>> no_origin_index;
      tf::buffer<def_t> staged_origin_defs;
      tf::buffer<plane_birth_edge_record<Index>> staged_origin_index;
      if (!classify_plane_entrant_origins(
              world, raw_entrants, _immutable_canon_extent, no_origin_defs,
              no_origin_index, staged_origin_defs, staged_origin_index,
              entrant_origins, fresh_origins)) {
        fall_back_to_stock(world, world.n_created_points(), get_base_point,
                           get_original_point, world.vertex_offsets(),
                           apply_to_form);
        return;
      }
    }

    tf::buffer<plane_refinement_split<Index, Int>> mapped_splits;
    if (raw_entrants.edges().size() == 0) {
      mapped_splits = std::move(initial.splits);
    } else {
      tf::buffer<Index> immutable_group_ticket;
      tf::buffer<Index> entrant_group_ticket;
      if (!make_plane_refinement_overlay<Index, Int>(
              world, initial.splits, raw_entrants, entrant_origins,
              world.vertex_offsets(), _local_tables, _plane_ticket,
              _group_router, immutable_group_ticket,
              entrant_group_ticket) ||
          !map_plane_refinement_splits<Index, Int>(
              initial.splits, initial.physical_splits,
              immutable_group_ticket, raw_entrants, entrant_group_ticket,
              _promoted_descriptors, world.n_faces(), world.n_planes(),
              _local_tables, apply_to_form, n_flat_points(),
              world.vertex_offsets(), mapped_splits)) {
        fall_back_to_stock(world, world.n_created_points(), get_base_point,
                           get_original_point, world.vertex_offsets(),
                           apply_to_form);
        return;
      }
      if (mapped_splits.size() < initial.splits.size()) {
        fall_back_to_stock(world, world.n_created_points(), get_base_point,
                           get_original_point, world.vertex_offsets(),
                           apply_to_form);
        return;
      }
      _refinement_census.ring_boundary_splits =
          mapped_splits.size() - initial.splits.size();
    }
    // THE PROMOTION BARRIER: the pre-step above holds the base value and its
    // base extents; the wave below reads the promoted one.
    World promoted_world(world, _promoted_descriptors, _promoted_frames,
                               _promoted_orientations);
    if (!apply_refinement_wave(promoted_world, mapped_splits,
                               std::move(initial.welds), get_base_point,
                               get_original_point)) {
      fall_back_to_stock(world, world.n_created_points(), get_base_point,
                         get_original_point, world.vertex_offsets(),
                         apply_to_form);
      return;
    }

    _refinement_census.affected_planes =
        _plane_ticket.size() == 0
            ? std::size_t(0)
            : std::count_if(_plane_ticket.begin(), _plane_ticket.end(),
                            [](Index block) { return block != Index(-1); });
    _refinement_census.refused_planes = _refinement_refused.size();
    _census.created = _created_class.size();
    if (!run_stock(promoted_world, get_base_point, get_original_point,
                   apply_to_form))
      return;
    publish_products(promoted_world);
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets, typename ApplyToForm>
  auto build_core(World &world, Index n_base_created,
                  const GetBasePoint &get_base_point,
                  const GetOriginalPoint &get_original_point,
                  const VertexOffsets &vertex_offsets,
                  const ApplyToForm &apply_to_form) -> bool {
    if (!initialize(world, n_base_created, vertex_offsets))
      return false;
    return run_stock(world, get_base_point, get_original_point, apply_to_form);
  }

  /// The emission pass IS the refinement pass. Boundary splits are the wave's
  /// alone, so the producer that emits never moves a constraint.
  auto refine_at_emission(const tf::cdt_refine_config &config) -> void {
    _refine_config = config;
    _refine_config.split_encroached = false;
    _refined = true;
  }

  template <typename World, typename VertexOffsets>
  auto initialize(World &world, Index n_base_created,
                  const VertexOffsets &vertex_offsets) -> bool {
    clear();
    // the piece space a slot publishes IS the group space, so a build that
    // asked for it asks the world for its tier now rather than at a barrier
    if (_record_arrangement)
      world.materialize();
    _vertex_offsets = tf::make_range(vertex_offsets);
    _base_created = n_base_created;
    _base_faces = world.n_faces();
    _base_planes = world.n_planes();
    _immutable_canon_extent = world.n_canon();
    return world.n_planes() != 0;
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename ApplyToForm>
  auto run_stock(World &world, const GetBasePoint &get_base_point,
                 const GetOriginalPoint &get_original_point,
                 const ApplyToForm &apply_to_form) -> bool {
    _arena.clear();
    _arena.allocate_planes(world.n_planes());
    _arena.face_range.allocate(std::size_t(world.n_faces()));
    for (std::size_t f = 0; f < _arena.face_range.size(); ++f)
      _arena.face_range[f] = {Index(0), Index(0)};

    if (run_rounds(world, tf::make_sequence_range(world.n_planes()),
                   get_base_point, get_original_point,
                   apply_to_form) == plane_round_result::aborted) {
      fail_every_plane(world);
      return false;
    }
    return true;
  }

  /// Triangulate the given planes and close rounds until the waves exhaust.
  template <typename World, typename Planes, typename GetBasePoint,
            typename GetOriginalPoint, typename ApplyToForm>
  auto run_rounds(World &world, const Planes &planes,
                  const GetBasePoint &get_base_point,
                  const GetOriginalPoint &get_original_point,
                  const ApplyToForm &apply_to_form) -> plane_round_result {
    const auto flat_point = flat_point_of(get_base_point, get_original_point);
    const auto exact_point = exact_point_of(get_base_point, get_original_point);
    const auto name_point = name_point_of(get_base_point, get_original_point);
    plane_round_evidence<Index, Int> evidence;
    evidence.census.rounds = 1;
    evidence.census.rebuilt_planes = planes.size();
    triangulate_plane_round(world, planes, _local_tables, _plane_ticket,
                            _vertex_offsets, n_flat_points(),
                            crossing_name_kind, _refined, _record_arrangement,
                            _record_cells, _refine_config, _plane_steiner_sites,
                            flat_point, _arena, evidence);
    tf::buffer<Index> attempted;
    tf::buffer<Index> frontier;
    tf::buffer<Index> refusals;
    refusals.allocate_and_initialize(std::size_t(world.n_planes()), Index(0));
    const auto close_frontier = [&](const plane_round_evidence<Index, Int> &ev,
                                    plane_round_result result) {
      // A refusing carrier does not always argue toward an answer. Two of its
      // statements can undo each other — an identity minted one round and
      // merged away the next — and then the wave states something NEW every
      // round while the carrier's constraint set returns to itself, so no
      // progress test can see the loop. A budget on the carrier's OWN refusals
      // is what sees it instead: a carrier that recovers spends at most two,
      // and eight stands far enough above that, and above every scene coarser
      // than the supported envelope, that the budget cannot cut a live carrier
      // short.
      constexpr std::size_t refusal_budget = 8;
      // A carrier that spends it leaves the wave. It is already published as a
      // failure — @ref tf::arrangement::update_plane_failures states that every
      // round it refuses — so the spend publishes nothing and only drops the
      // stale product a carrier that produced BEFORE it began refusing still
      // holds, exactly as the no-progress guard does for the same reason.
      for (const auto plane : ev.refused)
        if (++refusals[std::size_t(plane)] == Index(refusal_budget)) {
          retire_stalled_plane<Index, Int>(world, plane, _arena);
          ++_census.spent_planes;
        }
      if (result != plane_round_result::retry)
        return result;
      std::size_t kept = 0;
      for (std::size_t at = 0; at != frontier.size(); ++at)
        if (std::size_t(refusals[std::size_t(frontier[at])]) < refusal_budget)
          frontier[kept++] = frontier[at];
      frontier.erase_till_end(frontier.begin() + std::ptrdiff_t(kept));
      return frontier.size() == 0 ? plane_round_result::done : result;
    };
    // THE ENTRANCE'S ASK: the wave states which of its roots are still the
    // world's and cut an original side, and which originals it retires; this
    // answers whether a face they reach is one no tier of this arrangement
    // has named yet. A world that holds every face states no source mesh, and
    // neither this nor the branch it feeds is compiled.
    tf::buffer<Index> entrants;
    plane_wave_answered<Index> answered;
    // the lambda is stated for every world, so the branch a world with no
    // source mesh takes is what lets it be: `world.face_offsets()` does not
    // depend on the lambda's own parameters, so it is checked either way
    const auto state_entrance = [&](const auto &roots, const auto &retired) {
      if constexpr (std::is_same<ApplyToForm, tf::none_t>::value) {
        (void)roots;
        (void)retired;
        return false;
      } else {
        discover_plane_wave_entrants(world, retired, roots, answered,
                                     world.face_offsets(), apply_to_form,
                                     entrants);
        return entrants.size() != 0;
      }
    };
    const auto close_round = [&](const auto &round_planes,
                                 plane_round_evidence<Index, Int> &ev) {
      const auto entrance = [&] {
        if constexpr (std::is_same<ApplyToForm, tf::none_t>::value)
          return tf::none;
        else
          return state_entrance;
      }();
      return close_plane_round(world, round_planes, ev, _local_tables,
                               _plane_ticket, _group_router, _merges,
                               _created_class, _class_t, _class_name,
                               _vertex_offsets, n_flat_points(), _base_created,
                               exact_point, name_point, _failed, _census,
                               frontier, entrance);
    };
    // THE BARRIER: a round that saw something on a world with no group space
    // makes one, states the extent for the first time, and hands its carriers
    // back to the loop. What that round saw is not translated — it is seen
    // again, against the real tables.
    //
    // THE WAVE ENTRANCE IS THAT SENTENCE with "no group space" replaced by "a
    // split reaching a face the cut world never named": the round is handed
    // back unconsumed, the faces are promoted, and the same evidence is seen
    // again against the promoted tables.
    auto round = plane_round_result::retry;
    if (!close_plane_lazy_round(world, evidence, _immutable_canon_extent,
                                _census, frontier))
      round = close_round(planes, evidence);
    // THE ENTRANCE EXISTS ONLY WHERE A SOURCE MESH DOES, so a world that
    // holds every face compiles none of it — not the trigger, not the
    // promotion, and not this branch. What it answers with is a round like
    // any other: its refusals are charged and its frontier is filtered.
    const auto take_entrance = [&](plane_round_result result) {
      if constexpr (std::is_same<ApplyToForm, tf::none_t>::value) {
        return result;
      } else {
        return result == plane_round_result::entrance
                   ? enter_wave_entrants(world, evidence, entrants, answered,
                                         get_base_point, get_original_point,
                                         apply_to_form, refusals, frontier)
                   : result;
      }
    };
    round = close_frontier(evidence, take_entrance(round));
    // THE NO-PROGRESS GUARD. A wave advances by STATING something a table
    // keeps: a new identity, a merge, or a smaller frontier. A round that
    // states none of the three has restated what the tables already hold.
    //
    // ONE such round is not the end — a wave legitimately rebuilds without
    // producing and then produces. THE BUDGET IS MEASURED, not chosen: over
    // the whole green ladder (exact, 1e-6 and 1e-9 harnesses) the longest
    // run of no-progress rounds a wave RECOVERS from is 1, while a carrier
    // trading one restated split forever runs 46,270. Eight rounds sits
    // eight times above everything observed to recover and three orders
    // below the cycle, so the guard cannot cut a live wave short and cannot
    // fail to catch a dead one.
    constexpr std::size_t no_progress_budget = 8;
    auto stated_created = _created_class.size();
    auto stated_merges = _merges.size();
    auto stated_frontier = frontier.size();
    // the entrance states no identity and no merge — it promotes faces — so
    // the set of faces this arrangement has answered for is its own term, and
    // it grows strictly every time the entrance fires
    auto stated_answered = answered.count;
    std::size_t stalled_run = 0;
    while (round == plane_round_result::retry) {
      attempted = std::move(frontier);
      evidence = plane_round_evidence<Index, Int>{};
      evidence.census.rounds = 1;
      evidence.census.rebuilt_planes = attempted.size();
      const auto retried = tf::make_range(attempted);
      triangulate_plane_round(world, retried, _local_tables, _plane_ticket,
                              _vertex_offsets, n_flat_points(),
                              crossing_name_kind, _refined,
                              _record_arrangement, _record_cells,
                              _refine_config, _plane_steiner_sites, flat_point,
                              _arena, evidence);
      round = close_frontier(evidence,
                             take_entrance(close_round(retried, evidence)));
      if (round != plane_round_result::retry)
        break;
      if (_created_class.size() == stated_created &&
          _merges.size() == stated_merges &&
          frontier.size() >= stated_frontier &&
          answered.count == stated_answered)
        ++stalled_run;
      else
        stalled_run = 0;
      if (stalled_run == no_progress_budget) {
        // the planes still refusing are ALREADY published — close_plane_round
        // states the failed set through its one producer every round — so the
        // guard publishes nothing and only drops the stale product a stalled
        // carrier still holds
        for (const auto plane : frontier)
          retire_stalled_plane<Index, Int>(world, plane, _arena);
        _census.stalled_planes += frontier.size();
        round = plane_round_result::done;
        break;
      }
      stated_created = _created_class.size();
      stated_merges = _merges.size();
      stated_frontier = frontier.size();
      stated_answered = answered.count;
    }
    return round;
  }

  /// THE WAVE ENTRANCE: promote the source faces a round's world-tier splits
  /// reached, seat their sides in this arrangement's own tier, and hand the
  /// loop a frontier that carries them.
  ///
  /// THE ENTRANT GETS A NEW SPAN in the local tables — every edge that is its
  /// alone. The ONE row that cannot go there is the SHARED edge: its
  /// canonical group already exists — the neighbour's instance and the wave's
  /// split live on it, the group IS the cross-face join, and a second group
  /// for one wall is the twin-wall defect — so that row JOINS the existing
  /// group as one more instance, riding the port that takes it.
  ///
  /// A face enters WHOLE, so a face holding one side this tier cannot seat is
  /// DECLINED. Both answers are recorded, because either way the face states
  /// its sides somewhere the world's own span cannot see, and the discovery
  /// must never offer it again — that record is also what ends the entrance.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename ApplyToForm>
  auto enter_wave_entrants(World &world,
                           const plane_round_evidence<Index, Int> &evidence,
                           const tf::buffer<Index> &entrants,
                           plane_wave_answered<Index> &answered,
                           const GetBasePoint &get_base_point,
                           const GetOriginalPoint &get_original_point,
                           const ApplyToForm &apply_to_form,
                           tf::buffer<Index> &refusals,
                           tf::buffer<Index> &frontier) -> plane_round_result {
    tf::buffer<descriptor_t> descriptors;
    tf::buffer<frame_t> frames;
    tf::buffer<std::int8_t> orientations;
    tf::intersect::graph::plane_tables<Index, Int> raw_entrants;
    tf::buffer<Index> world_group_of;
    tf::buffer<int> route_of;
    tf::buffer<char> seatable;
    const auto expand_entrant_side = [&](const def_t &parent,
                                         tf::buffer<def_t> &output) {
      state_plane_weld_entrant_side<Index>(world, world.vertex_offsets(),
                                           parent, output);
    };
    // AT THIS MOMENT ONLY THE WORLD'S SPLITS ARE APPLIED to the group the
    // shared side names — the wave has ordered its own and stated none — so
    // the entrant's side lifts through exactly the pieces its neighbour's
    // instance holds, and the existing lift is already the right one.
    const auto state_entrant_tables = [&](const auto &faces) {
      make_plane_weld_entrant_tables<Index, Int>(
          faces, world.n_faces(), _merges, world.vertex_offsets(),
          world.face_offsets(), get_base_point, apply_to_form,
          expand_entrant_side, descriptors, frames, raw_entrants);
      classify_plane_wave_entrants(world, raw_entrants, _group_router,
                                   world_group_of, route_of);
    };

    state_entrant_tables(tf::make_range(entrants));
    // the verdict is asked ONCE, of the faces the discovery offered: a
    // restate below states the tables for faces already answered seatable
    state_plane_wave_entrant_seats(raw_entrants, route_of, seatable);
    tf::buffer<Index> seated;
    tf::sequenced_generate(
        tf::make_sequence_range(entrants.size()), seated,
        [&](std::size_t at, tf::buffer<Index> &out) {
          if (seatable[at] != char(0))
            out.push_back(entrants[at]);
        },
        tf::checked);
    _census.declined_entrants += entrants.size() - seated.size();
    // either answer ends the question for the face: a promotion states it
    // in this tier, a decline states that this tier cannot
    for (const auto entrant : entrants)
      answered.answer(world.face_offsets(), entrant);
    // THE ROUND IS HANDED BACK EITHER WAY, so its frontier is the one every
    // discarded round states: the carriers that saw something. They state it
    // again, and with every face now answered for the entrance does not fire
    // twice on it.
    state_plane_round_frontier(evidence, frontier);
    if (seated.size() == 0)
      return plane_round_result::retry;
    // the tables are positional in the faces they were stated for, so a
    // decline restates them for the faces that remain
    if (seated.size() != entrants.size())
      state_entrant_tables(tf::make_range(seated));

    make_plane_refinement_entrant_orientations<Index>(
        descriptors, frames, get_original_point, apply_to_form, orientations);
    const auto plane_base = world.n_planes();
    const auto plane_of_face = [&world](Index face) {
      return world.plane_of_face(face);
    };
    if (!seat_plane_wave_entrants(world, raw_entrants, world_group_of,
                                  route_of, plane_base, plane_of_face,
                                  _local_tables, _plane_ticket,
                                  _group_router))
      return plane_round_result::aborted;
    tf::core::append(descriptors, _promoted_descriptors);
    tf::core::append(frames, _promoted_frames);
    tf::core::append(orientations, _promoted_orientations);
    // THE COMPOSITION RULE: the promotion constructor reads its base
    // extents off the value it is handed, never off that value's current
    // ones, so appending to the three buffers and rebuilding FROM THE VALUE
    // IN HAND composes. Layering one promoted value on another would
    // restate an extent that is already frozen.
    world = World(world, _promoted_descriptors, _promoted_frames,
                  _promoted_orientations);
    _arena.grow_planes(world.n_planes());
    _arena.grow_faces(world.n_faces());
    refusals.reallocate_and_initialize(std::size_t(world.n_planes()), Index(0));
    _census.entrant_planes += seated.size();

    // the entrants join the frontier DIRTY — they hold no product — beside
    // the carriers the discarded round saw something on; their planes stamp
    // past every world plane, so the frontier stays ascending and unique
    for (Index at = 0; at < Index(seated.size()); ++at)
      frontier.push_back(plane_base + at);
    return plane_round_result::retry;
  }

  template <typename World> auto publish_products(const World &world) -> void {
    materialize_plane_products(
        world, _arena, _record_arrangement, _record_cells, _base_created,
        n_flat_points(), _plane_offsets, _triangles, _slot_parents,
        _corner_subs, _coplanar_of, _coplanar, _stacked, _triangle_cells,
        _face_range, _direct_created_points, _created_class, _census,
        _refinement_census);
    // A carrier a weld did not re-triangulate holds the corners it was built
    // with; the closed table is what makes them the identities this
    // arrangement publishes. A weld can collapse a retained triangle whole —
    // an unconstrained diagonal between merged identities — so the store
    // compacts before anything downstream reads it.
    rewrite_plane_flat_triangles(_triangles, _merges, n_flat_points());
    if (_merges.size() != 0) {
      // The recorded cells are stated against the store the emission produced;
      compact_plane_weld_triangles(_triangles, _slot_parents, _corner_subs,
                                   _coplanar_of, _coplanar, _stacked,
                                   _triangle_cells, _plane_offsets,
                                   _face_range);
      _census.triangles = _triangles.size();
    }
    // THE PIECE SPACE IS ONE REQUEST'S PRODUCT: unasked, no slot carries a
    // ticket and no definition span is owed one
    if (_record_arrangement && _plane_ticket.size() != 0 &&
        !publish_plane_final_pieces(world, _local_tables, _group_router,
                                    _plane_ticket, _vertex_offsets,
                                    n_flat_points(), _immutable_canon_extent,
                                    _final_piece_definitions, _triangles,
                                    _slot_parents, _plane_offsets))
      fail_every_plane(world);
  }

  /// THE THIRD STATEMENT PRODUCER'S QUESTION, asked of the arrangement as it
  /// stands: every plane is read from the one tier its ticket names, through
  /// this arrangement's own exact readers.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint>
  auto discover_refinement(const World &world,
                           const tf::cdt_refine_config &config,
                           const GetBasePoint &get_base_point,
                           const GetOriginalPoint &get_original_point) const
      -> plane_refinement_plan<Index, Int> {
    return make_plane_refinement_plan(
        world, _local_tables, _plane_ticket, _vertex_offsets, n_flat_points(),
        config, flat_point_of(get_base_point, get_original_point),
        exact_point_of(get_base_point, get_original_point));
  }

  /// The discovery, TAKEN: its census becomes this build's, its sites become
  /// the arrangement's, and its refusals join the published set. Discovery
  /// states the sites; emission is where they become identities, and only the
  /// ones its own triangulation keeps — so the census counts none yet.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint>
  auto take_refinement_discovery(const World &world,
                                 const tf::cdt_refine_config &config,
                                 const GetBasePoint &get_base_point,
                                 const GetOriginalPoint &get_original_point)
      -> plane_refinement_plan<Index, Int> {
    auto initial = discover_refinement(world, config, get_base_point,
                                       get_original_point);
    _refinement_census = initial.census;
    _refinement_census.steiners = 0;
    _plane_steiner_sites = std::move(initial.steiners);
    append_refinement_refused(initial.refused_planes);
    return initial;
  }

  /// THE THREE READERS every step of this build resolves a coordinate
  /// through: one flat identity, one (tag, id) pair, and one class name at its
  /// dyadic parameter. The steps are free operations, so the arrangement binds
  /// its own tables into them here and stays the only authority on what an
  /// identity of this tier means.
  template <typename GetBasePoint, typename GetOriginalPoint>
  auto flat_point_of(const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point) const {
    return [this, &get_base_point, &get_original_point](Index flat) {
      return point_of_flat(flat, get_base_point, get_original_point);
    };
  }

  template <typename GetBasePoint, typename GetOriginalPoint>
  auto exact_point_of(const GetBasePoint &get_base_point,
                      const GetOriginalPoint &get_original_point) const {
    return [this, &get_base_point, &get_original_point](std::int16_t tag,
                                                        Index id) {
      return final_point(tag, id, get_base_point, get_original_point);
    };
  }

  template <typename GetBasePoint, typename GetOriginalPoint>
  auto name_point_of(const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point) const {
    return [this, &get_base_point, &get_original_point](param_t parameter,
                                                        const name_t &name) {
      return form_position(parameter, name, get_base_point, get_original_point);
    };
  }

  /// One identity created by a wave of this arrangement: the exact position of
  /// the class that named it, which is a blend of its root's endpoints at the
  /// class's own dyadic parameter. Positions are never stored.
  template <typename GetPoint, typename GetMeshPoint>
  auto class_position(Index cls, Index owner, const GetPoint &get_point,
                      const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return form_position(_class_t[std::size_t(cls)],
                         _class_name[std::size_t(cls)], owner, get_point,
                         get_mesh_point);
  }

  /// A created point is (name, parameter) and nothing else: a landing reads
  /// the identity its name carries, and a crossing blends its birth edge's
  /// two endpoints — feature[1] and feature[2], the same identities the
  /// births matched the edge by — at the dyadic parameter. No group id
  /// enters persistence; identities are the only stable currency.
  template <typename GetPoint, typename GetMeshPoint>
  auto form_position(param_t parameter, const name_t &name,
                     const GetPoint &get_point,
                     const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return form_position(parameter, name, Index(-1), get_point, get_mesh_point);
  }

  template <typename GetPoint, typename GetMeshPoint>
  auto form_position(param_t parameter, const name_t &name, Index owner,
                     const GetPoint &get_point,
                     const GetMeshPoint &get_mesh_point) const -> pt3_t {
    const auto flat_point = [&](Index flat) {
      if (flat >= n_flat_points())
        return final_point(std::int16_t(-1), flat - n_flat_points(), owner,
                           get_point, get_mesh_point);
      const auto tag = tf::exact::tag_of_flat_vertex(_vertex_offsets, flat);
      return final_point(std::int16_t(tag),
                         flat - _vertex_offsets[std::size_t(tag)], owner,
                         get_point, get_mesh_point);
    };
    if (name.feature[0] == Index(0))
      return final_point(std::int16_t(-1), name.feature[1], owner, get_point,
                         get_mesh_point);
    if (name.feature[0] == Index(1))
      return flat_point(name.feature[1]);
    const auto lo = flat_point(name.feature[1]);
    const auto hi = flat_point(name.feature[2]);
    pt3_t position;
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      position[coordinate] =
          tf::exact::dyadic_blend(lo[coordinate], hi[coordinate], parameter);
    return position;
  }

  template <typename GetPoint, typename GetMeshPoint>
  auto final_point(std::int16_t tag, Index id, const GetPoint &get_point,
                   const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return final_point(tag, id, Index(-1), get_point, get_mesh_point);
  }

  template <typename GetPoint, typename GetMeshPoint>
  auto final_point(std::int16_t tag, Index id, Index owner,
                   const GetPoint &get_point,
                   const GetMeshPoint &get_mesh_point) const -> pt3_t {
    // Names persist across waves, so every endpoint first speaks the one
    // closed identity table. While evaluating a minted identity, however, its
    // own carrier still speaks the pre-mutation sources that define the mint.
    bool frozen_source = false;
    if (_merges.size() != 0) {
      std::int16_t merged_tag;
      Index merged_id;
      tf::intersect::graph::plane_merged_endpoint(
          _merges, _vertex_offsets, tag, id, merged_tag, merged_id);
      frozen_source = merged_tag < 0 && merged_id == owner;
      if (!frozen_source) {
        tag = merged_tag;
        id = merged_id;
      }
    }
    if (tag >= 0 || id < _base_created)
      return get_point(tag, id);
    const auto cls = _created_class[std::size_t(id - _base_created)];
    // a plane-local point is a direct exact form; everything else blends
    // the class that named it
    return cls < Index(0)
               ? _direct_created_points[std::size_t(-cls - Index(1))]
               : class_position(cls, frozen_source ? owner : id, get_point,
                                get_mesh_point);
  }

  /// THE one way a coordinate is read here: an original vertex answers from
  /// its mesh, a point of the arrangement that entered from its table, and an
  /// identity a wave of this tier created from the class that named it.
  template <typename GetPoint, typename GetMeshPoint>
  auto point_of_flat(Index flat, const GetPoint &get_point,
                     const GetMeshPoint &get_mesh_point) const -> pt3_t {
    if (flat >= n_flat_points())
      return final_point(std::int16_t(-1), flat - n_flat_points(), get_point,
                         get_mesh_point);
    const auto tag = tf::exact::tag_of_flat_vertex(_vertex_offsets, flat);
    return final_point(std::int16_t(tag),
                       flat - _vertex_offsets[std::size_t(tag)], get_point,
                       get_mesh_point);
  }

  template <typename World> auto fail_every_plane(const World &world) -> void {
    _failed.allocate(std::size_t(world.n_planes()));
    for (Index plane = 0; plane < world.n_planes(); ++plane)
      _failed[std::size_t(plane)] = plane;
    _census.failed_planes = _failed.size();
    // a terminal failure owes nobody its scratch: the dirty tier resets to
    // the never-happened surface
    _local_tables.defs().clear();
    _local_tables.def_offsets().clear();
    _local_tables.edges().clear();
    _local_tables.n_canon() = Index(0);
    _group_router.clear();
    _plane_ticket.clear();
    materialize_empty(world);
  }

  template <typename World> auto materialize_empty(const World &world) -> void {
    _triangles.clear();
    _slot_parents.clear();
    _corner_subs.clear();
    _coplanar_of.clear();
    _coplanar.clear();
    _stacked.clear();
    _triangle_cells.clear();
    _final_piece_definitions.clear();
    _plane_offsets.allocate(std::size_t(world.n_planes()) + 1);
    std::fill(_plane_offsets.begin(), _plane_offsets.end(), Index(0));
    _face_range.allocate(std::size_t(world.n_faces()));
    std::fill(_face_range.begin(), _face_range.end(),
              std::array<Index, 2>{Index(0), Index(0)});
    _census.triangles = 0;
  }

  tf::buffer<std::array<Index, 3>> _triangles;
  tf::buffer<std::array<Index, 3>> _slot_parents;
  tf::buffer<std::array<tf::topo_id<short>, 3>> _corner_subs;
  tf::buffer<Index> _plane_offsets;
  tf::buffer<std::array<Index, 2>> _face_range;
  tf::buffer<Index> _coplanar_of;
  tf::buffer<coplanar_t> _coplanar;
  tf::buffer<char> _stacked;
  tf::buffer<Index> _failed;
  tf::buffer<Index> _triangle_cells;

  plane_arrangement_arena<Index, Int> _arena;
  tf::intersect::graph::plane_tables<Index, Int> _local_tables;
  tf::buffer<Index> _group_router;
  tf::buffer<Index> _plane_ticket;
  /// Closed, so one search states what any retired identity speaks now. Empty
  /// while nothing has been welded.
  tf::buffer<std::array<Index, 3>> _merges;
  tf::offset_block_buffer<Index, def_t> _final_piece_definitions;
  tf::buffer<Index> _created_class;
  // plane-local refined points: exact direct forms, owner-aligned; a
  // negative _created_class entry indexes them
  tf::buffer<pt3_t> _direct_created_points;
  /// Discovery's interior sites, by plane. Positions only — emission is where
  /// they become identities, if the plane's own triangulation keeps them.
  tf::offset_block_buffer<Index, pt3_t> _plane_steiner_sites;
  tf::buffer<Index> _refinement_refused;
  tf::buffer<descriptor_t> _promoted_descriptors;
  tf::buffer<frame_t> _promoted_frames;
  tf::buffer<std::int8_t> _promoted_orientations;
  tf::buffer<param_t> _class_t;
  tf::buffer<name_t> _class_name;
  tf::range<const Index *, tf::dynamic_size> _vertex_offsets;
  tf::cdt_refine_config _refine_config;
  Index _base_created = 0;
  Index _base_faces = 0;
  Index _base_planes = 0;
  Index _immutable_canon_extent = 0;
  bool _refined = false;
  bool _record_arrangement = false;
  bool _record_cells = false;
  plane_arrangement_census _census;
  plane_refinement_census _refinement_census;
};

} // namespace tf::arrangement
