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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/none.hpp"
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
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/final_identity_point.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"
#include "../../intersect/graph/local_arrangement.hpp"
#include "../../intersect/graph/near_identity_rows.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../intersect/graph/rebuild_plane_blocks.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../identity/compose_local_moves.hpp"
#include "../identity/extent_shift.hpp"
#include "../identity/identity_substrate.hpp"
#include "./advance_plane_wave.hpp"
#include "./canonicalize_plane_wave.hpp"
#include "./classify_plane_entrant_origins.hpp"
#include "./classify_plane_wave_entrants.hpp"
#include "./close_plane_lazy_round.hpp"
#include "./close_plane_round.hpp"
#include "./commit_plane_wave_blocks.hpp"
#include "./discover_plane_refinement_entrants.hpp"
#include "./discover_plane_wave_entrants.hpp"
#include "./elect_standing_plane_steiners.hpp"
#include "./make_plane_refinement_cut_mask.hpp"
#include "./make_plane_refinement_evidence.hpp"
#include "./make_plane_refinement_overlay.hpp"
#include "./make_plane_refinement_physical_splits.hpp"
#include "./make_plane_refinement_plan.hpp"
#include "./make_plane_weld_entrant_tables.hpp"
#include "./map_plane_refinement_splits.hpp"
#include "./materialize_plane_products.hpp"
#include "./plane_arrangement_arena.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_arrangement_store.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_refinement_plan.hpp"
#include "./plane_round_evidence.hpp"
#include "./plane_tier_definitions.hpp"
#include "./plane_wave_answered.hpp"
#include "./plane_world.hpp"
#include "./port_plane_diff.hpp"
#include "./propagate_plane_refinement_rings.hpp"
#include "./propose_plane_split_pieces.hpp"
#include "./publish_plane_final_pieces.hpp"
#include "./retire_stalled_plane.hpp"
#include "./seat_plane_entrant_blocks.hpp"
#include "./seat_plane_promoted_faces.hpp"
#include "./seat_plane_wave_entrants.hpp"
#include "./state_plane_edge_union.hpp"
#include "./state_plane_promoted_members.hpp"
#include "./state_plane_round_frontier.hpp"
#include "./state_plane_weld_entrant_side.hpp"
#include "./triangulate_plane_round.hpp"
#include "./weld_plane_triangles.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
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
/// to. It owns no identity table: it reads the local layer and states its own
/// through the verbs of the substrate it is handed (@ref
/// tf::arrangement::identity_substrate).
template <typename Index, typename Int> class plane_arrangement {
public:
  using pt3_t = tf::exact::pt3<Int>;
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using name_t = plane_recovery_name<Index, param_t>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;
  using identities_t = identity_substrate<Index, Int>;

  /// A junction the triangulation found, named by the endpoint identities of
  /// the two exact pieces it joins.
  static constexpr Index crossing_name_kind = 5;
  static constexpr Index refinement_name_kind = 6;
  /// The verdicts a diff's router fold joins past a route (`-1` the world's,
  /// `-2` retired, a local group otherwise): the world's beside retired
  /// roots alone, two takings, and a taking beside the world's or a retired
  /// root.
  static constexpr Index retired_route_fold = -4;
  static constexpr Index taken_route_fold = -5;
  static constexpr Index world_route_fold = -6;

  /// Every triangle of the arrangement, contiguous per plane and inside
  /// a plane contiguous per member. A corner is a FLAT identity:
  /// `< n_flat_points()` an original vertex in the tags' flat space,
  /// past it a point of the intersection space.
  auto triangles() const { return tf::make_range(_store.triangles); }
  /// Per triangle, per slot s (corner s -> corner s + 1): the canonical
  /// piece it belongs to, or -1 for a filler diagonal. Recorded only when
  /// `record_triangle_arrangement()` was asked before the build.
  auto slot_parents() const { return tf::make_range(_store.slot_parents); }
  /// Per triangle, per corner: where the corner sits on the emitting
  /// member's own polygon — its corner ordinal, the original side it lies
  /// on, or the interior.
  auto corner_subs() const { return tf::make_range(_store.corner_subs); }
  auto immutable_piece_extent() const -> Index {
    return _immutable_canon_extent;
  }
  /// The exclusive upper bound of the final-piece ticket address space. A
  /// PA-owned suffix piece can shadow an immutable prefix identity, so this is
  /// not the cardinality of live pieces.
  auto final_piece_ticket_extent() const -> Index {
    return _immutable_canon_extent +
           Index(_store.final_piece_definitions.size());
  }
  /// The complete definition span named by a non-filler slot ticket. The
  /// immutable prefix stays caller-owned; PA-owned suffix spans include every
  /// inactive immutable instance sharing the final piece.
  template <typename Immutable>
  auto piece_definitions(const Immutable &immutable, Index ticket) const {
    if constexpr (tf::core::verification) {
      tf::core::verify(immutable.n_canon() == _immutable_canon_extent,
                       "a piece ticket reads the tier it was built over");
      tf::core::verify(ticket >= Index(0) &&
                           ticket < final_piece_ticket_extent(),
                       "a piece ticket lies in the final ticket space");
    }
    if (ticket < _immutable_canon_extent) {
      const auto span = immutable.canon_group(ticket);
      return tf::make_range(span.begin(), span.end());
    }
    const auto span = _store.final_piece_definitions[std::size_t(
        ticket - _immutable_canon_extent)];
    return tf::make_range(span.begin(), span.end());
  }
  /// The local block each plane's definitions resolve through, `-1` for a
  /// plane that still reads the world. Empty while every plane does.
  auto plane_tickets() const { return tf::make_range(_plane_ticket); }
  /// One local plane block's definitions, in key order, each read from
  /// the tier that owns it (grain law 3): a block this arrangement holds
  /// still names the world's own rows for every group no wave changed.
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
    return tf::slice(tf::make_range(_store.triangles),
                     std::size_t(_store.plane_offsets[std::size_t(p)]),
                     std::size_t(_store.plane_offsets[std::size_t(p) + 1]));
  }
  auto plane_range(Index p) const -> std::array<Index, 2> {
    return {_store.plane_offsets[std::size_t(p)],
            _store.plane_offsets[std::size_t(p) + 1]};
  }
  /// The plane that emitted triangle `t`; triangles are plane-contiguous.
  auto plane_of_triangle(Index t) const -> Index {
    return Index(std::upper_bound(_store.plane_offsets.begin() + 1,
                                  _store.plane_offsets.end(), t) -
                 (_store.plane_offsets.begin() + 1));
  }
  /// A graph face's own emitted span: for a stack member, the subset it
  /// covers, in that member's own winding.
  auto face_range(Index face) const -> std::array<Index, 2> {
    return _store.face_range[std::size_t(face)];
  }
  /// Per triangle: -1 for a survivor, else its row in
  /// `coplanar_descriptors()`. Recorded only when
  /// `record_triangle_arrangement()` was asked before the build.
  auto coplanar_of() const { return tf::make_range(_store.coplanar_of); }
  auto coplanar_descriptors() const { return tf::make_range(_store.coplanar); }
  /// Per triangle: its region is covered by more than one member. Recorded
  /// only when `record_triangle_arrangement()` was asked before the build.
  auto stacked() const { return tf::make_range(_store.stacked); }
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
  /// The faces promoted past the world's own: face `n_base_faces() + i` is
  /// stated by row `i` of `promoted_descriptors`, its winding in its plane's
  /// frame by `promoted_orientations`, and its plane by `promoted_plane_of` —
  /// a standing plane its name holds, or one the promotion founded. Plane
  /// `n_base_planes() + j` is framed by row `j` of `promoted_frames` and
  /// named by row `j` of the plane tier's `plane_names`, and
  /// `promoted_member_offsets` counts, per plane, the promoted faces seated
  /// on it after its own. A consumer reads the promoted side of every carrier
  /// fact here rather than rederiving it.
  auto promoted_descriptors() const {
    return tf::make_range(_promoted_descriptors);
  }
  auto promoted_frames() const { return tf::make_range(_promoted_frames); }
  auto promoted_orientations() const {
    return tf::make_range(_promoted_orientations);
  }
  auto promoted_plane_of() const { return tf::make_range(_promoted_plane_of); }
  auto promoted_member_offsets() const {
    return tf::make_range(_promoted_member_offsets);
  }
  /// The world extents this build stood on. A face, plane or created
  /// identity below them is the world's own; this value appended everything
  /// at or past them.
  auto n_base_faces() const -> Index { return _base_faces; }
  auto n_base_planes() const -> Index { return _base_planes; }
  auto n_base_created() const -> Index { return _base_created; }
  /// Per triangle: the plane-local 2-cell it came out of — the block a walk
  /// that never crosses a constraint reaches. Cells meet only at pieces, so
  /// `(plane, cell)` names the coarse carrier a component flood walks, and a
  /// stack's members share the cells they cover. Recorded only when
  /// `record_triangle_cells()` was asked before the build; a weld that
  /// compacted the store leaves it empty.
  auto triangle_cells() const { return tf::make_range(_store.triangle_cells); }
  auto n_planes() const -> Index {
    return _store.plane_offsets.size() == 0
               ? Index(0)
               : Index(_store.plane_offsets.size()) - Index(1);
  }
  auto n_faces() const -> Index { return Index(_store.face_range.size()); }
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

  /// Ask every recovery wave for its near table at `window` (@ref
  /// close_plane_recovery_identities); the request stands until it is
  /// withdrawn, and a diff value inherits it.
  auto record_near_identities(Int window) -> void { _near_window = window; }
  auto near_window() const -> Int { return _near_window; }
  /// The near table the waves stated, deferred (@ref
  /// tf::intersect::graph::near_identity_rows), one block per wave in this
  /// value's currency; a reader resolves a retired member through the
  /// substrate.
  auto near_rows() const
      -> const tf::intersect::graph::near_identity_rows<Index, pt3_t> & {
    return _near_rows;
  }

  auto clear() -> void {
    _store.clear();
    _failed.clear();
    _near_rows.clear();
    clear_definition_tier();
    _plane_steiner_sites.clear();
    _refinement_refused.clear();
    _promoted_descriptors.clear();
    _promoted_frames.clear();
    _promoted_orientations.clear();
    _promoted_plane_of.clear();
    _promoted_member_offsets.clear();
    _promoted_members.clear();
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
  /// created extent and the flat prefix come from the caller, and the
  /// identities the waves state enter `identities` as its plane layer.
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
  auto build(plane_world<Policy> &world, identities_t &identities,
             Index n_base_created, const GetBasePoint &get_base_point,
             const GetOriginalPoint &get_original_point,
             const VertexOffsets &vertex_offsets,
             const ApplyToForm &apply_to_form = tf::none) -> void {
    if (!build_core(world, identities, n_base_created, get_base_point,
                    get_original_point, vertex_offsets, apply_to_form))
      return;
    publish_products(world, identities);
  }

  /// THE REFINED BUILD: complete refinement's exact split, four-ring carrier
  /// and plane-local Steiner pre-step, then hand that prepared state to the
  /// ordinary stock path once. No stock triangulation or refusal recovery
  /// occurs inside the pre-step.
  template <typename Policy, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets>
  auto build_refined(plane_world<Policy> &world, identities_t &identities,
                     Index n_base_created, const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point,
                     const VertexOffsets &vertex_offsets,
                     const tf::cdt_refine_config &config) -> void {
    build_refined_closed_world(world, identities, n_base_created,
                               get_base_point, get_original_point,
                               vertex_offsets, config);
  }

  /// The build on a local arrangement: the world reads the local layer of
  /// `identities`, and this tier states its own as the plane layer.
  template <typename RealType, typename GetOriginalPoint,
            typename ApplyToForm = tf::none_t>
  auto build(const tf::intersect::graph::local_arrangement<Index, RealType, Int>
                 &arrangement,
             identities_t &identities,
             const GetOriginalPoint &get_original_point,
             const ApplyToForm &apply_to_form = tf::none) -> void {
    build_on_local_arrangement(
        arrangement, identities, get_original_point,
        [&](auto &world, const auto &get_base_point) {
          build(world, identities, world.n_created_points(), get_base_point,
                get_original_point, world.vertex_offsets(), apply_to_form);
        });
  }

  template <typename RealType, typename GetOriginalPoint, typename ApplyToForm>
  auto build_refined(
      const tf::intersect::graph::local_arrangement<Index, RealType, Int>
          &arrangement,
      identities_t &identities, const GetOriginalPoint &get_original_point,
      const ApplyToForm &apply_to_form, const tf::cdt_refine_config &config)
      -> void {
    build_on_local_arrangement(arrangement, identities, get_original_point,
                               [&](auto &world, const auto &get_base_point) {
                                 run_refined(world, identities,
                                             get_original_point, apply_to_form,
                                             get_base_point, config);
                               });
  }

  template <typename RealType, typename GetOriginalPoint, typename ApplyToForm>
  auto build(const tf::intersect::graph::local_arrangement<Index, RealType, Int>
                 &arrangement,
             identities_t &identities,
             const GetOriginalPoint &get_original_point,
             const ApplyToForm &apply_to_form,
             const tf::cdt_refine_config &config) -> void {
    build_refined(arrangement, identities, get_original_point, apply_to_form,
                  config);
  }

  /// `prior` carried onto the restated world and rewaved from the statements'
  /// raw evidence (@ref tf::arrangement::compose_local_moves); the plane
  /// layer of `identities` enters as `prior`'s, leaves as this value's; empty
  /// images are identity.
  template <typename RealType, typename GetOriginalPoint,
            typename ApplyToForm = tf::none_t>
  auto
  build_diff(const tf::intersect::graph::local_arrangement<Index, RealType, Int>
                 &arrangement,
             identities_t &identities, const plane_arrangement &prior,
             tf::buffer<std::array<Index, 3>> &&carve_moved,
             const tf::buffer<std::array<Index, 3>> &census_moved,
             const tf::buffer<plane_flat_weld<Index>> &plane_welds, bool carved,
             const tf::buffer<Index> &row_images,
             const tf::buffer<Index> &canon_images,
             const tf::buffer<Index> &dying_faces,
             tf::buffer<Index> &&dirty_planes,
             const GetOriginalPoint &get_original_point,
             const ApplyToForm &apply_to_form = tf::none) -> void {
    auto world = make_plane_world(arrangement, identities);
    // the diff is a post-release consumer, so it reads the way the
    // exposure reads: anchors and dyadic parameters, never the released
    // intersection table
    const auto get_base_point = [&](std::int16_t tag, Index id) -> pt3_t {
      return tag >= 0
                 ? pt3_t(get_original_point(int(tag), id))
                 : arrangement.point_of(identities, id, get_original_point);
    };
    diff_core(world, identities, prior, std::move(carve_moved), census_moved,
              plane_welds, carved, row_images, canon_images, dying_faces,
              std::move(dirty_planes), get_base_point, get_original_point,
              apply_to_form);
  }

  /// Planes the refinement PRODUCER declined to plan for. Not a failure:
  /// the stock kernel triangulates them in the same pass, so the carrier
  /// holds an unrefined product and `failed()` stays the one surface that
  /// states an absent one.
  auto refinement_refused_planes() const {
    return tf::make_range(_refinement_refused);
  }

  /// Resolve one identity created by this tier from the exact authority that
  /// named it, through the plane layer this tier stated into `identities`.
  /// `get_base_point` resolves the immutable prefix; PA-owned birth edges,
  /// rootless names and direct plane-local forms are evaluated on demand
  /// without a point table.
  template <typename GetBasePoint, typename GetMeshPoint>
  auto resolve_created_point(const identities_t &identities, Index id,
                             const GetBasePoint &get_base_point,
                             const GetMeshPoint &get_mesh_point) const
      -> pt3_t {
    if constexpr (tf::core::verification)
      tf::core::verify(id >= _base_created &&
                           id < _base_created + identities.n_plane_created(),
                       "a created point resolves an identity this tier "
                       "minted");
    return final_point(identities, std::int16_t(-1), id, get_base_point,
                       get_mesh_point);
  }

  /// The statement of one identity this tier created, every endpoint it
  /// names read through `read(tag, id)` — the layer a reader evaluates at is
  /// its `read`: a plane-local point is a direct exact form, everything else
  /// the name of its class at the class's parameter.
  template <typename Read>
  auto statement_point(const identities_t &identities, Index id,
                       const Read &read) const -> pt3_t {
    const auto cls =
        identities.plane_created_classes()[std::size_t(id - _base_created)];
    if (cls < Index(0))
      return identities.plane_direct_points()[std::size_t(-cls - Index(1))];
    return name_position(identities.plane_class_ts()[std::size_t(cls)],
                         identities.plane_class_names()[std::size_t(cls)],
                         read);
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
      const identities_t &identities,
      const GetOriginalPoint &get_original_point, const Build &build_with)
      -> void {
    auto world = make_plane_world(arrangement, identities);
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
  auto apply_refinement_wave(const World &world, identities_t &identities,
                             const Splits &splits,
                             tf::buffer<plane_flat_weld<Index>> &&welds,
                             const GetBasePoint &get_base_point,
                             const GetOriginalPoint &get_original_point)
      -> bool {
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
        world, evidence, _local_tables, _plane_ticket, _group_router,
        identities, _vertex_offsets, n_flat_points(), _base_created,
        exact_point_of(identities, get_base_point, get_original_point),
        name_point_of(identities, get_base_point, get_original_point), _census,
        frontier, Int(0), _near_rows);
    return result != plane_wave_result::unsupported;
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets, typename ApplyToForm>
  auto fall_back_to_stock(World &world, identities_t &identities,
                          Index n_base_created,
                          const GetBasePoint &get_base_point,
                          const GetOriginalPoint &get_original_point,
                          const VertexOffsets &vertex_offsets,
                          const ApplyToForm &apply_to_form) -> void {
    if (build_core(world, identities, n_base_created, get_base_point,
                   get_original_point, vertex_offsets, apply_to_form))
      publish_products(world, identities);
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets>
  auto build_refined_closed_world(World &world, identities_t &identities,
                                  Index n_base_created,
                                  const GetBasePoint &get_base_point,
                                  const GetOriginalPoint &get_original_point,
                                  const VertexOffsets &vertex_offsets,
                                  const tf::cdt_refine_config &config) -> void {
    // refinement's splits are keyed by the canonical group and its discovery
    // reads a root's instances, so this arm needs the group space up front
    world.materialize();
    identities.clear_plane();
    if (!initialize(world, n_base_created, vertex_offsets))
      return;
    refine_at_emission(config);
    auto initial = take_refinement_discovery(
        world, identities, config, get_base_point, get_original_point);
    if (!apply_refinement_wave(world, identities, initial.splits,
                               std::move(initial.welds), get_base_point,
                               get_original_point)) {
      fall_back_to_stock(world, identities, n_base_created, get_base_point,
                         get_original_point, vertex_offsets, tf::none);
      return;
    }
    _refinement_census.refused_planes = _refinement_refused.size();
    if (!run_stock(world, identities, get_base_point, get_original_point,
                   tf::none))
      return;
    publish_products(world, identities);
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename ApplyToForm>
  auto diff_core(World &world, identities_t &identities,
                 const plane_arrangement &prior,
                 tf::buffer<std::array<Index, 3>> &&carve_moved,
                 const tf::buffer<std::array<Index, 3>> &census_moved,
                 const tf::buffer<plane_flat_weld<Index>> &plane_welds,
                 bool carved, const tf::buffer<Index> &row_images,
                 const tf::buffer<Index> &canon_images,
                 const tf::buffer<Index> &dying_faces,
                 tf::buffer<Index> &&dirty_planes,
                 const GetBasePoint &get_base_point,
                 const GetOriginalPoint &get_original_point,
                 const ApplyToForm &apply_to_form) -> void {
    // the plane layer `prior` stated stays in the substrate until this value
    // carries it into its own currency below
    const auto n_prior_created = identities.n_plane_created();
    if (!initialize(world, world.n_created_points(), world.vertex_offsets())) {
      identities.clear_plane();
      return;
    }
    _refined = prior._refined;
    _refine_config = prior._refine_config;
    _record_arrangement = prior._record_arrangement;
    _record_cells = prior._record_cells;
    _near_window = prior._near_window;
    // the one renumbering `prior` crosses into this value, and the moves the
    // statements stated in the local layer composed into this value's
    // evidence: the dirty set the statements touched gains what the world
    // founded and every witness the composition carries
    const auto extents = make_diff_extents(prior, n_prior_created, world);
    tf::buffer<plane_flat_weld<Index>> wave_welds;
    compose_local_moves(world.merges(), extents, n_flat_points(),
                        std::move(carve_moved), census_moved, plane_welds,
                        carved, wave_welds, dirty_planes);
    std::sort(dirty_planes.begin(), dirty_planes.end());
    dirty_planes.erase_till_end(
        std::unique(dirty_planes.begin(), dirty_planes.end()));

    // the plane layer in this value's currency before the evidence above
    // closes into it in the wave below: forward, then append
    identities.carry_plane(extents.created, n_flat_points(), crossing_name_kind,
                           refinement_name_kind);
    // the deferred table crosses as its rows, through the renumbering
    _near_rows = prior._near_rows;
    _near_rows.renumber(
        [&extents](Index id) { return extents.created.forward(id); });

    // the promotion restated: value copies, with the references the
    // promotion itself founded moved past the restated world's extents
    _promoted_descriptors = prior._promoted_descriptors;
    _promoted_frames = prior._promoted_frames;
    _promoted_orientations = prior._promoted_orientations;
    _promoted_plane_of = prior._promoted_plane_of;
    tf::parallel_for_each(
        _promoted_plane_of,
        [&](Index &plane) { plane = extents.planes.forward(plane); },
        tf::checked);
    if (_promoted_descriptors.size() != 0) {
      state_plane_promoted_members(
          _promoted_plane_of, _base_faces,
          _base_planes + Index(_promoted_frames.size()),
          _promoted_member_offsets, _promoted_members);
      world = World(world, _promoted_descriptors, _promoted_frames,
                    identities.founded_plane_names(), _promoted_orientations,
                    _promoted_plane_of, _promoted_member_offsets,
                    _promoted_members);
    }

    // the dirty planes' refinement reads the discovery's standing sites
    if (prior._plane_steiner_sites.size() != 0) {
      const auto n_prior_blocks = Index(prior._plane_steiner_sites.size());
      const auto prior_block = [&](Index p) {
        const auto old_p = extents.planes.inverse(p);
        return old_p < n_prior_blocks ? old_p : Index(-1);
      };
      auto &offsets = _plane_steiner_sites.offsets_buffer();
      auto &sites = _plane_steiner_sites.data_buffer();
      offsets.allocate(std::size_t(world.n_planes()) + 1);
      offsets[0] = Index(0);
      tf::parallel_for_each(
          tf::make_sequence_range(world.n_planes()),
          [&](Index p) {
            const auto old_p = prior_block(p);
            offsets[std::size_t(p) + 1] =
                old_p == Index(-1)
                    ? Index(0)
                    : Index(prior._plane_steiner_sites[std::size_t(old_p)]
                                .size());
          },
          tf::checked);
      for (std::size_t p = 1; p < offsets.size(); ++p)
        offsets[p] += offsets[p - 1];
      sites.allocate(std::size_t(offsets[offsets.size() - 1]));
      tf::parallel_for_each(
          tf::make_sequence_range(world.n_planes()), [&](Index p) {
            const auto old_p = prior_block(p);
            if (old_p == Index(-1))
              return;
            const auto block = prior._plane_steiner_sites[std::size_t(old_p)];
            std::copy(block.begin(), block.end(),
                      sites.begin() + std::ptrdiff_t(offsets[std::size_t(p)]));
          });
    }

    tf::buffer<Index> taken_planes;
    // a promoted plane holds its defs only in the local table, a dirty
    // plane re-reads the tier its ticket names — the prior's splits
    // included — and a local row's world references move through the
    // restatement's maps; the local canon space keeps its numbering
    if (prior._plane_ticket.size() != 0) {
      _local_tables = prior._local_tables;
      tf::parallel_for_each(
          _local_tables.defs(),
          [&](tf::intersect::graph::plane_edge_def<Index> &def) {
            if (def.point_tag_0 < 0)
              def.point_0 = extents.created.forward(def.point_0);
            if (def.point_tag_1 < 0)
              def.point_1 = extents.created.forward(def.point_1);
            def.face = extents.faces.forward(def.face);
          },
          tf::checked);
      // a reference translates through the images, and a reference the
      // restatement dropped — a row the welds collapsed or retired —
      // drops here on the same terms, as does a local row of a face the
      // carve replaced: kept counts per block, one prefix, and each block
      // writes its own slice
      {
        const auto image_of = [&](Index row) {
          return row_images.size() == 0
                     ? Index(-1) - row
                     : row_images[std::size_t(Index(-1) - row)];
        };
        const auto local_defs = tf::make_range(_local_tables.defs());
        const auto kept = [&](Index row) {
          if (row < Index(0))
            return image_of(row) != Index(-1);
          return !std::binary_search(dying_faces.begin(), dying_faces.end(),
                                     local_defs[std::size_t(row)].face);
        };
        auto &offsets = _local_tables.edges().offsets_buffer();
        auto &rows = _local_tables.edges().data_buffer();
        if (offsets.size() == 0)
          rows.clear();
        else {
          tf::buffer<Index> kept_offsets;
          tf::buffer<Index> kept_rows;
          tf::intersect::graph::rebuild_plane_blocks(
              offsets.size() - 1,
              [&](std::size_t block) {
                Index count = 0;
                for (auto at = std::size_t(offsets[block]);
                     at < std::size_t(offsets[block + 1]); ++at)
                  count += Index(kept(rows[at]));
                return count;
              },
              [&](std::size_t block, std::size_t begin) {
                auto write = begin;
                for (auto at = std::size_t(offsets[block]);
                     at < std::size_t(offsets[block + 1]); ++at) {
                  const auto row = rows[at];
                  if (!kept(row))
                    continue;
                  kept_rows[write++] =
                      row >= Index(0) ? row : Index(-1) - image_of(row);
                }
              },
              kept_offsets, kept_rows);
          offsets = std::move(kept_offsets);
          rows = std::move(kept_rows);
        }
      }
      if constexpr (tf::core::verification)
        tf::core::verify(extents.planes.base >= extents.planes.prior,
                         "a restatement founds planes and retires none");
      _plane_ticket.allocate(std::size_t(world.n_planes()));
      tf::parallel_for_each(
          tf::make_sequence_range(world.n_planes()),
          [&](Index np) {
            const auto p = extents.planes.inverse(np);
            _plane_ticket[std::size_t(np)] =
                p != Index(-1) && p < Index(prior._plane_ticket.size())
                    ? prior._plane_ticket[std::size_t(p)]
                    : Index(-1);
          },
          tf::checked);
      // a world row no image names is one the carve founded: it joins the
      // block of its carrier, and it is the world's instance of its group
      tf::buffer<char> imaged;
      tf::buffer<Index> founded;
      if (row_images.size() != 0) {
        const auto n_rows = Index(world.edge_defs().size());
        imaged.allocate_and_initialize(std::size_t(n_rows), char(0));
        tf::parallel_for_each(
            row_images,
            [&imaged](Index image) {
              if (image != Index(-1))
                imaged[std::size_t(image)] = char(1);
            },
            tf::checked);
        tf::sequenced_generate(
            tf::make_sequence_range(n_rows), founded,
            [&imaged](Index row, tf::buffer<Index> &out) {
              if (!imaged[std::size_t(row)])
                out.push_back(row);
            },
            tf::checked);
      }
      if (founded.size() != 0)
        found_plane_block_rows(world, imaged, dirty_planes);
      // a group several standing groups fold into keeps a link only when
      // they all state the same one, and is the world's again otherwise. The
      // verdict is a join over the set of routes a fold's constituents
      // state, so the routes group by image and each image joins its own
      // run. A retired root states no taking. Two takings are the wave's:
      // its weld surgery restates both keys and its canonicalize closes them.
      // The other two verdicts are consumed below — a retired fold is taken
      // when a carrier of it reads no block, and a world fold is taken and
      // fused with its takings
      const auto fold = [](Index link, Index route) {
        if (link == route || link == world_route_fold)
          return link;
        if (link == retired_route_fold)
          return route >= Index(0) ? world_route_fold : link;
        if (link == taken_route_fold)
          return route < Index(0) ? world_route_fold : link;
        if (link >= Index(0) && route >= Index(0))
          return taken_route_fold;
        return link >= Index(0) || route >= Index(0) ? world_route_fold
                                                     : retired_route_fold;
      };
      tf::buffer<std::array<Index, 2>> routes;
      tf::sequenced_generate(
          tf::make_sequence_range(prior._group_router.size()), routes,
          [&](std::size_t c, tf::buffer<std::array<Index, 2>> &out) {
            const auto image =
                canon_images.size() == 0 ? Index(c) : canon_images[c];
            if (image != Index(-1))
              out.push_back({image, prior._group_router[c]});
          },
          tf::checked);
      // a founded row reads the world, so a group this arrangement holds a
      // link of meets it as a world fold
      {
        const auto world_defs = world.edge_defs();
        tf::generic_generate(
            founded, routes,
            [&world_defs](Index row, tf::buffer<std::array<Index, 2>> &out) {
              out.push_back({world_defs[std::size_t(row)].id, Index(-1)});
            },
            tf::checked);
      }
      tbb::parallel_sort(routes.begin(), routes.end());
      _group_router.allocate_and_initialize(
          std::size_t(_immutable_canon_extent), Index(-1));
      tf::buffer<std::array<Index, 2>> folds;
      tf::sequenced_generate(
          tf::make_sequence_range(routes.size()), folds,
          [&](std::size_t at, tf::buffer<std::array<Index, 2>> &out) {
            const auto image = routes[at][0];
            if (at != 0 && routes[at - 1][0] == image)
              return;
            auto link = routes[at][1];
            for (auto next = at + 1;
                 next < routes.size() && routes[next][0] == image; ++next)
              link = fold(link, routes[next][1]);
            if (link == retired_route_fold || link == world_route_fold)
              out.push_back({image, link});
            _group_router[std::size_t(image)] =
                link < Index(-2) ? Index(-1) : link;
          },
          tf::checked);
      // a reference the restatement rewrote moved its key under a block
      // ordered by the prior's keys, and the restatement names every plane
      // such a row stands on, so exactly those carriers are ordered again
      const auto tier =
          make_plane_tier_definitions(world.tables(), _local_tables, true);
      auto &block_offsets = _local_tables.edges().offsets_buffer();
      auto &block_rows = _local_tables.edges().data_buffer();
      struct block_local_t {
        tf::buffer<std::array<Index, 5>> keyed;
      };
      tf::parallel_for_each(
          tf::make_range(dirty_planes),
          [&](Index plane, block_local_t &local) {
            const auto block = _plane_ticket[std::size_t(plane)];
            if (block == Index(-1))
              return;
            sort_plane_block_by_key(tier, block_rows,
                                    std::size_t(block_offsets[block]),
                                    std::size_t(block_offsets[block + 1]),
                                    local.keyed);
          },
          block_local_t{});
      // a retired fold holding a carrier no block names is one the
      // statements below cannot reach, and a world fold is one key this
      // arrangement already holds a group of, so the take moves either local
      // with every carrier of it first. A carrier a retired fold's take ports
      // re-enters the wave; a world fold's take restates no constraint
      tf::buffer<Index> taken_folds;
      tf::buffer<Index> carriers;
      for (const auto &row : folds) {
        const auto group = row[0];
        const bool retired = row[1] == retired_route_fold;
        bool reached = retired;
        for (const auto &instance : world.canon_group(group))
          reached = reached &&
                    _plane_ticket[std::size_t(world.plane_of_face(
                        instance.face))] != Index(-1);
        if (reached)
          continue;
        taken_folds.push_back(group);
        for (const auto &instance : world.canon_group(group)) {
          const auto plane = world.plane_of_face(instance.face);
          carriers.push_back(plane);
          if (retired && _plane_ticket[std::size_t(plane)] == Index(-1))
            taken_planes.push_back(plane);
        }
      }
      if (taken_folds.size() != 0) {
        std::sort(carriers.begin(), carriers.end());
        carriers.erase_till_end(std::unique(carriers.begin(), carriers.end()));
        std::sort(taken_planes.begin(), taken_planes.end());
        taken_planes.erase_till_end(
            std::unique(taken_planes.begin(), taken_planes.end()));
        if (!port_plane_diff(world.tables(), tf::make_range(carriers),
                             taken_folds, _local_tables, _plane_ticket,
                             _group_router)) {
          fail_every_plane(world);
          return;
        }
      }
      // a world fold's take and its live links state one key on two shelves,
      // so they enter the canonicalize as the participants a weld surgery's
      // changed groups are: their class mints one group, and every one of
      // them retires through the losers into it
      tf::buffer<Index> participants;
      for (const auto &row : folds) {
        if (row[1] != world_route_fold)
          continue;
        participants.push_back(_group_router[std::size_t(row[0])]);
        for (auto at = std::lower_bound(routes.begin(), routes.end(),
                                        std::array<Index, 2>{row[0], Index(0)});
             at != routes.end() && (*at)[0] == row[0]; ++at)
          participants.push_back((*at)[1]);
      }
      if (participants.size() != 0) {
        std::sort(participants.begin(), participants.end());
        participants.erase_till_end(
            std::unique(participants.begin(), participants.end()));
        const tf::buffer<plane_split_piece_layout<Index>> layout;
        const tf::buffer<def_t> proposals;
        const tf::buffer<Index> standing_probe;
        const tf::buffer<Index> carrier_planes;
        tf::buffer<Index> final_row;
        tf::buffer<std::array<Index, 2>> losers;
        const auto plane_of_face = [&world](Index face) {
          return world.plane_of_face(face);
        };
        if (!canonicalize_plane_wave(world.tables(), layout, proposals,
                                     standing_probe, _plane_ticket,
                                     plane_of_face, _local_tables,
                                     _group_router, final_row, losers,
                                     participants) ||
            !commit_plane_wave_blocks(world.tables(), layout, final_row,
                                      carrier_planes, losers, plane_of_face,
                                      _local_tables, _plane_ticket)) {
          fail_every_plane(world);
          return;
        }
      }
    }

    _arena.reset(world.n_planes(), world.n_faces());

    // the statement enters as the wave's own evidence: the weld rewrite
    // sweeps, ports and recloses the carried local rows naming a retired
    // identity — the tier the restate cannot reach — the splits an edge the
    // welds collapsed owes its carriers land beside them, and the frontier
    // it hands back is the wave's own naming of the dirty set
    tf::buffer<Index> dirty;
    dirty.reserve(dirty_planes.size() + taken_planes.size());
    tf::core::append(dirty_planes, dirty);
    tf::core::append(taken_planes, dirty);
    plane_round_evidence<Index, Int> evidence;
    evidence.welds = std::move(wave_welds);
    state_plane_edge_union(
        world, _local_tables, _plane_ticket, dirty,
        identities.plane_created_classes(), identities.plane_class_ts(),
        identities.plane_class_names(), _vertex_offsets, n_flat_points(),
        _base_created, crossing_name_kind, refinement_name_kind,
        param_t(1) << tf::exact::meta<Int>::param_bits, evidence.statements,
        evidence.topology, evidence.welds);
    tf::buffer<Index> frontier;
    if (evidence.welds.size() != 0 || evidence.statements.size() != 0) {
      const auto result = advance_plane_wave(
          world, evidence, _local_tables, _plane_ticket, _group_router,
          identities, _vertex_offsets, n_flat_points(), _base_created,
          exact_point_of(identities, get_base_point, get_original_point),
          name_point_of(identities, get_base_point, get_original_point),
          _census, frontier, _near_window, _near_rows);
      if (result == plane_wave_result::unsupported) {
        fail_every_plane(world);
        return;
      }
    }
    tf::core::append(frontier, dirty);
    std::sort(dirty.begin(), dirty.end());
    dirty.erase_till_end(std::unique(dirty.begin(), dirty.end()));

    port_prior_arena(prior, extents, canon_images, dirty);
    if (dirty.size() != 0 &&
        run_rounds(world, identities, dirty, get_base_point, get_original_point,
                   apply_to_form) == plane_round_result::aborted) {
      fail_every_plane(world);
      return;
    }
    std::sort(_failed.begin(), _failed.end());
    _failed.erase_till_end(std::unique(_failed.begin(), _failed.end()));
    _census.failed_planes = _failed.size();
    // a rebuilt plane elects from the prior plane's triangles, read in this
    // value's currency
    const auto flat = extents.created.lifted(n_flat_points());
    const auto standing_triangles = [&](Index p) {
      const auto old_p = extents.planes.inverse(p);
      const auto span = old_p != Index(-1) && old_p < prior.n_planes()
                            ? prior.plane_range(old_p)
                            : std::array<Index, 2>{Index(0), Index(0)};
      return tf::make_mapped_range(
          tf::make_range(prior._store.triangles.begin() + span[0],
                         prior._store.triangles.begin() + span[1]),
          [flat](std::array<Index, 3> triangle) {
            for (auto &corner : triangle)
              corner = flat.forward(corner);
            return triangle;
          });
    };
    elect_standing_plane_steiners(
        _arena, standing_triangles, identities.plane_created_classes(),
        identities.plane_direct_points(), identities.plane_merges(),
        n_flat_points(), _base_created);
    // carried and rebuilt spans materialize as one store, and the final
    // piece space rebuilds whole from the carried local tables — the one
    // publish is the diff's publish
    publish_products(world, identities);
  }

  /// The rows a carve founded on a carrier this arrangement holds a block
  /// for join that block by reference — a world row `imaged` does not mark
  /// is the carve's own, standing where the rows of the faces it replaced
  /// stood. Each block counts its founded rows, one prefix places them, and
  /// a block that took one joins `dirty_planes`.
  template <typename World>
  auto found_plane_block_rows(const World &world,
                              const tf::buffer<char> &imaged,
                              tf::buffer<Index> &dirty_planes) -> void {
    auto &offsets = _local_tables.edges().offsets_buffer();
    auto &rows = _local_tables.edges().data_buffer();
    const auto n_blocks =
        offsets.size() == 0 ? std::size_t(0) : offsets.size() - 1;
    // a carve founds rows on the world's own carriers alone: a plane past
    // `_base_planes` is a promoted one, whose rows the local table holds
    tf::buffer<Index> plane_of_block;
    plane_of_block.allocate_and_initialize(n_blocks, Index(-1));
    tf::parallel_for_each(
        tf::make_sequence_range(Index(_plane_ticket.size())),
        [&](Index plane) {
          const auto block = _plane_ticket[std::size_t(plane)];
          if (block != Index(-1) && plane < _base_planes)
            plane_of_block[std::size_t(block)] = plane;
        },
        tf::checked);
    tf::buffer<Index> founded;
    founded.allocate(n_blocks + 1);
    founded[0] = Index(0);
    tf::parallel_for_each(
        tf::make_sequence_range(n_blocks),
        [&](std::size_t block) {
          Index count = 0;
          const auto plane = plane_of_block[block];
          if (plane != Index(-1))
            for (const auto row : world.plane_edges(plane))
              count += Index(!imaged[std::size_t(row)]);
          founded[block + 1] = count;
        },
        tf::checked);
    for (std::size_t block = 1; block <= n_blocks; ++block)
      founded[block] += founded[block - 1];
    if (founded[n_blocks] == Index(0))
      return;
    tf::buffer<Index> joined;
    joined.allocate(rows.size() + std::size_t(founded[n_blocks]));
    tf::parallel_for_each(
        tf::make_sequence_range(n_blocks),
        [&](std::size_t block) {
          auto write = std::size_t(offsets[block] + founded[block]);
          for (auto at = std::size_t(offsets[block]);
               at < std::size_t(offsets[block + 1]); ++at)
            joined[write++] = rows[at];
          if (founded[block + 1] == founded[block])
            return;
          for (const auto row : world.plane_edges(plane_of_block[block]))
            if (!imaged[std::size_t(row)])
              joined[write++] = Index(-1) - row;
        },
        tf::checked);
    for (std::size_t block = 0; block <= n_blocks; ++block)
      offsets[block] += founded[block];
    rows = std::move(joined);
    for (std::size_t block = 0; block < n_blocks; ++block)
      if (founded[block + 1] != founded[block])
        dirty_planes.push_back(plane_of_block[block]);
    std::sort(dirty_planes.begin(), dirty_planes.end());
    dirty_planes.erase_till_end(
        std::unique(dirty_planes.begin(), dirty_planes.end()));
  }

  /// The prior value's published store carried into the arena at the diff
  /// numbering; a dirty plane carries nothing — the wave owns it. Corners,
  /// planes and faces cross `extents`; an immutable slot ticket follows its
  /// group through the canon images, while a local plane's slots pass as they
  /// stand — the publish's boundary lift re-reads every ticketed plane's
  /// slots from the carried tier; in-plane references ride their spans.
  auto port_prior_arena(const plane_arrangement &prior,
                        const diff_extents<Index> &extents,
                        const tf::buffer<Index> &canon_images,
                        const tf::buffer<Index> &dirty_planes) -> void {
    const auto carried = [&](Index old_p) {
      return !std::binary_search(dirty_planes.begin(), dirty_planes.end(),
                                 extents.planes.forward(old_p));
    };
    const auto prior_planes = prior.n_planes();
    const bool record =
        _record_arrangement && prior._store.slot_parents.size() != 0;
    const bool cells = _record_cells && prior._store.triangle_cells.size() != 0;
    // a plane's triangles are its own span, so each plane counts its
    // stacked members alone and one prefix over the planes places them
    tf::buffer<Index> cop_counts;
    if (record) {
      cop_counts.allocate(std::size_t(prior_planes));
      tf::parallel_for_each(
          tf::make_sequence_range(prior_planes), [&](Index p) {
            Index count = 0;
            if (carried(p)) {
              const auto span = prior.plane_range(p);
              for (auto t = span[0]; t < span[1]; ++t)
                count += Index(prior._store.coplanar_of[std::size_t(t)] !=
                               Index(-1));
            }
            cop_counts[std::size_t(p)] = count;
          });
    }
    Index tris_n = 0;
    Index cops_n = 0;
    for (Index p = 0; p < prior_planes; ++p) {
      if (!carried(p))
        continue;
      const auto np = std::size_t(extents.planes.forward(p));
      const auto span = prior.plane_range(p);
      _arena.range[np] = {tris_n, tris_n + span[1] - span[0]};
      tris_n += span[1] - span[0];
      if (record) {
        const auto count = cop_counts[std::size_t(p)];
        _arena.cop_range[np] = {cops_n, cops_n + count};
        cops_n += count;
      }
    }
    _arena.tris.allocate(std::size_t(tris_n));
    _arena.subs.allocate(std::size_t(tris_n));
    if (record) {
      _arena.slots.allocate(std::size_t(tris_n));
      _arena.coplanar_of.allocate(std::size_t(tris_n));
      _arena.stacked.allocate(std::size_t(tris_n));
      _arena.coplanar.allocate(std::size_t(cops_n));
    }
    if (cells)
      _arena.cell_of.allocate(std::size_t(tris_n));

    const auto flat = extents.created.lifted(n_flat_points());
    const auto ported_ticket = [&](Index t) -> Index {
      if (t < Index(0) || t >= prior._immutable_canon_extent)
        return t;
      return canon_images.size() == 0 ? t : canon_images[std::size_t(t)];
    };
    tf::parallel_for_each(
        tf::make_sequence_range(prior_planes), [&](Index p) {
          if (!carried(p))
            return;
          const auto np = std::size_t(extents.planes.forward(p));
          const auto span = prior.plane_range(p);
          const auto delta = _arena.range[np][0] - span[0];
          auto cop_write = record ? _arena.cop_range[np][0] : Index(0);
          for (auto t = span[0]; t < span[1]; ++t) {
            const auto at = std::size_t(t);
            const auto to = std::size_t(t + delta);
            const auto &tri = prior._store.triangles[at];
            _arena.tris[to] = {flat.forward(tri[0]), flat.forward(tri[1]),
                               flat.forward(tri[2])};
            _arena.subs[to] = prior._store.corner_subs[at];
            if (record) {
              const auto &slots = prior._store.slot_parents[at];
              _arena.slots[to] = {ported_ticket(slots[0]),
                                  ported_ticket(slots[1]),
                                  ported_ticket(slots[2])};
              _arena.stacked[to] = prior._store.stacked[at];
              const auto link = prior._store.coplanar_of[at];
              if (link == Index(-1))
                _arena.coplanar_of[to] = Index(-1);
              else {
                auto twin = prior._store.coplanar[std::size_t(link)];
                twin.survivor += delta;
                _arena.coplanar[std::size_t(cop_write)] = twin;
                _arena.coplanar_of[to] = cop_write;
                ++cop_write;
              }
            }
            if (cells)
              _arena.cell_of[to] = prior._store.triangle_cells[at];
          }
        });

    tf::parallel_for_each(
        tf::make_sequence_range(Index(prior._store.face_range.size())),
        [&](Index f) {
          const auto span = prior._store.face_range[std::size_t(f)];
          if (span[0] == span[1])
            return;
          const auto p = prior.plane_of_triangle(span[0]);
          if (!carried(p))
            return;
          const auto nf = std::size_t(extents.faces.forward(f));
          const auto delta =
              _arena.range[std::size_t(extents.planes.forward(p))][0] -
              prior._store.plane_offsets[std::size_t(p)];
          _arena.face_range[nf] = {span[0] + delta, span[1] + delta};
        },
        tf::checked);

    for (const auto p : prior._failed) {
      const auto np = extents.planes.forward(p);
      if (!std::binary_search(dirty_planes.begin(), dirty_planes.end(), np))
        _failed.push_back(np);
    }
    for (const auto p : prior._refinement_refused) {
      const auto np = extents.planes.forward(p);
      if (!std::binary_search(dirty_planes.begin(), dirty_planes.end(), np))
        _refinement_refused.push_back(np);
    }
    _refinement_census.refused_planes = _refinement_refused.size();
  }

  template <typename World, typename GetOriginalPoint, typename ApplyToForm,
            typename GetBasePoint>
  auto run_refined(World &world, identities_t &identities,
                   const GetOriginalPoint &get_original_point,
                   const ApplyToForm &apply_to_form,
                   const GetBasePoint &get_base_point,
                   const tf::cdt_refine_config &config) -> void {
    identities.clear_plane();
    if (!initialize(world, world.n_created_points(), world.vertex_offsets()))
      return;
    refine_at_emission(config);

    auto initial = take_refinement_discovery(
        world, identities, config, get_base_point, get_original_point);

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
        fall_back_to_stock(world, identities, world.n_created_points(),
                           get_base_point, get_original_point,
                           world.vertex_offsets(), apply_to_form);
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
    tf::buffer<frame_t> face_frames;
    make_plane_weld_entrant_tables<Index, Int>(
        tf::make_range(initial.promoted_faces), world.n_faces(), no_merges,
        world.vertex_offsets(), world.face_offsets(), get_base_point,
        apply_to_form, expand_entrant_side, _promoted_descriptors, face_frames,
        raw_entrants);
    const auto n_new_planes = identities.seat_promoted_planes(
        world, _promoted_descriptors, face_frames, apply_to_form,
        get_original_point, _promoted_plane_of, _promoted_frames,
        _promoted_orientations);
    state_plane_promoted_members(_promoted_plane_of, world.n_faces(),
                                 world.n_planes() + n_new_planes,
                                 _promoted_member_offsets, _promoted_members);

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
        fall_back_to_stock(world, identities, world.n_created_points(),
                           get_base_point, get_original_point,
                           world.vertex_offsets(), apply_to_form);
        return;
      }
    }

    tf::buffer<plane_refinement_split<Index, Int>> mapped_splits;
    if (raw_entrants.edges().size() == 0) {
      mapped_splits = std::move(initial.splits);
    } else {
      tf::buffer<Index> entrant_planes;
      seat_plane_entrant_blocks(world, _promoted_plane_of, raw_entrants,
                                entrant_planes);
      tf::buffer<Index> immutable_group_ticket;
      tf::buffer<Index> entrant_group_ticket;
      if (!make_plane_refinement_overlay<Index, Int>(
              world, initial.splits, raw_entrants, entrant_planes,
              n_new_planes, entrant_origins, _local_tables, _plane_ticket,
              _group_router, immutable_group_ticket, entrant_group_ticket) ||
          !map_plane_refinement_splits<Index, Int>(
              initial.splits, initial.physical_splits,
              immutable_group_ticket, raw_entrants, entrant_group_ticket,
              _promoted_descriptors, world.n_faces(), _promoted_plane_of,
              _local_tables, apply_to_form, n_flat_points(),
              world.vertex_offsets(), mapped_splits)) {
        fall_back_to_stock(world, identities, world.n_created_points(),
                           get_base_point, get_original_point,
                           world.vertex_offsets(), apply_to_form);
        return;
      }
      if (mapped_splits.size() < initial.splits.size()) {
        fall_back_to_stock(world, identities, world.n_created_points(),
                           get_base_point, get_original_point,
                           world.vertex_offsets(), apply_to_form);
        return;
      }
      _refinement_census.ring_boundary_splits =
          mapped_splits.size() - initial.splits.size();
    }
    // the pre-step above holds the base value and its base extents; the
    // wave below reads the promoted one
    World promoted_world(world, _promoted_descriptors, _promoted_frames,
                         identities.founded_plane_names(),
                         _promoted_orientations, _promoted_plane_of,
                         _promoted_member_offsets, _promoted_members);
    if (!apply_refinement_wave(promoted_world, identities, mapped_splits,
                               std::move(initial.welds), get_base_point,
                               get_original_point)) {
      fall_back_to_stock(world, identities, world.n_created_points(),
                         get_base_point, get_original_point,
                         world.vertex_offsets(), apply_to_form);
      return;
    }

    _refinement_census.affected_planes =
        _plane_ticket.size() == 0
            ? std::size_t(0)
            : std::count_if(_plane_ticket.begin(), _plane_ticket.end(),
                            [](Index block) { return block != Index(-1); });
    _refinement_census.refused_planes = _refinement_refused.size();
    if (!run_stock(promoted_world, identities, get_base_point,
                   get_original_point, apply_to_form))
      return;
    publish_products(promoted_world, identities);
  }

  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename VertexOffsets, typename ApplyToForm>
  auto build_core(World &world, identities_t &identities, Index n_base_created,
                  const GetBasePoint &get_base_point,
                  const GetOriginalPoint &get_original_point,
                  const VertexOffsets &vertex_offsets,
                  const ApplyToForm &apply_to_form) -> bool {
    identities.clear_plane();
    if (!initialize(world, n_base_created, vertex_offsets))
      return false;
    return run_stock(world, identities, get_base_point, get_original_point,
                     apply_to_form);
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
  auto run_stock(World &world, identities_t &identities,
                 const GetBasePoint &get_base_point,
                 const GetOriginalPoint &get_original_point,
                 const ApplyToForm &apply_to_form) -> bool {
    _arena.reset(world.n_planes(), world.n_faces());

    if (run_rounds(world, identities, tf::make_sequence_range(world.n_planes()),
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
  auto run_rounds(World &world, identities_t &identities, const Planes &planes,
                  const GetBasePoint &get_base_point,
                  const GetOriginalPoint &get_original_point,
                  const ApplyToForm &apply_to_form) -> plane_round_result {
    const auto flat_point =
        flat_point_of(identities, get_base_point, get_original_point);
    const auto exact_point =
        exact_point_of(identities, get_base_point, get_original_point);
    const auto name_point =
        name_point_of(identities, get_base_point, get_original_point);
    plane_round_evidence<Index, Int> evidence;
    evidence.census.rounds = 1;
    evidence.census.rebuilt_planes = planes.size();
    triangulate_plane_round(world, planes, _local_tables, _plane_ticket,
                            _vertex_offsets, n_flat_points(),
                            n_flat_points() + _base_created,
                            crossing_name_kind, _refined, _record_arrangement,
                            _record_cells, _refine_config, _plane_steiner_sites,
                            flat_point, _arena, evidence);
    // only a value that took a discovery holds plans to count
    if (_refinement_census.planes != 0)
      _refinement_census.round1_retries = std::size_t(std::count_if(
          evidence.refused.begin(), evidence.refused.end(), [&](Index plane) {
            return plane < _base_planes &&
                   !std::binary_search(_refinement_refused.begin(),
                                       _refinement_refused.end(), plane);
          }));
    tf::buffer<Index> attempted;
    tf::buffer<Index> frontier;
    tf::buffer<Index> refusals;
    refusals.allocate_and_initialize(std::size_t(world.n_planes()), Index(0));
    const auto close_frontier = [&](const plane_round_evidence<Index, Int> &ev,
                                    plane_round_result result) {
      // Two statements of one carrier can undo each other — a mint one
      // round, its merge the next — so the wave progresses while the
      // carrier's constraint set returns to itself and no progress test
      // sees the loop. Only a budget on the carrier's own refusals does:
      // a carrier that recovers spends at most two, so eight cannot cut
      // a live one short.
      constexpr std::size_t refusal_budget = 8;
      // The spend publishes nothing — every refusing round is already in
      // the failed set — it only drops the stale product of a carrier
      // that produced before it began refusing.
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
    // The entrance's ask: which roots are still the world's and cut an
    // original side, and which originals are retired — whether a reached
    // face is one no tier has named yet. A world that holds every face
    // states no source mesh, and none of this is compiled.
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
                               _plane_ticket, _group_router, identities,
                               _vertex_offsets, n_flat_points(), _base_created,
                               exact_point, name_point, _failed, _census,
                               frontier, _near_window, _near_rows, entrance);
    };
    // The barrier: a round that saw something on a world with no group
    // space makes one and hands its carriers back — what it saw is not
    // translated, it is seen again against the real tables. The wave
    // entrance is the same sentence with "no group space" replaced by "a
    // split reaching a face the cut world never named".
    auto round = plane_round_result::retry;
    if (!close_plane_lazy_round(world, evidence, _immutable_canon_extent,
                                _census, frontier))
      round = close_round(planes, evidence);
    // What the entrance answers with is a round like any other: its
    // refusals are charged and its frontier is filtered.
    const auto take_entrance = [&](plane_round_result result) {
      if constexpr (std::is_same<ApplyToForm, tf::none_t>::value) {
        return result;
      } else {
        return result == plane_round_result::entrance
                   ? enter_wave_entrants(world, identities, evidence, entrants,
                                         answered, get_base_point,
                                         get_original_point, apply_to_form,
                                         refusals, frontier)
                   : result;
      }
    };
    round = close_frontier(evidence, take_entrance(round));
    // A wave advances by stating something a table keeps: a new
    // identity, a merge, a split — a landing states no identity, yet splits
    // its constraint — or a smaller frontier; a round stating none of the
    // four has restated what the tables already hold. One such
    // round is legitimate rebuilding. The budget of eight is measured,
    // not chosen — far above every recovery observed, far below a
    // restating cycle — so the guard cannot cut a live wave short and
    // cannot fail to catch a dead one.
    constexpr std::size_t no_progress_budget = 8;
    auto stated_created = identities.n_plane_created();
    auto stated_merges = identities.plane_merges().size();
    auto stated_splits = _census.splits;
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
                              n_flat_points() + _base_created,
                              crossing_name_kind, _refined,
                              _record_arrangement, _record_cells,
                              _refine_config, _plane_steiner_sites, flat_point,
                              _arena, evidence);
      round = close_frontier(evidence,
                             take_entrance(close_round(retried, evidence)));
      if (round != plane_round_result::retry)
        break;
      if (identities.n_plane_created() == stated_created &&
          identities.plane_merges().size() == stated_merges &&
          _census.splits == stated_splits &&
          frontier.size() >= stated_frontier &&
          answered.count == stated_answered)
        ++stalled_run;
      else
        stalled_run = 0;
      if (stalled_run == no_progress_budget) {
        // the refusing planes are already in the failed set, so the
        // guard publishes nothing and only drops the stale product a
        // stalled carrier still holds
        for (const auto plane : frontier)
          retire_stalled_plane<Index, Int>(world, plane, _arena);
        _census.stalled_planes += frontier.size();
        round = plane_round_result::done;
        break;
      }
      stated_created = identities.n_plane_created();
      stated_merges = identities.plane_merges().size();
      stated_splits = _census.splits;
      stated_frontier = frontier.size();
      stated_answered = answered.count;
    }
    return round;
  }

  /// The wave entrance: promote the source faces a round's world-tier
  /// splits reached, seat their sides in this tier, and hand the loop a
  /// frontier that carries them.
  ///
  /// The entrant gets a new span in the local tables — every edge that is
  /// its alone. The one row that cannot go there is the shared edge: its
  /// canonical group already exists and is the cross-face join — a second
  /// group for one wall is the twin-wall defect — so that row joins the
  /// existing group as one more instance.
  ///
  /// A face enters whole, so one holding a side this tier cannot seat is
  /// declined. Both answers are recorded: either way the face states its
  /// sides somewhere the world's own span cannot see, and the discovery
  /// must never offer it again — that record is what ends the entrance.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint,
            typename ApplyToForm>
  auto enter_wave_entrants(World &world, identities_t &identities,
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
    // At this moment only the world's splits are applied to the group
    // the shared side names, so the entrant's side lifts through exactly
    // the pieces its neighbour's instance holds.
    const auto state_entrant_tables = [&](const auto &faces) {
      make_plane_weld_entrant_tables<Index, Int>(
          faces, world.n_faces(), identities.plane_merges(),
          world.vertex_offsets(), world.face_offsets(), get_base_point,
          apply_to_form, expand_entrant_side, descriptors, frames,
          raw_entrants);
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
    // The round is handed back either way, so its frontier is the one
    // every discarded round states: the carriers that saw something.
    // With every face now answered for, the entrance cannot fire twice.
    state_plane_round_frontier(evidence, frontier);
    if (seated.size() == 0)
      return plane_round_result::retry;
    // the tables are positional in the faces they were stated for, so a
    // decline restates them for the faces that remain
    if (seated.size() != entrants.size())
      state_entrant_tables(tf::make_range(seated));

    const auto n_world_planes = world.n_planes();
    tf::buffer<Index> plane_of;
    tf::buffer<frame_t> new_frames;
    tf::buffer<tf::exact::canonical_plane<Int>> new_names;
    const auto n_new = seat_plane_promoted_faces<Index, Int>(
        world, descriptors, frames, apply_to_form, get_original_point,
        plane_of, new_frames, new_names, orientations);
    const auto plane_of_face = [&world](Index face) {
      return world.plane_of_face(face);
    };
    tf::buffer<Index> joined;
    if (!seat_plane_wave_entrants(world, raw_entrants, world_group_of,
                                  route_of, plane_of, n_world_planes, n_new,
                                  plane_of_face, _local_tables, _plane_ticket,
                                  _group_router, joined))
      return plane_round_result::aborted;
    tf::core::append(descriptors, _promoted_descriptors);
    tf::core::append(new_frames, _promoted_frames);
    identities.seat_founded_planes(new_names);
    tf::core::append(orientations, _promoted_orientations);
    tf::core::append(plane_of, _promoted_plane_of);
    state_plane_promoted_members(_promoted_plane_of, _base_faces,
                                 n_world_planes + n_new,
                                 _promoted_member_offsets, _promoted_members);
    // The promotion constructor reads its base
    // extents off the value it is handed, never off that value's current
    // ones, so appending to the promoted tables and rebuilding from the
    // value in hand composes. Layering one promoted value on another would
    // restate an extent that is already frozen.
    world =
        World(world, _promoted_descriptors, _promoted_frames,
              identities.founded_plane_names(), _promoted_orientations,
              _promoted_plane_of, _promoted_member_offsets, _promoted_members);
    _arena.grow_planes(world.n_planes());
    _arena.grow_faces(world.n_faces());
    refusals.reallocate_and_initialize(std::size_t(world.n_planes()), Index(0));
    _census.entrant_planes += seated.size();

    // the entrants join the frontier DIRTY — they hold no product — beside
    // the carriers the discarded round saw something on, and so does every
    // standing plane that gained a member
    for (Index r = 0; r < n_new; ++r)
      frontier.push_back(n_world_planes + r);
    tf::core::append(joined, frontier);
    std::sort(frontier.begin(), frontier.end());
    frontier.erase_till_end(std::unique(frontier.begin(), frontier.end()));
    return plane_round_result::retry;
  }

  template <typename World>
  auto publish_products(const World &world, identities_t &identities) -> void {
    materialize_plane_products(world, _arena, _record_arrangement,
                               _record_cells, _base_created, n_flat_points(),
                               _store, identities, _refinement_census);
    // A carrier a weld did not re-triangulate holds the corners it was built
    // with; the closed table is what makes them the identities this
    // arrangement publishes. A weld can collapse a retained triangle whole —
    // an unconstrained diagonal between merged identities — so the store
    // compacts before anything downstream reads it.
    weld_plane_triangles(identities.plane_merges(), n_flat_points(), _store);
    _census.triangles = _store.triangles.size();
    // the piece space is one request's product: unasked, no slot carries
    // a ticket and no definition span is owed one
    if (_record_arrangement && _plane_ticket.size() != 0 &&
        !publish_plane_final_pieces(
            world, _local_tables, _group_router, _plane_ticket, _vertex_offsets,
            n_flat_points(), _immutable_canon_extent, _store))
      fail_every_plane(world);
  }

  /// THE THIRD STATEMENT PRODUCER'S QUESTION, asked of the arrangement as it
  /// stands: every plane is read from the one tier its ticket names, through
  /// this arrangement's own exact readers.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint>
  auto discover_refinement(const World &world, const identities_t &identities,
                           const tf::cdt_refine_config &config,
                           const GetBasePoint &get_base_point,
                           const GetOriginalPoint &get_original_point) const
      -> plane_refinement_plan<Index, Int> {
    return make_plane_refinement_plan(
        world, _local_tables, _plane_ticket, _vertex_offsets, n_flat_points(),
        config, flat_point_of(identities, get_base_point, get_original_point),
        exact_point_of(identities, get_base_point, get_original_point));
  }

  /// The discovery, TAKEN: its census becomes this build's, its sites become
  /// the arrangement's, and its refusals join the published set. Discovery
  /// states the sites; emission is where they become identities, and only the
  /// ones its own triangulation keeps — so the census counts none yet.
  template <typename World, typename GetBasePoint, typename GetOriginalPoint>
  auto take_refinement_discovery(const World &world,
                                 const identities_t &identities,
                                 const tf::cdt_refine_config &config,
                                 const GetBasePoint &get_base_point,
                                 const GetOriginalPoint &get_original_point)
      -> plane_refinement_plan<Index, Int> {
    auto initial = discover_refinement(world, identities, config,
                                       get_base_point, get_original_point);
    _refinement_census = initial.census;
    _refinement_census.steiners = 0;
    _plane_steiner_sites = std::move(initial.steiners);
    append_refinement_refused(initial.refused_planes);
    return initial;
  }

  /// THE THREE READERS every step of this build resolves a coordinate
  /// through: one flat identity, one (tag, id) pair, and one class name at its
  /// dyadic parameter. The steps are free operations, so the arrangement binds
  /// the tables its waves state into them here and stays the only authority
  /// on what an identity of this tier means.
  template <typename GetBasePoint, typename GetOriginalPoint>
  auto flat_point_of(const identities_t &identities,
                     const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point) const {
    return
        [this, &identities, &get_base_point, &get_original_point](Index flat) {
          return point_of_flat(identities, flat, get_base_point,
                               get_original_point);
        };
  }

  template <typename GetBasePoint, typename GetOriginalPoint>
  auto exact_point_of(const identities_t &identities,
                      const GetBasePoint &get_base_point,
                      const GetOriginalPoint &get_original_point) const {
    return [this, &identities, &get_base_point,
            &get_original_point](std::int16_t tag, Index id) {
      return final_point(identities, tag, id, get_base_point,
                         get_original_point);
    };
  }

  template <typename GetBasePoint, typename GetOriginalPoint>
  auto name_point_of(const identities_t &identities,
                     const GetBasePoint &get_base_point,
                     const GetOriginalPoint &get_original_point) const {
    return [this, &identities, &get_base_point,
            &get_original_point](param_t parameter, const name_t &name) {
      return form_position(identities, parameter, name, get_base_point,
                           get_original_point);
    };
  }

  /// A created point is (name, parameter) and nothing else: a landing reads
  /// the identity its name carries, and a crossing blends its birth edge's
  /// two endpoints — feature[1] and feature[2], the same identities the
  /// births matched the edge by — at the dyadic parameter. No group id
  /// enters persistence; identities are the only stable currency.
  template <typename Read>
  auto name_position(param_t parameter, const name_t &name,
                     const Read &read) const -> pt3_t {
    const auto flat_point = [&](Index flat) -> pt3_t {
      if (flat >= n_flat_points())
        return read(std::int16_t(-1), flat - n_flat_points());
      const auto tag = tf::exact::tag_of_flat_vertex(_vertex_offsets, flat);
      return read(std::int16_t(tag), flat - _vertex_offsets[std::size_t(tag)]);
    };
    if (name.feature[0] == Index(0))
      return read(std::int16_t(-1), name.feature[1]);
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
  auto form_position(const identities_t &identities, param_t parameter,
                     const name_t &name, const GetPoint &get_point,
                     const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return name_position(parameter, name, [&](std::int16_t tag, Index id) {
      return final_point(identities, tag, id, nullptr, get_point,
                         get_mesh_point);
    });
  }

  template <typename GetPoint, typename GetMeshPoint>
  auto final_point(const identities_t &identities, std::int16_t tag, Index id,
                   const GetPoint &get_point,
                   const GetMeshPoint &get_mesh_point) const -> pt3_t {
    return final_point(identities, tag, id, nullptr, get_point, get_mesh_point);
  }

  /// Names persist across waves, so every endpoint speaks the plane layer by
  /// the one position law (@ref tf::intersect::graph::final_identity_point);
  /// the prefix this tier stands on answers `get_point`.
  template <typename GetPoint, typename GetMeshPoint>
  auto final_point(
      const identities_t &identities, std::int16_t tag, Index id,
      const tf::intersect::graph::evaluated_identity<Index> *mints,
      const GetPoint &get_point, const GetMeshPoint &get_mesh_point) const
      -> pt3_t {
    // a layer retiring nothing answers an original or a prefix key with
    // itself, so the read is the prefix's and no evaluation is entered
    const bool retires = identities.plane_merges().size() != 0;
    if (!retires && (tag >= 0 || id < _base_created))
      return get_point(tag, id);
    return tf::intersect::graph::final_identity_point<pt3_t>(
        retires,
        [&identities](Index source, Index key) {
          return identities.plane_target(source, key);
        },
        _vertex_offsets, tag, id, mints,
        [&](std::int16_t at_tag, Index at,
            const tf::intersect::graph::evaluated_identity<Index> *self)
            -> pt3_t {
          if (at_tag >= 0 || at < _base_created)
            return get_point(at_tag, at);
          return statement_point(
              identities, at, [&](std::int16_t read_tag, Index point) {
                return final_point(identities, read_tag, point, self,
                                   get_point, get_mesh_point);
              });
        });
  }

  /// THE one way a coordinate is read here: an original vertex answers from
  /// its mesh, a point of the arrangement that entered from its table, and an
  /// identity a wave of this tier created from the class that named it.
  template <typename GetPoint, typename GetMeshPoint>
  auto point_of_flat(const identities_t &identities, Index flat,
                     const GetPoint &get_point,
                     const GetMeshPoint &get_mesh_point) const -> pt3_t {
    if (flat >= n_flat_points())
      return final_point(identities, std::int16_t(-1), flat - n_flat_points(),
                         get_point, get_mesh_point);
    const auto tag = tf::exact::tag_of_flat_vertex(_vertex_offsets, flat);
    return final_point(identities, std::int16_t(tag),
                       flat - _vertex_offsets[std::size_t(tag)], get_point,
                       get_mesh_point);
  }

  /// The dirty tier as if no wave had happened: no local row, no group, no
  /// ticket.
  auto clear_definition_tier() -> void {
    _local_tables.clear();
    _group_router.clear();
    _plane_ticket.clear();
  }

  template <typename World> auto fail_every_plane(const World &world) -> void {
    _failed.allocate(std::size_t(world.n_planes()));
    for (Index plane = 0; plane < world.n_planes(); ++plane)
      _failed[std::size_t(plane)] = plane;
    _census.failed_planes = _failed.size();
    // a terminal failure owes nobody its scratch
    clear_definition_tier();
    materialize_empty(world);
  }

  template <typename World> auto materialize_empty(const World &world) -> void {
    _store.reset(world.n_planes(), world.n_faces());
    _census.triangles = 0;
  }

  plane_arrangement_store<Index> _store;
  tf::buffer<Index> _failed;

  plane_arrangement_arena<Index, Int> _arena;
  tf::intersect::graph::plane_tables<Index, Int> _local_tables;
  tf::buffer<Index> _group_router;
  tf::buffer<Index> _plane_ticket;
  /// Discovery's interior sites, by plane. Positions only — emission is where
  /// they become identities, if the plane's own triangulation keeps them.
  tf::offset_block_buffer<Index, pt3_t> _plane_steiner_sites;
  tf::buffer<Index> _refinement_refused;
  tf::buffer<descriptor_t> _promoted_descriptors;
  tf::buffer<frame_t> _promoted_frames;
  tf::buffer<std::int8_t> _promoted_orientations;
  tf::buffer<Index> _promoted_plane_of;
  tf::buffer<Index> _promoted_member_offsets;
  tf::buffer<Index> _promoted_members;
  tf::range<const Index *, tf::dynamic_size> _vertex_offsets;
  tf::cdt_refine_config _refine_config;
  Index _base_created = 0;
  Index _base_faces = 0;
  Index _base_planes = 0;
  Index _immutable_canon_extent = 0;
  bool _refined = false;
  bool _record_arrangement = false;
  bool _record_cells = false;
  Int _near_window = Int(0);
  tf::intersect::graph::near_identity_rows<Index, pt3_t> _near_rows;
  plane_arrangement_census _census;
  plane_refinement_census _refinement_census;
};

} // namespace tf::arrangement
