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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/close_plane_classes.hpp"
#include "../../intersect/graph/detect_plane_crossings.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/near_identity_rows.hpp"
#include "../../intersect/graph/plane_census.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"
#include "../../intersect/graph/seat_entrant_planes.hpp"
#include "../planes/plane_recovery_name.hpp"
#include "../planes/seat_plane_promoted_faces.hpp"
#include "./assert_welds_are_lawful.hpp"
#include "./carry_plane_identities.hpp"
#include "./extent_shift.hpp"
#include "./local_identity_tables.hpp"
#include "./mint_plane_classes.hpp"
#include "./plane_identity_tables.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::arrangement {

/// Where a run of plane-layer births stands: its first direct point and its
/// first created identity.
struct plane_point_births {
  std::size_t direct;
  std::size_t created;
};

/// THE IDENTITY SUBSTRATE — the one authority over an arrangement's
/// identities, of its points and of its planes. AN IDENTITY IS MINTED,
/// WELDED OR NAMED ONLY THROUGH THIS: the tables are its own and
/// private, every table reads through it as it stands, and every statement
/// enters through one of its verbs, which states the verb's law at the call.
/// The local arrangement states the layer `L` and the plane arrangement the
/// layer `P`; neither holds a table.
///
/// The substrate is a value: it moves with the owner that holds it, and the
/// plane layer a prior arrangement stated is the one its diff carries forward
/// (@ref carry_plane).
///
/// ONE RELATION, TWO CLOSED LAYERS. `L` is the local tier's closed rewrite
/// table and `P` the plane tier's; what a key speaks is the composed answer
/// `P∘L`, the local layer read first and the plane layer on what it left
/// (@ref composed_target). A reader of a standing arrangement names a key
/// there, and an answer past the created identities its currency holds is
/// foreign to it: the carve, the one reader that asks across a currency,
/// refuses such an answer (`reseat_refusal::name_foreign`). A layer alone
/// answers only its own tier's passes.
///
/// THE VERBS: a mint states a birth, a weld a relabel and a seat a plane's
/// name. A death is stated nowhere: at either tier, an identity no published
/// row or triangle names is dead by definition, since a dropped row cannot
/// be read through, and it stays in the tables unreferenced. Every created
/// identity is therefore live, retired or vanished (@ref
/// assert_created_identities_are_accounted). A mint at a dead identity's
/// lattice position elects it back to life — one position, one identity.
///
/// ONE PLANE SPACE, TWO TIERS. A plane's identity is its name (@ref
/// tf::intersect::graph::name_plane_carriers); the local tier names its own
/// planes and the plane tier the ones its promotion founded past them, so
/// every reader seating a face sees every plane either tier holds (@ref
/// plane_name). A name is geometry, so the renumbering between plane values
/// never reaches one: the suffix is positional past the local tier's extent,
/// and slides with it.
///
/// THE DESCENT RULE: a row's target speaks the currency of the layer that
/// states it, or a lower one's. `L`'s writers — the gate, the census and the
/// carve — target originals and the local tier's created identities only, so
/// the renumbering between plane values (@ref extent_shift) never reaches
/// `L`; `P`'s rows may target anything. A weld's target is live in what it
/// is stated into, and one key holds one target; a death that later takes
/// the target takes its retired sources with it (@ref
/// account_created_identities).
///
/// INVARIANT I: `P` is closed over `L` for every key a plane product can
/// hold, so a plane reader reads `P` alone and is answered `P∘L`. A build
/// keeps it by construction — the world reaches the plane tier already
/// rewritten by `L` — and a diff keeps it by carrying `P` into the next
/// value's currency and then composing every row a statement moved in `L`
/// into `P`'s evidence (@ref compose_local_moves): forward, then append, which
/// order and closure both survive because the shift is monotone. Only a key
/// that can be retired in both layers reads @ref composed_target.
template <typename Index, typename Int> class identity_substrate {
public:
  using param_t = typename tf::exact::meta<Int>::param_type;
  using pt3_t = tf::exact::pt3<Int>;
  using row_t = std::array<Index, 3>;
  using plane_name_t = tf::exact::canonical_plane<Int>;
  using name_statement_t = tf::intersect::graph::plane_name_statement<Index>;
  using class_name_t = plane_recovery_name<Index, param_t>;

  // ---- the composed read ---------------------------------------------

  /// `P∘L` on `(source, id)`: the identity it speaks now, `-1` for a key
  /// neither layer retires.
  auto composed_target(Index source, Index id) const -> Index {
    const auto local_answer = local_target(source, id);
    const auto plane_answer = local_answer == Index(-1)
                                  ? plane_target(source, id)
                                  : plane_target(Index(1), local_answer);
    return plane_answer == Index(-1) ? local_answer : plane_answer;
  }
  /// The local layer's answer alone, `-1` for a key it does not retire.
  auto local_target(Index source, Index id) const -> Index {
    return tf::intersect::graph::plane_merge_target(
        tf::make_range(_local.merges), source, id);
  }
  /// The plane layer's answer alone, `-1` for a key it does not retire.
  auto plane_target(Index source, Index id) const -> Index {
    return tf::intersect::graph::plane_merge_target(
        tf::make_range(_plane.merges), source, id);
  }

  /// The name of plane `id`: the local tier's below its extent, the
  /// promotion's past it.
  auto plane_name(Index id) const -> const plane_name_t & {
    const auto n_local = Index(_local.plane_names.size());
    return id < n_local ? _local.plane_names[std::size_t(id)]
                        : _plane.plane_names[std::size_t(id - n_local)];
  }
  /// Every plane's name, in plane order.
  auto plane_names() const {
    return tf::make_mapped_range(
        tf::make_sequence_range(
            Index(_local.plane_names.size() + _plane.plane_names.size())),
        [this](Index at) -> const plane_name_t & { return plane_name(at); });
  }

  // ---- the local layer, as it stands ----------------------------------

  auto local_merges() const -> const tf::buffer<row_t> & {
    return _local.merges;
  }
  auto local_plane_names() const -> const tf::buffer<plane_name_t> & {
    return _local.plane_names;
  }
  auto local_statements() const -> const tf::buffer<name_statement_t> & {
    return _local.statements;
  }
  auto local_class_offsets() const -> const tf::buffer<Index> & {
    return _local.class_offsets;
  }
  auto local_class_ids() const -> const tf::buffer<Index> & {
    return _local.class_id;
  }
  auto local_class_roots() const -> const tf::buffer<Index> & {
    return _local.class_root;
  }
  auto local_class_ts() const -> const tf::buffer<param_t> & {
    return _local.class_t;
  }
  auto local_created_classes() const -> const tf::buffer<Index> & {
    return _local.created_class;
  }
  auto local_direct_points() const -> const tf::buffer<pt3_t> & {
    return _local.direct_points;
  }
  auto n_local_classes() const -> Index { return _local.n_classes; }
  auto n_local_created() const -> Index { return _local.n_created(); }

  // ---- the plane layer, as it stands ----------------------------------

  auto plane_merges() const -> const tf::buffer<row_t> & {
    return _plane.merges;
  }
  /// The names of the planes the promotion founded, past the local tier's.
  auto founded_plane_names() const -> const tf::buffer<plane_name_t> & {
    return _plane.plane_names;
  }
  auto plane_created_classes() const -> const tf::buffer<Index> & {
    return _plane.created_class;
  }
  auto plane_direct_points() const -> const tf::buffer<pt3_t> & {
    return _plane.direct_points;
  }
  auto plane_class_ts() const -> const tf::buffer<param_t> & {
    return _plane.class_t;
  }
  auto plane_class_names() const -> const tf::buffer<class_name_t> & {
    return _plane.class_name;
  }
  auto n_plane_created() const -> Index { return _plane.n_created(); }

  // ---- the local layer's verbs ----------------------------------------

  /// A build of the local tier restates its whole layer.
  auto clear_local() -> void { _local.clear(); }

  /// The cut planes named by the graph's build, which states the layer's
  /// plane names whole (@ref tf::intersect::graph::plane_graph::build).
  template <typename Ibp, typename ApplyToFace, typename ApplyToForm,
            typename GetPoint, typename SideNonManifold>
  auto name_local_planes(tf::intersect::graph::plane_graph<Index, Int> &graph,
                         const Ibp &ibp, const ApplyToFace &apply_to_face,
                         const ApplyToForm &apply_to_form,
                         const GetPoint &get_point,
                         const SideNonManifold &side_non_manifold) -> void {
    graph.build(ibp, apply_to_face, apply_to_form, get_point, side_non_manifold,
                _local.plane_names);
  }
  /// The names of the planes the local tier founds, in plane order past the
  /// ones it holds.
  auto seat_local_planes(const tf::buffer<plane_name_t> &names) -> void {
    tf::core::append(names, _local.plane_names);
  }
  /// The local tier's entrants seated by name beside its cut planes (@ref
  /// tf::intersect::graph::seat_entrant_planes), the planes they found
  /// named past them.
  template <typename ApplyToForm, typename GetPoint>
  auto seat_local_entrant_planes(
      const tf::intersect::graph::plane_graph<Index, Int> &graph,
      const tf::buffer<tf::intersect::graph::face_descriptor<Index>>
          &entrant_descriptors,
      const ApplyToForm &apply_to_form, const GetPoint &get_point,
      tf::buffer<Index> &entrant_plane_of,
      tf::buffer<tf::exact::plane_frame<Int>> &entrant_planes,
      tf::buffer<std::int8_t> &entrant_orientations,
      tf::offset_block_buffer<Index, Index> &seated) -> Index {
    return tf::intersect::graph::seat_entrant_planes(
        graph, entrant_descriptors, apply_to_form, get_point,
        _local.plane_names, entrant_plane_of, entrant_planes,
        entrant_orientations, seated);
  }

  /// The name statements a build's crossing detection states (@ref
  /// tf::intersect::graph::detect_plane_crossings), once per build.
  template <typename RealType, typename GetPoint, typename GetMeshPoint,
            typename ApplyToFace, typename VertexOffsets>
  auto state_local_names(
      const tf::intersect::graph::plane_graph<Index, Int> &graph,
      const tf::polygon_intersections<Index, RealType, Int> &ibp,
      const GetPoint &get_point, const GetMeshPoint &get_mesh_point,
      const ApplyToFace &apply_to_face, const VertexOffsets &vertex_offsets,
      bool resolve_self_contours, tf::intersect::graph::plane_census &census)
      -> void {
    if constexpr (tf::core::verification)
      tf::core::verify(_local.statements.size() == 0,
                       "a build states its names once");
    tf::intersect::graph::detect_plane_crossings(
        graph, ibp, get_point, get_mesh_point, apply_to_face, vertex_offsets,
        resolve_self_contours, _local.statements, census);
  }
  /// The statements closed into classes (@ref
  /// tf::intersect::graph::close_plane_classes).
  template <typename GetPoint>
  auto close_local_classes(
      const tf::intersect::graph::plane_graph<Index, Int> &graph,
      const GetPoint &get_point, tf::intersect::graph::plane_census &census)
      -> void {
    _local.n_classes = tf::intersect::graph::close_plane_classes<Index, Int>(
        graph, get_point, _local.statements, _local.class_offsets,
        _local.class_id, _local.class_root, _local.class_t, census);
  }
  /// THE GATE: every identity the layer names quantized in its own frame,
  /// one identity per position — a mint for a class that names none, a weld
  /// for every other (@ref tf::intersect::graph::collapse_plane_identities).
  /// A class the gate minted to NAME a position states no root and never
  /// enters the split walk. `n_points` are the intersections' own points.
  /// Returns the positions that absorbed more than one identity.
  template <typename Graph, typename GetPoint, typename ClassPosition,
            typename VertexOffsets, typename Landings>
  auto collapse_local_identities(
      const Graph &graph, const GetPoint &get_point,
      const ClassPosition &class_position, const VertexOffsets &vertex_offsets,
      const Landings &landings, bool placed_originals, Index n_points,
      Int near_window,
      tf::intersect::graph::near_identity_rows<Index, pt3_t> &near_rows)
      -> std::size_t {
    const auto absorbed = tf::intersect::graph::collapse_plane_identities(
        graph, get_point, class_position, vertex_offsets, landings,
        placed_originals, _local.statements, _local.class_offsets,
        _local.n_classes, n_points, _local.class_id, _local.created_class,
        _local.merges, near_window, near_rows);
    const auto n_classes_after = _local.class_offsets.size() == 0
                                     ? std::size_t(0)
                                     : _local.class_offsets.size() - 1;
    for (auto cls = std::size_t(_local.n_classes); cls < n_classes_after;
         ++cls) {
      _local.class_root.push_back(Index(-1));
      _local.class_t.push_back(param_t(0));
    }
    if constexpr (tf::core::verification)
      assert_welds_are_lawful(
          _local.merges, n_points + _local.n_created(),
          [this](Index id) { return local_target(Index(1), id) == Index(-1); });
    return absorbed;
  }
  /// Each point a birth of the local layer, a direct created identity in
  /// order; returns the created index of the first.
  auto mint_local_points(const tf::buffer<pt3_t> &points) -> Index {
    const auto first = _local.n_created();
    for (const auto &point : points) {
      _local.created_class.push_back(-Index(_local.direct_points.size()) -
                                     Index(1));
      _local.direct_points.push_back(point);
    }
    return first;
  }
  /// `welds` closed into the local layer (@ref
  /// tf::intersect::graph::close_plane_merges); `moved` receives the closed
  /// rows the layer did not hold. A weld speaks the layer's own currency: an
  /// original below `n_flat`, a created identity below `n_created`.
  auto weld_local(const tf::buffer<row_t> &welds, Index n_flat, Index n_created,
                  tf::buffer<row_t> &moved) -> void {
    if constexpr (tf::core::verification)
      for (const auto &weld : welds)
        tf::core::verify(
            (weld[0] == Index(0) ? weld[1] < n_flat : weld[1] < n_created) &&
                weld[2] < n_created,
            "a local weld speaks the local layer's identities");
    tf::intersect::graph::close_plane_merges<Index>(_local.merges, welds,
                                                    moved);
    if constexpr (tf::core::verification)
      assert_welds_are_lawful(_local.merges, n_created, [this](Index id) {
        return local_target(Index(1), id) == Index(-1);
      });
  }

  // ---- the plane layer's verbs ----------------------------------------

  /// A build of the plane tier restates its whole layer.
  auto clear_plane() -> void { _plane.clear(); }

  /// A wave's classes minted (@ref mint_plane_classes): `created_classes`
  /// names, per created identity it mints in order, the class it names among
  /// `class_ts`, and `class_name(cls)` that class's name.
  template <typename ClassName>
  auto mint_plane_classes(const tf::buffer<param_t> &class_ts,
                          const ClassName &class_name,
                          const tf::buffer<Index> &created_classes) -> void {
    tf::arrangement::mint_plane_classes(class_ts, class_name, created_classes,
                                        _plane.class_t, _plane.class_name,
                                        _plane.created_class);
  }
  /// The births of `n` direct identities of the plane layer: their slots,
  /// each stated by @ref state_plane_point. The first is created index
  /// `births.created`.
  auto mint_plane_points(std::size_t n) -> plane_point_births {
    const plane_point_births births{_plane.direct_points.size(),
                                    _plane.created_class.size()};
    _plane.direct_points.reallocate(births.direct + n);
    _plane.created_class.reallocate(births.created + n);
    return births;
  }
  /// Birth `k` of `births` stands at `point`; distinct births are distinct
  /// slots.
  auto state_plane_point(const plane_point_births &births, std::size_t k,
                         const pt3_t &point) -> void {
    _plane.direct_points[births.direct + k] = point;
    _plane.created_class[births.created + k] =
        -Index(births.direct + k) - Index(1);
  }
  /// A wave's closed table — the standing rows and its own, closed once —
  /// becomes the plane layer.
  auto weld_plane(tf::buffer<row_t> &&closed) -> void {
    _plane.merges = std::move(closed);
    if constexpr (tf::core::verification)
      assert_welds_are_lawful(_plane.merges, Index(-1), [this](Index id) {
        return composed_target(Index(1), id) == Index(-1);
      });
  }
  /// The refinement's promoted faces seated by name (@ref
  /// seat_plane_promoted_faces), the planes they found named as the layer's
  /// founded planes, whole.
  template <typename World, typename Descriptors, typename FaceFrames,
            typename ApplyToForm, typename GetPoint>
  auto seat_promoted_planes(const World &world, const Descriptors &descriptors,
                            const FaceFrames &face_frames,
                            const ApplyToForm &apply_to_form,
                            const GetPoint &get_point,
                            tf::buffer<Index> &plane_of,
                            tf::buffer<tf::exact::plane_frame<Int>> &frames,
                            tf::buffer<std::int8_t> &orientations) -> Index {
    return seat_plane_promoted_faces<Index, Int>(
        world, descriptors, face_frames, apply_to_form, get_point, plane_of,
        frames, _plane.plane_names, orientations);
  }
  /// The names of the planes the promotion founds, past the ones it holds.
  auto seat_founded_planes(const tf::buffer<plane_name_t> &names) -> void {
    tf::core::append(names, _plane.plane_names);
  }
  /// The plane layer a prior value stated, carried across `created` into
  /// this value's currency (@ref carry_plane_identities).
  auto carry_plane(const extent_shift<Index> &created, Index n_flat,
                   Index crossing_kind, Index refinement_kind) -> void {
    auto prior = std::move(_plane);
    _plane.clear();
    carry_plane_identities(std::move(prior), created, n_flat, crossing_kind,
                           refinement_kind, _plane);
  }

private:
  local_identity_tables<Index, Int> _local;
  plane_identity_tables<Index, Int> _plane;
};

} // namespace tf::arrangement
