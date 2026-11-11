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
#include "../core/algorithm/parallel_contains.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/config/verify.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/range.hpp"
#include "../core/reallocate.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/input_lattice.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact/tag_of_flat_vertex.hpp"
#include "../exact/vertex_converter.hpp"
#include "../intersect/graph/local_arrangement.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../intersect/polygon_intersections.hpp"
#include "../topology/cdt_refine_config.hpp"
#include "../topology/triangulation_type.hpp"
#include "./arrangement_config.hpp"
#include "./identity/assert_created_identities_are_accounted.hpp"
#include "./identity/assert_plane_layer_is_closed.hpp"
#include "./identity/composed_identity_point.hpp"
#include "./identity/identity_substrate.hpp"
#include "./identity/near_identity_request.hpp"
#include "./planes/assert_promotion_is_complete.hpp"
#include "./planes/make_plane_arrangement_cells.hpp"
#include "./planes/make_plane_piece_fences.hpp"
#include "./planes/make_plane_piece_incidence.hpp"
#include "./planes/plane_arrangement.hpp"
#include "./planes/plane_arrangement_face.hpp"
#include "./planes/plane_coplanar_triple.hpp"
#include "./planes/plane_flat_weld.hpp"
#include "./planes/plane_triangulation_types.hpp"
#include "./reseat/carve_census.hpp"
#include "./reseat/name_piece_corners.hpp"
#include "./reseat/rank_pieces_by_tag.hpp"
#include "./reseat/reseat_pieces.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf {

/// @ingroup arrangement
/// @brief The arrangement of a set of forms over the plane pipeline —
///        everything below classification.
///
/// The pipeline runs
/// `intersections -> plane graph -> local arrangement -> plane
///  arrangement` and exposes the stream currency the consumers read:
/// one slot per face of every form, contiguous per tag, an uncut face
/// keeping its descriptor row with an EMPTY triangle span; the exposed
/// triangles in tag-major face order with their corners as stream
/// vertices; and the unified created-points table.
///
/// THE CURRENCY IS THE CALLER'S. A carve is internal to the build: the
/// slots are the caller's faces, a face the carve cut spans every piece of
/// it, and a point it added is a created identity like any other.
///
/// The storage policy holds the operands exactly as the other
/// arrangement classes' policies do, and everything per-operand is
/// reached through `apply_to_form(tag, f)`.
template <typename Policy, typename Int = tf::none_t> class arrangement_graph {
public:
  using policy_type = Policy;
  using index_type = typename Policy::index_type;
  using input_real_type = typename Policy::input_real_type;
  using resolved_int_type = tf::exact::resolve_int_type<Int, input_real_type>;
  // the pipeline runs at the input's own width: the f32 lattice is
  // int32 and its gated pipeline is the float one
  using pipeline_real_type = input_real_type;
  using world_type = tf::intersect::graph::local_arrangement<
      index_type, pipeline_real_type, resolved_int_type>;
  using arrangement_type =
      tf::arrangement::plane_arrangement<index_type, resolved_int_type>;
  using identities_type =
      tf::arrangement::identity_substrate<index_type, resolved_int_type>;
  using lattice_type =
      tf::exact::input_lattice<index_type, pipeline_real_type, resolved_int_type>;
  using vertex_t = tf::intersect::graph::vertex<index_type>;
  using exposed_descriptor =
      tf::arrangement::exposed_descriptor<index_type>;
  using coplanar_triple = tf::arrangement::plane_coplanar_triple<index_type>;
  static constexpr std::size_t face_static_size = Policy::face_static_size;
  static constexpr std::size_t static_n_tags = Policy::static_n_tags;

  /// The stream view the emission and map builders read, answered by the
  /// adapter tables this graph materialized once at build.
  struct global_view {
    const arrangement_graph *owner;
    auto exposed_tris() const { return tf::make_range(owner->_exposed_tris); }
    auto exposed_descriptors() const {
      return tf::make_range(owner->_exposed_descriptors);
    }
    auto face_offset_base() const {
      return tf::make_range(owner->_face_offset_base);
    }
    /// Per tag: where its face slots begin in the exposed descriptors. The
    /// world states this prefix — it is the face space the whole pipeline
    /// was built over.
    auto face_slot_base() const { return owner->_world.face_offsets(); }
    auto face_offsets() const { return tf::make_range(owner->_face_offsets); }
    auto tag_offsets() const { return tf::make_range(owner->_tag_offsets); }
    auto exposed_cons_bits() const {
      const auto *graph = owner;
      return tf::make_mapped_range(
          tf::make_sequence_range(owner->_row_of.size()),
          [graph](std::size_t exposed) {
            const auto &parents =
                graph->_arr.slot_parents()[std::size_t(
                    graph->_row_of[exposed])];
            std::uint8_t bits = 0;
            for (int c = 0; c < 3; ++c)
              if (parents[std::size_t(c)] != index_type(-1))
                bits = std::uint8_t(bits | (std::uint8_t(1) << c));
            return bits;
          });
    }
    /// Per exposed triangle corner: the parent piece ticket of the edge
    /// it opens, -1 when the edge lies on no piece.
    auto exposed_parent_of() const {
      const auto *graph = owner;
      return tf::make_mapped_range(
          tf::make_sequence_range(owner->_row_of.size() * 3),
          [graph](std::size_t slot) {
            return graph->_arr.slot_parents()[std::size_t(
                graph->_row_of[slot / 3])][slot % 3];
          });
    }
  };

  /// The lattice view and its `quantum` (@ref
  /// tf::exact::pt_converter_real::quantum) are built above this
  /// graph, over the same operands, and handed in: where every original
  /// vertex stands and what the input can tell apart, of which this build has
  /// no second opinion. `request` asks every tier that names an identity for
  /// its near table at `request.quanta` quanta.
  arrangement_graph(Policy policy, lattice_type lattice,
                    tf::arrangement_config config, resolved_int_type quantum,
                    tf::arrangement::near_identity_request request)
      : _policy(std::move(policy)), _lattice(std::move(lattice)),
        _quantum(quantum) {
    _build(config, resolved_int_type(request.quanta) * quantum,
           [](const auto &, const auto &, const auto &, auto &) {});
  }

  /// The build a carve may enter: once the world is built,
  /// `carve(world, apply_to_form, get_mesh_point, graph)` may state a carve
  /// into it (`state_reseat`) before the plane arrangement is made.
  template <typename Carve>
  arrangement_graph(Policy policy, lattice_type lattice,
                    tf::arrangement_config config, resolved_int_type quantum,
                    tf::arrangement::near_identity_request request,
                    const Carve &carve)
      : _policy(std::move(policy)), _lattice(std::move(lattice)),
        _quantum(quantum) {
    _build(config, resolved_int_type(request.quanta) * quantum, carve);
  }

  /// The carve this build states into its own world; the pieces' sources
  /// stay here for the exposure and the records. A carve stated into a
  /// world already carved appends: its pieces join the record after the
  /// pieces before them, and a piece a later carve retired keeps its name
  /// and leaves its slot. `row_images`, `canon_images`, `touched_planes` and
  /// `moved_rows` are what @ref
  /// tf::intersect::graph::local_arrangement::restate_carve states.
  template <typename Package>
  auto state_reseat(const Package &package, tf::buffer<index_type> &row_images,
                    tf::buffer<index_type> &canon_images,
                    tf::buffer<index_type> &touched_planes,
                    tf::buffer<std::array<index_type, 3>> &moved_rows) -> void {
    const auto first_face = _world.n_faces();
    auto retired = package.retired;
    auto restated = package.restated;
    std::sort(retired.begin(), retired.end());
    std::sort(restated.begin(), restated.end());
    _world.restate_carve(_identities, package.minted, package.welds, retired,
                         restated, package.piece_descriptors,
                         _carve_sources(package), package.piece_orientations,
                         package.piece_plane_of, package.new_frames,
                         package.new_plane_names, package.rows, package.planes,
                         _lattice.reader(_policy.make_apply_to_form()),
                         row_images, canon_images, touched_planes, moved_rows);
    _world.state_near_rows(package.near_rows);
    for (std::size_t k = 0; k < _piece_faces.size(); ++k)
      if (std::binary_search(retired.begin(), retired.end(), _piece_faces[k]))
        _piece_live[k] = char(0);
    for (std::size_t k = 0; k < package.piece_sources.size(); ++k) {
      _piece_sources.push_back(package.piece_sources[k]);
      _piece_sides.push_back(package.piece_sides[k]);
      _piece_corners.push_back(package.piece_corners[k]);
      _piece_faces.push_back(first_face + index_type(k));
      _piece_live.push_back(char(1));
    }
  }

  /// The pieces standing in this world, in its names now (@ref
  /// tf::arrangement::reseat::standing_pieces): the carve a census states
  /// enters a source face as it stands.
  auto standing_pieces() const
      -> tf::arrangement::reseat::standing_pieces<index_type> {
    tf::arrangement::reseat::standing_pieces<index_type> out;
    const auto n_tags = _policy.n_tags();
    tf::buffer<index_type> tag_base;
    tf::buffer<index_type> rank;
    tf::arrangement::reseat::rank_pieces_by_tag(_piece_sources, n_tags,
                                                tag_base, rank);
    out.stated.allocate(std::size_t(n_tags));
    for (std::size_t tag = 0; tag < std::size_t(n_tags); ++tag)
      out.stated[tag] = tag_base[tag + 1] - tag_base[tag];
    tf::buffer<std::array<tf::arrangement::reseat::piece_corner<index_type>, 3>>
        named;
    // a standing piece's corners are named as the graph stands, `P∘L`, in
    // the currency of the local tier whose writer the carve is
    const auto n_created = _world.n_created_points(_identities);
    tf::arrangement::reseat::name_piece_corners(
        [this](index_type source, index_type id) {
          return _identities.composed_target(source, id);
        },
        _world.vertex_offsets(), _piece_corners, named);
    out.foreign = tf::parallel_contains(
        named,
        [n_created](const auto &corners) {
          for (const auto &c : corners)
            if (c.tag < 0 && c.id >= n_created)
              return true;
          return false;
        },
        tf::checked);
    if (out.foreign)
      return out;
    tf::buffer<index_type> live;
    for (std::size_t k = 0; k < _piece_sources.size(); ++k)
      if (_piece_live[k])
        live.push_back(index_type(k));
    std::stable_sort(live.begin(), live.end(),
                     [&](index_type x, index_type y) {
                       return _piece_sources[std::size_t(x)] <
                              _piece_sources[std::size_t(y)];
                     });
    const auto faces = _world.face_offsets();
    for (const auto k : live) {
      const auto tag = _piece_sources[std::size_t(k)][0];
      const auto name = faces[std::size_t(tag) + 1] - faces[std::size_t(tag)] +
                        rank[std::size_t(k)];
      tf::arrangement::reseat::reseat_piece<index_type> piece;
      piece.tag = tag;
      piece.source = _piece_sources[std::size_t(k)][1];
      piece.sides = _piece_sides[std::size_t(k)];
      piece.corners = named[std::size_t(k)];
      out.by_key.push_back({tag, name, index_type(out.pieces.size())});
      out.keys.push_back(name);
      out.faces.push_back(_piece_faces[std::size_t(k)]);
      out.pieces.push_back(piece);
    }
    std::sort(out.by_key.begin(), out.by_key.end());
    return out;
  }

  /// The census's statement carved into this built world once the census
  /// has stated its welds (@ref
  /// tf::arrangement::reseat::carve_census): `minted` and `rows` as it
  /// reads them, `row_images`, `canon_images` and `moved_rows` what the
  /// statement states, empty when nothing was stated.
  auto carve_census(const tf::buffer<tf::point<resolved_int_type, 3>> &minted,
                    const tf::buffer<std::array<index_type, 3>> &rows,
                    tf::buffer<index_type> &row_images,
                    tf::buffer<index_type> &canon_images,
                    tf::buffer<index_type> &touched_planes,
                    tf::buffer<std::array<index_type, 3>> &moved_rows,
                    tf::buffer<index_type> &dying_faces) -> bool {
    static_assert(face_static_size == 3, "a carve cuts triangles");
    const auto apply_form = _policy.make_apply_to_form();
    return tf::arrangement::reseat::carve_census<index_type, resolved_int_type>(
        _world, standing_pieces(), minted, rows, apply_form,
        _lattice.reader(apply_form), _lattice.converter(), _policy.n_tags(),
        _config.intersect, *this, row_images, canon_images, touched_planes,
        moved_rows, dying_faces);
  }

  /// The census diff round, stated against the standing graph: the
  /// created-only welds into the standing tables, the carve of the new
  /// targets, the plane arrangement rebuilt as a diff value whose dirty
  /// set is exactly what the statements touched, and the one global
  /// re-read. The two row tables speak the world's created ids;
  /// `pa_welds` speak the exposed stream's flat ids with their witness
  /// planes, a weld onto an original naming the mint `carve_rows` retire it
  /// into. Returns false when the carve refuses — the graph stands as
  /// built.
  auto state_census_diff(
      const tf::buffer<std::array<index_type, 3>> &la_welds,
      const tf::buffer<tf::point<resolved_int_type, 3>> &carve_minted,
      const tf::buffer<std::array<index_type, 3>> &carve_rows,
      const tf::buffer<tf::arrangement::plane_flat_weld<index_type>>
          &pa_welds) -> bool {
    const auto apply_form = _policy.make_apply_to_form();
    auto get_mesh_point = _lattice.reader(apply_form);
    tf::buffer<index_type> la_rows, la_canons, la_dirty;
    tf::buffer<std::array<index_type, 3>> la_moved;
    _world.restate_census(_identities, la_welds, get_mesh_point, la_rows,
                          la_canons, la_dirty, la_moved);

    tf::buffer<index_type> cv_rows, cv_canons, cv_dirty, cv_dying;
    tf::buffer<std::array<index_type, 3>> cv_moved;
    // a refused carve leaves its relations unfused, and the round still
    // rebuilds over what was stated, so the graph never stands half-read;
    // a carve cuts triangles, so a dynamic-arity graph states no carve and
    // its retirements stand unfused on the same terms
    bool carved;
    if constexpr (face_static_size == 3)
      carved = carve_census(carve_minted, carve_rows, cv_rows, cv_canons,
                            cv_dirty, cv_moved, cv_dying);
    else
      carved = carve_rows.size() == 0 && carve_minted.size() == 0;

    const auto compose = [](tf::buffer<index_type> &first,
                            tf::buffer<index_type> &second) {
      if (first.size() == 0)
        return std::move(second);
      if (second.size() == 0)
        return std::move(first);
      tf::buffer<index_type> images;
      images.allocate(first.size());
      tf::parallel_transform(
          first, images,
          [&second](index_type mid) {
            return mid == index_type(-1) ? index_type(-1)
                                         : second[std::size_t(mid)];
          },
          tf::checked);
      return images;
    };
    auto canon_images = compose(la_canons, cv_canons);
    auto row_images = compose(la_rows, cv_rows);

    // the statements' evidence enters raw: the diff value composes it with
    // the local layer across its own extents (@ref
    // tf::arrangement::compose_local_moves)
    tf::buffer<index_type> dirty;
    tf::core::append(la_dirty, dirty);
    tf::core::append(cv_dirty, dirty);

    arrangement_type next;
    next.build_diff(_world, _identities, _arr, std::move(cv_moved), la_moved,
                    pa_welds, carved, row_images, canon_images, cv_dying,
                    std::move(dirty), get_mesh_point, apply_form);
    _arr = std::move(next);

    _expose(get_mesh_point);
    return carved;
  }

  /// The faces the world's records name: a tag's own, then the pieces a
  /// carve stated past them, whose corners past the tag's points are its
  /// seats.
  auto record_apply_to_face() const {
    return [this, world_form = _policy.make_apply_to_form()](
               int tag, index_type object, const auto &f) {
      const auto tag_faces = _world.face_offsets();
      const auto own =
          tag_faces[std::size_t(tag) + 1] - tag_faces[std::size_t(tag)];
      if (_piece_tail_corners.size() != 0 && object >= own) {
        f(_piece_tail_corners[std::size_t(_piece_tag_base[std::size_t(tag)] +
                                          (object - own))]);
        return;
      }
      world_form(index_type(tag),
                 [&](const auto &form) { f(form.faces()[object]); });
    };
  }
  /// The reader of the corners @ref record_apply_to_face names.
  auto record_reader() const {
    return _lattice.tail_reader(_policy.make_apply_to_form(),
                                tf::make_range(_seat_tail_offsets),
                                tf::make_range(_seat_tail_points));
  }

  /// The caller's own operands: every slot, original and source id this
  /// graph publishes names them.
  auto apply_to_form() const { return _policy.make_apply_to_form(); }
  auto policy() const -> const Policy & { return _policy; }
  /// The carve states its census on the policy; nothing else reaches it.
  auto policy() -> Policy & { return _policy; }
  auto n_tags() const -> index_type { return _policy.n_tags(); }

  auto converter() const -> const
      tf::exact::vertex_converter<resolved_int_type, pipeline_real_type, 3> & {
    return _lattice.converter();
  }

  /// @brief Where every original vertex of every operand stands on the
  ///        lattice — the one producer of that fact for this build.
  auto lattice() const -> const lattice_type & { return _lattice; }
  /// @brief The input's quantum on that lattice, read by every tier of
  ///        this build that decides what the input can tell apart; `0`
  ///        when no such tier runs.
  auto quantum() const -> resolved_int_type { return _quantum; }

  /// @brief The PA world: graph, local arrangement, point contracts.
  auto world() const -> const world_type & { return _world; }
  /// @brief The plane arrangement: triangles, pieces, spans.
  auto arrangement() const -> const arrangement_type & { return _arr; }
  /// @brief The identity substrate: every identity table the local and the
  ///        plane arrangement state, read here.
  auto identities() const -> const identities_type & { return _identities; }

  /// @brief The carriers that hold no product, by arrangement plane.
  ///
  /// EMPTY IS THE COMPLETENESS CLAIM: every plane that bounds area holds
  /// the triangulation of its constraint set, so every face of every
  /// operand holds the span this graph exposes for it. A plane is named
  /// here only when its triangulation refused every round of the recovery
  /// wave; when a wave names a fact the arrangement's producers cannot
  /// publish, every plane is named at once and the whole product is empty.
  /// A carrier that bounds no area — a line — holds its product by emitting
  /// nothing, and is named here on the same terms as every other.
  ///
  /// Quality is a different question: a refinement the discovery declined
  /// leaves its carrier the stock triangulation, which IS a product.
  auto failed() const { return _arr.failed(); }

  auto global() const -> global_view { return {this}; }

  auto vertex_offsets() const -> const tf::buffer<index_type> & {
    return _lattice.vertex_offsets();
  }
  auto created_points() const
      -> const tf::buffer<tf::point<resolved_int_type, 3>> & {
    return _created_points;
  }
  /// Per exposed triangle: the tag of its slot's descriptor — the
  /// descriptor states it, for consumers that walk the stream without its
  /// slots.
  auto triangle_tags() const {
    const auto *graph = this;
    return tf::make_mapped_range(
        tf::make_range(_triangle_slots), [graph](index_type slot) {
          return index_type(
              graph->_exposed_descriptors[std::size_t(slot)].tag);
        });
  }
  auto triangle_slots() const { return tf::make_range(_triangle_slots); }
  /// @brief Per exposed triangle: `1` = dead coplanar duplicate.
  auto dead() const {
    const auto *graph = this;
    return tf::make_mapped_range(
        tf::make_range(_row_of), [graph](index_type row) {
          return char(graph->_arr.coplanar_of()[std::size_t(row)] !=
                      index_type(-1));
        });
  }
  /// @brief Per exposed triangle: `1` = member of a coincident stack
  ///        (survivor AND dead).
  auto stacked() const {
    const auto *graph = this;
    return tf::make_mapped_range(
        tf::make_range(_row_of), [graph](index_type row) {
          return graph->_arr.stacked()[std::size_t(row)];
        });
  }
  /// @brief Coincident stacks, sorted by survivor — a survivor's partners
  ///        are contiguous and its smallest opposing partner is found in
  ///        one binary search.
  auto coplanar_triples() const { return tf::make_range(_triples); }

  /// @brief The arrangement's 2-cells, numbered densely across its planes.
  ///
  /// A cell is bounded by CONSTRAINTS and nothing else — rims included —
  /// so a walk inside one crosses no edge the arrangement stated, and a
  /// filler diagonal never leaves the cell it was cut in. This is the
  /// carrier classification stands on; a coplanar duplicate carries its
  /// survivor's cell because they are the same cell.
  auto cells() const
      -> const tf::arrangement::plane_arrangement_cells<index_type> & {
    return _cells;
  }
  /// @brief The piece <-> cell incidence: a cell's boundary is pieces and
  ///        nothing else, so this IS the adjacency a component flood walks.
  auto piece_incidence() const
      -> const tf::arrangement::plane_piece_incidence<index_type> & {
    return _incidence;
  }
  /// @brief Per piece: does it carry a fan, and may the flood cross it.
  ///        A constraint bounds; the fence decides.
  auto piece_fences() const -> const tf::arrangement::plane_piece_fences & {
    return _fences;
  }
  /// @brief Per exposed triangle: the arrangement row it exposes. The
  ///        piece incidence and the cells speak rows, the classification
  ///        speaks the stream; this is the one crossing between them.
  auto row_of() const { return tf::make_range(_row_of); }
  /// @brief Its inverse: arrangement row -> its exposed triangle, `-1`
  ///        for a span the exposure did not carry.
  auto exposed_of_row() const { return tf::make_range(_exposed_of_row); }
  /// @brief Per exposed slot: `1` = its plane carries more than one
  ///        member, so the pooled CDT can hand one member's triangle a
  ///        corner owned by another form. A single-member slot's
  ///        originals are its own tag's by construction.
  auto pooled_slots() const {
    const auto *graph = this;
    return tf::make_mapped_range(
        tf::make_range(_exposed_descriptors),
        [graph](const exposed_descriptor &descriptor) {
          const auto carrier = descriptor.plane;
          return char(carrier != index_type(-1) &&
                      tf::arrangement::plane_arrangement_member_count(
                          graph->_arr, graph->_world, carrier) >
                          index_type(1));
        });
  }
  /// @brief The form a flat-original id belongs to.
  auto tag_of_flat(index_type id) const -> index_type {
    return tf::exact::tag_of_flat_vertex(_lattice.vertex_offsets(), id);
  }
  /// @brief Exposed slot -> its `[begin, end)` triangle range.
  auto slot_range(index_type slot) const -> std::array<index_type, 2> {
    const auto &descriptor = _exposed_descriptors[std::size_t(slot)];
    const auto base =
        _face_offset_base[std::size_t(descriptor.tag)] + descriptor.object;
    return {_face_offsets[std::size_t(base)],
            _face_offsets[std::size_t(base) + 1]};
  }
  /// @brief `(tag, object)` of every uncut face refinement pulled into
  ///        the graph, sorted.
  auto refine_promoted() const { return tf::make_range(_refine_promoted); }
  auto with_self() const -> bool { return _with_self; }
  auto config() const -> const tf::arrangement_config & { return _config; }

private:
  template <typename Carve>
  auto _build(tf::arrangement_config a_config, resolved_int_type near_window,
              const Carve &carve) -> void {
    _config = a_config;
    auto config = a_config.intersect;
    const auto n_tags = _policy.n_tags();
    if (n_tags == index_type(1))
      config.mode = config.mode | tf::intersect_mode::within;
    _with_self = bool(config.mode & tf::intersect_mode::within);

    tf::polygon_intersections<index_type, pipeline_real_type,
                              resolved_int_type>
        intersections;
    intersections.with_edge_splits(false);
    _policy.build_intersections(intersections, _lattice, config);

    auto apply_form = _policy.make_apply_to_form();
    auto apply_to_face = [apply_form](int tag, index_type object,
                                      const auto &f) {
      apply_form(index_type(tag),
                 [&](const auto &form) { f(form.faces()[object]); });
    };
    // Every tier below reads an original's position through this and
    // through nothing else, so the door's placement reaches the plane
    // graph, the triangulation, the created points and the emission by
    // being the reader's answer rather than by being copied.
    auto get_mesh_point = _lattice.reader(apply_form);

    tf::buffer<index_type> face_counts;
    face_counts.allocate(std::size_t(n_tags) + 1);
    face_counts[0] = index_type(0);
    for (index_type t = 0; t < n_tags; ++t)
      apply_form(t, [&, t](const auto &form) {
        face_counts[std::size_t(t) + 1] =
            face_counts[std::size_t(t)] + index_type(form.faces().size());
      });

    const bool refined =
        a_config.triangulation == tf::triangulation_type::refined_cdt;
    tf::cdt_refine_config refine_config;
    _world.record_near_identities(near_window);
    _arr.record_near_identities(near_window);
    _world.build(_identities, std::move(intersections), get_mesh_point,
                 apply_to_face, apply_form, tf::make_range(face_counts),
                 bool(config.mode & tf::intersect_mode::within),
                 _lattice.moved_points().size() != 0);
    carve(_world, apply_form, get_mesh_point, *this);
    // the cell is the classification carrier and the piece is the fence's;
    // the triangulation holds both only while the plane is being emitted, so
    // the requests stand before the build
    _arr.record_triangle_cells();
    _arr.record_triangle_arrangement();
    if (refined)
      _arr.build_refined(_world, _identities, get_mesh_point, apply_form,
                         refine_config);
    else
      _arr.build(_world, _identities, get_mesh_point, apply_form);
    _world.release_intersection_points();

    _expose(get_mesh_point);
  }

  /// The exposure adapter: PA's plane-major product laid out in the
  /// stream's tag-major face order, corners in the stream's vertex
  /// language, one slot per face of every form. Every per-row fact is the
  /// emission's own statement — this reads it, it derives nothing.
  template <typename GetMeshPoint>
  auto _expose(const GetMeshPoint &get_mesh_point) -> void {
    const auto n_tags = _policy.n_tags();
    const auto face_counts = _world.face_offsets();
    const auto n_slots = std::size_t(face_counts[std::size_t(n_tags)]);
    const auto triangles = _arr.triangles();
    const auto coplanar_of = _arr.coplanar_of();
    const auto n_flat = _arr.n_flat_points();

    // slot -> arrangement face, -1 = uncut. Every arrangement face names a
    // distinct `(tag, object)`, so the scatter is injective: the world's
    // descriptors are one per face and the promotion masks the faces it
    // pulls in against them.
    tf::buffer<index_type> face_of_slot;
    face_of_slot.allocate(n_slots);
    tf::parallel_fill(face_of_slot, index_type(-1));
    // a carve's pieces are world faces named past their tag's own faces
    const auto is_piece = [&](const auto &descriptor) {
      return descriptor.object >= face_counts[std::size_t(descriptor.tag) + 1] -
                                      face_counts[std::size_t(descriptor.tag)];
    };
    tf::parallel_for_each(
        tf::make_sequence_range(_arr.n_faces()),
        [&](index_type face) {
          const auto &descriptor =
              tf::arrangement::plane_arrangement_face_descriptor(_arr, _world,
                                                                 face);
          if (is_piece(descriptor))
            return;
          face_of_slot[std::size_t(
              face_counts[std::size_t(descriptor.tag)] + descriptor.object)] =
              face;
        },
        tf::checked);
    // a carve's standing pieces join the slot of the source face they were
    // cut from
    const auto n_records = _piece_sources.size();
    tf::buffer<index_type> slot_piece_offsets;
    tf::buffer<index_type> slot_pieces;
    if (n_records != 0) {
      slot_piece_offsets.allocate_and_initialize(n_slots + 1, index_type(0));
      for (std::size_t k = 0; k < n_records; ++k) {
        if (!_piece_live[k])
          continue;
        const auto &source = _piece_sources[k];
        ++slot_piece_offsets[std::size_t(
            face_counts[std::size_t(source[0])] + source[1]) + 1];
      }
      for (std::size_t at = 1; at < slot_piece_offsets.size(); ++at)
        slot_piece_offsets[at] += slot_piece_offsets[at - 1];
      slot_pieces.allocate(std::size_t(slot_piece_offsets[n_slots]));
      tf::buffer<index_type> cursor;
      cursor.allocate(n_slots);
      std::copy(slot_piece_offsets.begin(), slot_piece_offsets.end() - 1,
                cursor.begin());
      for (std::size_t k = 0; k < n_records; ++k) {
        if (!_piece_live[k])
          continue;
        const auto &source = _piece_sources[k];
        const auto slot =
            std::size_t(face_counts[std::size_t(source[0])] + source[1]);
        slot_pieces[std::size_t(cursor[slot]++)] = index_type(k);
      }
    }
    // the scatter's own contract, checked after its barrier: a face that
    // shares a slot finds the winner's id in it, so any tier that promotes
    // a face the world already names fails here and not silently; a piece
    // joins its source's slot through the slot CSR instead
    if constexpr (tf::core::verification) {
      tf::parallel_for_each(
          tf::make_sequence_range(_arr.n_faces()),
          [&](index_type face) {
            const auto &descriptor =
                tf::arrangement::plane_arrangement_face_descriptor(_arr, _world,
                                                                   face);
            if (is_piece(descriptor))
              return;
            tf::core::verify(
                face_of_slot[std::size_t(
                    face_counts[std::size_t(descriptor.tag)] +
                    descriptor.object)] == face,
                "two arrangement faces name one (tag, object)");
          },
          tf::checked);
      tf::arrangement::assert_promotion_is_complete(
          _identities, _world, face_counts,
          tf::make_range(_lattice.vertex_offsets()), face_of_slot,
          _policy.make_apply_to_form());
      tf::arrangement::assert_plane_layer_is_closed(
          _identities, triangles, n_flat, _arr.n_base_created());
      if (_arr.failed().size() == 0)
        tf::arrangement::assert_created_identities_are_accounted(
            _identities, _world, _arr);
    }

    // the caller's layout: one slot per caller face, spanned by its own
    // arrangement face and then by the pieces a carve stated of it
    const auto with_piece_count = [&](const auto &pass) {
      if (n_records == 0)
        pass([](std::size_t) { return std::size_t(1); });
      else
        pass([&](std::size_t slot) {
          return std::size_t(1) + std::size_t(slot_piece_offsets[slot + 1] -
                                              slot_piece_offsets[slot]);
        });
    };
    // the `k`-th piece of a slot, `k >= 1`, as its record
    const auto piece_at = [&](std::size_t slot, std::size_t k) -> index_type {
      return slot_pieces[std::size_t(slot_piece_offsets[slot]) + (k - 1)];
    };
    const auto face_at = [&](std::size_t slot, std::size_t k) -> index_type {
      return k == 0 ? face_of_slot[slot]
                    : _piece_faces[std::size_t(piece_at(slot, k))];
    };

    // descriptors + fenceposted per-slot spans, then one serial prefix
    _face_offset_base.allocate(std::size_t(n_tags) + 1);
    _face_offset_base[0] = index_type(0);
    for (index_type tag = 0; tag < n_tags; ++tag)
      _face_offset_base[std::size_t(tag) + 1] =
          _face_offset_base[std::size_t(tag)] +
          (face_counts[std::size_t(tag) + 1] - face_counts[std::size_t(tag)]) +
          index_type(1);
    _face_offsets.allocate(
        std::size_t(_face_offset_base[std::size_t(n_tags)]));
    _tag_offsets.allocate(std::size_t(n_tags) + 1);
    _tag_offsets[0] = index_type(0);
    _exposed_descriptors.allocate(n_slots);

    with_piece_count([&](const auto &piece_count) {
      tf::parallel_for_each(
          tf::make_sequence_range(n_slots),
          [&](std::size_t slot) {
            const auto tag = std::size_t(std::upper_bound(face_counts.begin(),
                                                          face_counts.end(),
                                                          index_type(slot)) -
                                         face_counts.begin() - 1);
            const auto begin = std::size_t(face_counts[tag]);
            const auto base = std::size_t(_face_offset_base[tag]);
            index_type plane = -1;
            index_type count = 0;
            for (std::size_t k = 0; k < piece_count(slot); ++k) {
              const auto face = face_at(slot, k);
              if (face == index_type(-1))
                continue;
              const auto triangle_span = _arr.face_range(face);
              if (plane == index_type(-1) &&
                  (triangle_span[1] > triangle_span[0] ||
                   k + 1 == piece_count(slot)))
                plane = tf::arrangement::plane_arrangement_face_plane(
                    _arr, _world, face);
              count += triangle_span[1] - triangle_span[0];
            }
            _exposed_descriptors[slot] = {short(tag), index_type(slot - begin),
                                          plane};
            _face_offsets[base + (slot - begin) + 1] = count;
          },
          tf::checked);
    });
    index_type at = 0;
    for (index_type tag = 0; tag < n_tags; ++tag) {
      const auto base = std::size_t(_face_offset_base[std::size_t(tag)]);
      const auto count = std::size_t(face_counts[std::size_t(tag) + 1] -
                                     face_counts[std::size_t(tag)]);
      _face_offsets[base] = at;
      for (std::size_t object = 0; object < count; ++object) {
        at += _face_offsets[base + object + 1];
        _face_offsets[base + object + 1] = at;
      }
      _tag_offsets[std::size_t(tag) + 1] = at;
    }

    const auto corner_subs = _arr.corner_subs();
    const auto source_forms = _policy.make_apply_to_form();

    // the stream: corners in the vertex language, the sub the emission
    // stamped on each, and the row every per-row fact is read through
    _exposed_tris.allocate(std::size_t(at));
    _triangle_slots.allocate(std::size_t(at));
    _row_of.allocate(std::size_t(at));
    _exposed_of_row.allocate(triangles.size());
    tf::parallel_fill(_exposed_of_row, index_type(-1));
    using vsource = tf::intersect::graph::vertex_source;
    with_piece_count([&](const auto &piece_count) {
      tf::parallel_for_each(
          tf::make_sequence_range(n_slots),
          [&](std::size_t slot) {
            const auto tag = _exposed_descriptors[slot].tag;
            const auto base = std::size_t(_face_offset_base[std::size_t(tag)]);
            const auto object = std::size_t(_exposed_descriptors[slot].object);
            auto write = std::size_t(_face_offsets[base + object]);
            for (std::size_t k = 0; k < piece_count(slot); ++k) {
              const auto face = face_at(slot, k);
              if (face == index_type(-1))
                continue;
              const auto triangle_span = _arr.face_range(face);
              for (auto row = triangle_span[0]; row < triangle_span[1]; ++row) {
                const auto &corners = triangles[std::size_t(row)];
                const auto &subs = corner_subs[std::size_t(row)];
                auto &out = _exposed_tris[write];
                for (int c = 0; c < 3; ++c) {
                  const auto flat = corners[std::size_t(c)];
                  out[std::size_t(c)] =
                      flat < n_flat ? vertex_t{vsource::original, flat,
                                               subs[std::size_t(c)]}
                                    : vertex_t{vsource::created,
                                               index_type(flat - n_flat),
                                               subs[std::size_t(c)]};
                }
                if (k != 0)
                  _stated_piece_subs(source_forms, index_type(tag),
                                     index_type(object), piece_at(slot, k),
                                     out);
                _triangle_slots[write] = index_type(slot);
                _row_of[write] = index_type(row);
                _exposed_of_row[std::size_t(row)] = index_type(write);
                ++write;
              }
            }
          },
          tf::checked);
    });

    // the arrangement's own structure: its cells, the pieces bounding them,
    // and each piece's verdict. Counts plus one prefix on both sides — the
    // identity spaces are dense, so nothing is sorted into place.
    _cells = tf::arrangement::make_plane_arrangement_cells(_arr);
    _incidence = tf::arrangement::make_plane_piece_incidence(_arr, _cells);
    _fences = tf::arrangement::make_plane_piece_fences(_arr, _world, _incidence,
                                               _cells);

    // the stack triples in the stream's indices: the arrangement states the
    // stacks, the exposure only translates them
    const auto coplanar_descriptors = _arr.coplanar_descriptors();
    _triples.clear();
    for (std::size_t row = 0; row < triangles.size(); ++row) {
      const auto index = coplanar_of[row];
      if (index == index_type(-1))
        continue;
      const auto &descriptor = coplanar_descriptors[std::size_t(index)];
      const auto dead = _exposed_of_row[row];
      const auto survivor =
          _exposed_of_row[std::size_t(descriptor.survivor)];
      if (dead == index_type(-1) || survivor == index_type(-1))
        continue; // unexposed span
      _triples.push_back({survivor, dead, descriptor.opposing});
    }
    std::sort(_triples.begin(), _triples.end(),
              [](const auto &a, const auto &b) {
                return std::tie(a.survivor, a.dead) <
                       std::tie(b.survivor, b.dead);
              });
    const auto promoted = _arr.promoted_descriptors();
    _refine_promoted.allocate(promoted.size());
    tf::parallel_transform(
        promoted, _refine_promoted,
        [](const auto &descriptor) {
          return std::array<index_type, 2>{index_type(descriptor.tag),
                                           descriptor.object};
        },
        tf::checked);
    tbb::parallel_sort(_refine_promoted.begin(), _refine_promoted.end());

    // the unified created-points table, materialized once: the world's
    // points first, then everything this arrangement minted, each where the
    // composed layer states it
    const auto base_created = _arr.n_base_created();
    _created_points.allocate(
        std::size_t(base_created + _identities.n_plane_created()));
    tf::parallel_for_each(
        tf::make_sequence_range(
            index_type(base_created + _identities.n_plane_created())),
        [&](index_type id) {
          _created_points[std::size_t(id)] =
              tf::arrangement::composed_identity_point(
                  _identities, _world, _arr, get_mesh_point, std::int16_t(-1),
                  id);
        },
        tf::checked);
    if (_piece_sources.size() != 0)
      _state_piece_records();
    _seat_tail_points.allocate(_seat_tail.size());
    for (std::size_t k = 0; k < _seat_tail.size(); ++k) {
      const auto &seat = _seat_tail[k];
      _seat_tail_points[k] =
          seat.tag < 0 ? _created_points[std::size_t(seat.id)]
                       : get_mesh_point(int(seat.tag), seat.id);
    }
  }

  /// Per face the carve enters, its source face and the source side each of
  /// its sides keeps whole: a piece's side keeps one only when both its
  /// corners are that side's own, and an entered face is its own source.
  template <typename Package>
  auto _carve_sources(const Package &package) const
      -> tf::buffer<std::array<index_type, 4>> {
    tf::buffer<std::array<index_type, 4>> sources;
    const auto apply_form = _policy.make_apply_to_form();
    const auto n_pieces = package.piece_sources.size();
    for (std::size_t k = 0; k < n_pieces; ++k) {
      const auto tag = package.piece_sources[k][0];
      std::array<index_type, 4> source{package.piece_sources[k][1],
                                       index_type(-1), index_type(-1),
                                       index_type(-1)};
      apply_form(tag, [&](const auto &form) {
        const auto face = form.faces()[package.piece_sources[k][1]];
        for (std::size_t at = 0; at < 3; ++at) {
          const auto side = package.piece_sides[k][at];
          const auto &a = package.piece_corners[k][at];
          const auto &b = package.piece_corners[k][(at + 1) % 3];
          if (side < 0 || index_type(a.tag) != tag || index_type(b.tag) != tag)
            continue;
          const auto u = index_type(face[std::size_t(side)]);
          const auto v =
              index_type(face[(std::size_t(side) + 1) % face.size()]);
          if ((a.id == u && b.id == v) || (a.id == v && b.id == u))
            source[at + 1] = index_type(side);
        }
      });
      sources.push_back(source);
    }
    for (auto k = n_pieces; k < package.piece_descriptors.size(); ++k)
      sources.push_back({package.piece_descriptors[k].object, index_type(0),
                         index_type(1), index_type(2)});
    return sources;
  }

  /// The records' language over every piece stated: a piece is a face past
  /// its tag's own, named in the order its tag's pieces were stated, and a
  /// corner its own form does not hold — a created identity, or an original
  /// of another form — a seat past the tag's points. A corner speaks the
  /// composed answer `P∘L` the exposed triangles speak, so the corners a weld
  /// of either layer folds are one seat.
  auto _state_piece_records() -> void {
    const auto n_tags = _policy.n_tags();
    tf::buffer<index_type> rank;
    tf::arrangement::reseat::rank_pieces_by_tag(_piece_sources, n_tags,
                                                _piece_tag_base, rank);
    tf::buffer<std::array<tf::arrangement::reseat::piece_corner<index_type>, 3>>
        merged;
    tf::arrangement::reseat::name_piece_corners(
        [this](index_type source, index_type id) {
          return _identities.composed_target(source, id);
        },
        _world.vertex_offsets(), _piece_corners, merged);
    tf::buffer<std::array<tf::arrangement::reseat::piece_corner<index_type>, 3>>
        named;
    named.allocate(_piece_sources.size());
    tf::parallel_for_each(
        tf::make_sequence_range(_piece_sources.size()),
        [&](std::size_t k) {
          named[std::size_t(
              _piece_tag_base[std::size_t(_piece_sources[k][0])] + rank[k])] =
              merged[k];
        },
        tf::checked);
    _seat_tail.clear();
    _seat_tail_offsets.clear();
    _seat_tail_offsets.push_back(index_type(0));
    for (index_type tag = 0; tag < n_tags; ++tag) {
      const auto begin = _seat_tail.size();
      for (auto at = std::size_t(_piece_tag_base[std::size_t(tag)]);
           at < std::size_t(_piece_tag_base[std::size_t(tag) + 1]); ++at)
        for (const auto &c : named[at])
          if (index_type(c.tag) != tag)
            _seat_tail.push_back(c);
      std::sort(_seat_tail.begin() + std::ptrdiff_t(begin), _seat_tail.end());
      _seat_tail.erase_till_end(
          std::unique(_seat_tail.begin() + std::ptrdiff_t(begin),
                      _seat_tail.end()));
      _seat_tail_offsets.push_back(index_type(_seat_tail.size()));
    }
    _piece_tail_corners.allocate(_piece_sources.size());
    for (index_type tag = 0; tag < n_tags; ++tag) {
      const auto begin =
          _seat_tail.begin() + _seat_tail_offsets[std::size_t(tag)];
      const auto end =
          _seat_tail.begin() + _seat_tail_offsets[std::size_t(tag) + 1];
      for (auto at = std::size_t(_piece_tag_base[std::size_t(tag)]);
           at < std::size_t(_piece_tag_base[std::size_t(tag) + 1]); ++at) {
        const auto &piece = named[at];
        auto &corners = _piece_tail_corners[at];
        for (std::size_t c = 0; c < 3; ++c)
          corners[c] =
              index_type(piece[c].tag) == tag
                  ? piece[c].id
                  : _lattice.vertex_offsets()[std::size_t(tag) + 1] -
                        _lattice.vertex_offsets()[std::size_t(tag)] +
                        index_type(std::lower_bound(begin, end, piece[c]) -
                                   begin);
      }
    }
  }

  /// A stated piece's corners restated on the source face it was cut from,
  /// by the piece's own sides and corners.
  template <typename SourceForms>
  auto _stated_piece_subs(const SourceForms &source_forms, index_type tag,
                          index_type object, index_type piece,
                          std::array<vertex_t, 3> &corners) const -> void {
    const tf::topo_id<short> inside{short(0), tf::topo_type::face};
    const auto &sides = _piece_sides[std::size_t(piece)];
    const auto &pc = _piece_corners[std::size_t(piece)];
    source_forms(tag, [&](const auto &form) {
      const auto source = form.faces()[object];
      for (auto &corner : corners) {
        const auto &sub = corner.sub_id;
        const auto k = std::size_t(sub.id);
        if (sub.label == tf::topo_type::edge) {
          corner.sub_id = sides[k] < 0 ? inside
                                       : tf::topo_id<short>{short(sides[k]),
                                                            tf::topo_type::edge};
          continue;
        }
        if (sub.label != tf::topo_type::vertex)
          continue;
        if (pc[k].tag >= 0 && index_type(pc[k].tag) == tag) {
          tf::topo_id<short> found = inside;
          for (std::size_t i = 0; i < std::size_t(source.size()); ++i)
            if (index_type(source[i]) == pc[k].id)
              found = {short(i), tf::topo_type::vertex};
          corner.sub_id = found;
          continue;
        }
        const auto edge = sides[k] >= 0 ? sides[k] : sides[(k + 2) % 3];
        corner.sub_id = edge < 0 ? inside
                                 : tf::topo_id<short>{short(edge),
                                                      tf::topo_type::edge};
      }
    });
  }

  Policy _policy;
  lattice_type _lattice;
  resolved_int_type _quantum = resolved_int_type(0);
  identities_type _identities;
  world_type _world;
  arrangement_type _arr;
  tf::buffer<tf::point<resolved_int_type, 3>> _created_points;
  tf::buffer<std::array<vertex_t, 3>> _exposed_tris;
  tf::buffer<exposed_descriptor> _exposed_descriptors;
  tf::buffer<index_type> _face_offset_base;
  tf::buffer<index_type> _face_offsets;
  tf::buffer<index_type> _tag_offsets;
  tf::buffer<index_type> _triangle_slots;
  /// Per exposed triangle: the arrangement row it exposes. Every per-row
  /// fact the stream answers is one lookup through this.
  tf::buffer<index_type> _row_of;
  /// Its inverse, `-1` where the exposure carried no span.
  tf::buffer<index_type> _exposed_of_row;
  tf::arrangement::plane_arrangement_cells<index_type> _cells;
  tf::arrangement::plane_piece_incidence<index_type> _incidence;
  tf::arrangement::plane_piece_fences _fences;
  tf::buffer<tf::arrangement::plane_coplanar_triple<index_type>> _triples;
  tf::buffer<std::array<index_type, 2>> _refine_promoted;
  tf::arrangement_config _config;
  bool _with_self = false;
  // per piece the carves stated, in the order they stated them: its source
  // face, the source side each side lies on, its corners, its world face,
  // and whether it stands — a later carve that cut its source retired it
  tf::buffer<std::array<index_type, 2>> _piece_sources;
  tf::buffer<std::array<std::int8_t, 3>> _piece_sides;
  tf::buffer<std::array<tf::arrangement::reseat::piece_corner<index_type>, 3>>
      _piece_corners;
  tf::buffer<index_type> _piece_faces;
  tf::buffer<char> _piece_live;
  // per tag, its first piece name: the records' pieces by tag, each tag's
  // in the order it stated them
  tf::buffer<index_type> _piece_tag_base;
  tf::buffer<tf::arrangement::reseat::piece_corner<index_type>> _seat_tail;
  tf::buffer<index_type> _seat_tail_offsets;
  tf::buffer<tf::point<resolved_int_type, 3>> _seat_tail_points;
  // per piece name, `_piece_tag_base` order: its corners in the records'
  // language
  tf::buffer<std::array<index_type, 3>> _piece_tail_corners;
};

} // namespace tf
