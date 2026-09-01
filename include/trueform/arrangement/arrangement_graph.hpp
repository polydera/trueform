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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/range.hpp"
#include "../core/transformed.hpp"
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
#include "./planes/assert_promotion_is_complete.hpp"
#include "./planes/make_plane_arrangement_cells.hpp"
#include "./planes/make_plane_piece_fences.hpp"
#include "./planes/make_plane_piece_incidence.hpp"
#include "./planes/plane_arrangement.hpp"
#include "./planes/plane_arrangement_face.hpp"
#include "./planes/plane_coplanar_triple.hpp"
#include "./planes/plane_triangulation_types.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cassert>
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

  /// The lattice view is built above this graph, over the same operands,
  /// and handed in: it is where every original vertex stands, and this
  /// build has no second opinion about it.
  arrangement_graph(Policy policy, lattice_type lattice,
                    tf::arrangement_config config = {})
      : _policy(std::move(policy)), _lattice(std::move(lattice)) {
    _build(config);
  }

  auto apply_to_form() const { return _policy.make_apply_to_form(); }
  auto policy() const -> const Policy & { return _policy; }
  auto n_tags() const -> index_type { return _policy.n_tags(); }

  auto converter() const -> const
      tf::exact::vertex_converter<resolved_int_type, pipeline_real_type, 3> & {
    return _lattice.converter();
  }

  /// @brief Where every original vertex of every operand stands on the
  ///        lattice — the one producer of that fact for this build.
  auto lattice() const -> const lattice_type & { return _lattice; }

  /// @brief The PA world: graph, local arrangement, point contracts.
  auto world() const -> const world_type & { return _world; }
  /// @brief The plane arrangement: triangles, pieces, spans.
  auto arrangement() const -> const arrangement_type & { return _arr; }

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
    return _vertex_offsets;
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
                      carrier < graph->_world.n_planes() &&
                      graph->_world.member_count(carrier) > index_type(1));
        });
  }
  /// @brief The form a flat-original id belongs to.
  auto tag_of_flat(index_type id) const -> index_type {
    return tf::exact::tag_of_flat_vertex(_vertex_offsets, id);
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
  auto _build(tf::arrangement_config a_config) -> void {
    _config = a_config;
    auto config = a_config.intersect;
    const auto n_tags = _policy.n_tags();
    if (n_tags == index_type(1))
      config.mode = config.mode | tf::intersect_mode::within;
    _with_self = bool(config.mode & tf::intersect_mode::self_intersections);

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

    const auto &lattice_offsets = _lattice.vertex_offsets();
    _vertex_offsets.allocate(lattice_offsets.size());
    std::copy(lattice_offsets.begin(), lattice_offsets.end(),
              _vertex_offsets.begin());
    tf::buffer<index_type> face_counts;
    face_counts.allocate(std::size_t(n_tags) + 1);
    face_counts[0] = index_type(0);
    for (index_type t = 0; t < n_tags; ++t)
      apply_form(t, [&, t](const auto &form) {
        face_counts[std::size_t(t) + 1] =
            face_counts[std::size_t(t)] + index_type(form.faces().size());
      });

    const bool resolve_self_contours =
        bool(config.mode & tf::intersect_mode::resolve_self_crossing_contours);
    const bool refined =
        a_config.triangulation == tf::triangulation_type::refined_cdt;
    tf::cdt_refine_config refine_config;
    _world.build(std::move(intersections), get_mesh_point, apply_to_face,
                 apply_form, tf::make_range(face_counts),
                 resolve_self_contours, _lattice.placed_points().size() != 0);
    // the cell is the classification carrier and the piece is the fence's;
    // the triangulation holds both only while the plane is being emitted, so
    // the requests stand before the build
    _arr.record_triangle_cells();
    _arr.record_triangle_arrangement();
    if (refined)
      _arr.build_refined(_world, get_mesh_point, apply_form, refine_config);
    else
      _arr.build(_world, get_mesh_point, apply_form);
    _world.release_intersection_points();

    _expose(face_counts, get_mesh_point);
  }

  /// The exposure adapter: PA's plane-major product laid out in the
  /// stream's tag-major face order, corners in the stream's vertex
  /// language, one slot per face of every form. Every per-row fact is the
  /// emission's own statement — this reads it, it derives nothing.
  template <typename GetMeshPoint>
  auto _expose(const tf::buffer<index_type> &face_counts,
               const GetMeshPoint &get_mesh_point) -> void {
    const auto n_tags = _policy.n_tags();
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
    tf::parallel_for_each(
        tf::make_sequence_range(_arr.n_faces()),
        [&](index_type face) {
          const auto &descriptor =
              tf::arrangement::plane_arrangement_face_descriptor(_arr, _world,
                                                                 face);
          face_of_slot[std::size_t(
              face_counts[std::size_t(descriptor.tag)] + descriptor.object)] =
              face;
        },
        tf::checked);
#ifndef NDEBUG
    // the scatter's own contract, checked after its barrier: a face that
    // shares a slot finds the winner's id in it, so any tier that promotes
    // a face the world already names fails here and not silently
    tf::parallel_for_each(
        tf::make_sequence_range(_arr.n_faces()),
        [&](index_type face) {
          const auto &descriptor =
              tf::arrangement::plane_arrangement_face_descriptor(_arr, _world,
                                                                 face);
          assert(face_of_slot[std::size_t(
                     face_counts[std::size_t(descriptor.tag)] +
                     descriptor.object)] == face &&
                 "two arrangement faces name one (tag, object)");
        },
        tf::checked);
    tf::arrangement::assert_promotion_is_complete(
        _world, _arr, face_counts, tf::make_range(_vertex_offsets),
        face_of_slot, _policy.make_apply_to_form());
#endif

    // descriptors + fenceposted per-face spans, then one serial prefix
    _face_offset_base.allocate(std::size_t(n_tags) + 1);
    _face_offset_base[0] = index_type(0);
    for (index_type tag = 0; tag < n_tags; ++tag)
      _face_offset_base[std::size_t(tag) + 1] =
          _face_offset_base[std::size_t(tag)] +
          (face_counts[std::size_t(tag) + 1] -
           face_counts[std::size_t(tag)]) +
          index_type(1);
    _face_offsets.allocate(
        std::size_t(_face_offset_base[std::size_t(n_tags)]));
    _tag_offsets.allocate(std::size_t(n_tags) + 1);
    _tag_offsets[0] = index_type(0);
    _exposed_descriptors.allocate(n_slots);

    tf::parallel_for_each(
        tf::make_sequence_range(index_type(n_tags)),
        [&](index_type tag) {
          const auto begin = std::size_t(face_counts[std::size_t(tag)]);
          const auto end = std::size_t(face_counts[std::size_t(tag) + 1]);
          const auto base = std::size_t(_face_offset_base[std::size_t(tag)]);
          for (auto slot = begin; slot < end; ++slot) {
            const auto face = face_of_slot[slot];
            _exposed_descriptors[slot] = {
                short(tag), index_type(slot - begin),
                face == index_type(-1)
                    ? index_type(-1)
                    : tf::arrangement::plane_arrangement_face_plane(
                          _arr, _world, face)};
            const auto range = face == index_type(-1)
                                   ? std::array<index_type, 2>{0, 0}
                                   : _arr.face_range(face);
            _face_offsets[base + (slot - begin) + 1] = range[1] - range[0];
          }
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

    // the stream: corners in the vertex language, the sub the emission
    // stamped on each, and the row every per-row fact is read through
    _exposed_tris.allocate(std::size_t(at));
    _triangle_slots.allocate(std::size_t(at));
    _row_of.allocate(std::size_t(at));
    _exposed_of_row.allocate(triangles.size());
    tf::parallel_fill(_exposed_of_row, index_type(-1));
    using vsource = tf::intersect::graph::vertex_source;
    tf::parallel_for_each(
        tf::make_sequence_range(n_slots),
        [&](std::size_t slot) {
          const auto face = face_of_slot[slot];
          if (face == index_type(-1))
            return;
          const auto tag = _exposed_descriptors[slot].tag;
          const auto base = std::size_t(_face_offset_base[std::size_t(tag)]);
          const auto object = std::size_t(_exposed_descriptors[slot].object);
          auto write = std::size_t(_face_offsets[base + object]);
          const auto range = _arr.face_range(face);
          for (auto row = range[0]; row < range[1]; ++row) {
            const auto &corners = triangles[std::size_t(row)];
            const auto &subs = corner_subs[std::size_t(row)];
            auto &out = _exposed_tris[write];
            for (int c = 0; c < 3; ++c) {
              const auto flat = corners[std::size_t(c)];
              out[std::size_t(c)] =
                  flat < n_flat
                      ? vertex_t{vsource::original, flat, subs[std::size_t(c)]}
                      : vertex_t{vsource::created, index_type(flat - n_flat),
                                 subs[std::size_t(c)]};
            }
            _triangle_slots[write] = index_type(slot);
            _row_of[write] = index_type(row);
            _exposed_of_row[std::size_t(row)] = index_type(write);
            ++write;
          }
        },
        tf::checked);

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
    _refine_promoted.clear();
    for (const auto &descriptor : _arr.promoted_descriptors())
      _refine_promoted.push_back(
          {index_type(descriptor.tag), descriptor.object});
    std::sort(_refine_promoted.begin(), _refine_promoted.end());

    // the unified created-points table, materialized once: the world's
    // points first, then everything this arrangement minted
    const auto base_created = _world.n_created_points();
    _created_points.allocate(
        std::size_t(base_created + _arr.n_created()));
    const auto get_base_point =
        [&](std::int16_t tag,
            index_type id) -> tf::point<resolved_int_type, 3> {
      return tag < std::int16_t(0) ? _world.point_of(id, get_mesh_point)
                                   : get_mesh_point(int(tag), id);
    };
    tf::parallel_for_each(
        tf::make_sequence_range(base_created),
        [&](index_type id) {
          _created_points[std::size_t(id)] =
              _world.point_of(id, get_mesh_point);
        },
        tf::checked);
    for (index_type id = base_created;
         id < base_created + _arr.n_created(); ++id)
      _created_points[std::size_t(id)] =
          _arr.resolve_created_point(id, get_base_point, get_mesh_point);
  }

  Policy _policy;
  lattice_type _lattice;
  world_type _world;
  arrangement_type _arr;
  tf::buffer<index_type> _vertex_offsets;
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
};

} // namespace tf
