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

#include "../core/buffer.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/range.hpp"
#include "./door/commit_target_planes.hpp"
#include "./door/find_rim_targets.hpp"
#include "./door/gather_exact_contacts.hpp"
#include "./door/place_door_vertices.hpp"
#include "./door/placement_census.hpp"
#include "./door/placement_tables.hpp"
#include "./door/pool/pool_input_planes.hpp"
#include "./door/rim_vertices.hpp"
#include "./input_lattice_reader.hpp"
#include "./input_lattice_tail_reader.hpp"
#include "./resolve_int_type.hpp"
#include "./vertex_converter.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <utility>

namespace tf::exact {

/// Where every original vertex of every operand stands on the exact
/// lattice, and the one producer of that fact for the whole pipeline:
/// the converter the whole run shares, the flat vertex space that names
/// a vertex `(tag, id)` by one integer, and — when a tolerance is given
/// — the sparse tables of the vertices the band moved and of the rim
/// vertices it landed on another form's edge.
///
/// THE BAND IS THE SMALLEST ACTIONABLE TOLERANCE CONTAINING THE REQUEST.
/// A lattice resolves nothing below one of its units, so a request below
/// one is honoured AT one — never discarded — and a request wider than the
/// pair search can carry is honoured at that widest band. The band that
/// acted is published as @ref placement_census::effective_tolerance, in the
/// caller's own units, so what was honoured is never silent. Only a
/// tolerance of zero is the identity.
///
/// A TOLERANCE DOES TWO THINGS, AND NOTHING ELSE.
///
/// 1. ALMOST-COPLANAR IS COPLANAR, BY QUANTIZATION. Every exact plane is
///    rounded — its direction onto a fixed chart whose pitch keeps two
///    planes of one cell within a quarter band over the whole lattice, its
///    intercept onto the band — and equal keys are one pool, whatever the
///    forms. The rounding is the whole relation, so it is transitive and
///    never grows from a face to its neighbour; a key one plane holds alone
///    pools nothing. A pool's plane is dense on the lattice, and every
///    vertex of a pooled face stands on it exactly, moved by at most the
///    band. A vertex that cannot stand on every pool around it stays where
///    it is and every pool at it releases its faces there, until nothing
///    more releases, so no face is left half on a plane. The mesh stays
///    cohesive because a vertex moves, never a face.
/// 2. A RIM CLOSES. A vertex on a boundary edge of a form is the only
///    vertex that can leave a result open, so every rim vertex within the
///    band of another form's face lands on the nearest feature of that
///    face, exactly: at a corner it takes the corner's point; on an edge it
///    takes the lattice point nearest its projection and is named on the
///    edge (@ref edge_names), so the exact tier reads it there whatever the
///    rounding left; inside a face it takes a lattice point on the face's
///    committed plane — the pool's, or the densest plane of the face's own
///    direction its corners can follow. One feature per vertex, the
///    nearest; a landing stands on everything the vertex rides and within
///    the band of its input point, or it is refused and counted.
///
/// A MEETING THE INPUT STATES IS KEPT. A rim vertex exactly on another
/// form's face or edge, and one whose RIM edge already crosses another
/// form's edge, is no gap: it is left where it is. A crossing an interior
/// edge of its faces makes is not the rim's and survives its landing. A
/// vertex nothing claims is not placed at all, and every tier reads it
/// exactly as at zero.
///
/// The arrangement is then the exact arrangement of the moved mesh,
/// computed at zero. A tolerance of zero is the identity: nothing is
/// pooled, nothing is discovered, no table is built, and the view is the
/// plain converter.
template <typename Index, typename RealType,
          typename Int = tf::exact::resolve_int_type<tf::none_t, RealType>>
class input_lattice {
public:
  using converter_type = tf::exact::vertex_converter<Int, RealType, 3>;

  /// `apply_to_form(tag, f)` is the caller's own form access; nothing of
  /// it is retained, so this view outlives no operand view and moves
  /// with the graph that owns it.
  template <typename ApplyToForm>
  auto build(converter_type input_converter, const ApplyToForm &apply_to_form,
             Index n_tags, double tolerance) -> void {
    _converter = std::move(input_converter);
    _ticket.clear();
    _moved.clear();
    _edge_names.clear();
    _census = tf::exact::door::placement_census{};
    _vertex_offsets.allocate(std::size_t(n_tags) + 1);
    _vertex_offsets[0] = Index(0);
    tf::buffer<Index> face_offsets;
    face_offsets.allocate(std::size_t(n_tags) + 1);
    face_offsets[0] = Index(0);
    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&, tag](const auto &form) {
        _vertex_offsets[std::size_t(tag) + 1] =
            _vertex_offsets[std::size_t(tag)] + Index(form.points().size());
        face_offsets[std::size_t(tag) + 1] =
            face_offsets[std::size_t(tag)] + Index(form.faces().size());
      });
    _n_tags = n_tags;
    // THE CLAMP: the band that acts is the smallest actionable tolerance
    // CONTAINING the request, and the widest the pair search can carry —
    // it grows one leaf box against another by twice the band, so the
    // widest is a quarter of the lattice's range.
    const Int widest = std::numeric_limits<Int>::max() / Int(4);
    if (tolerance > 0.0) {
      if (tolerance >= double(_converter.coords.deconvert_tolerance(widest)))
        _tolerance_int = widest;
      else {
        const auto raw =
            _converter.coords.convert_tolerance(RealType(tolerance));
        _tolerance_int = raw < Int(1) ? Int(1) : raw;
      }
    } else
      _tolerance_int = Int(0);
    _census.effective_tolerance =
        double(_converter.coords.deconvert_tolerance(_tolerance_int));
    if (_tolerance_int == Int(0))
      return;
    tf::exact::door::placement_tables<Index, Int, RealType> tables;
    tables.build(_converter, apply_to_form, n_tags, _vertex_offsets,
                 face_offsets);
    tf::exact::door::pool::input_pools<Int> pools;
    if (tables.triangles())
      tf::exact::door::pool::pool_input_planes(
          tables, _tolerance_int,
          pools);
#ifdef TF_POOL_CENSUS
    _election = pools.election;
    _certificate = pools.certificate;
#endif
    tf::buffer<char> rim;
    tf::exact::door::mark_rim_vertices(apply_to_form, n_tags, _vertex_offsets,
                                       rim);
    tf::buffer<tf::exact::door::rim_target<Index>> targets;
    tf::exact::door::find_rim_targets(tables, _converter, apply_to_form, n_tags,
                                      _vertex_offsets, face_offsets, rim,
                                      _tolerance_int, targets,
                                      _census.rim_vertices, _census.on_face);
    // a rim vertex one of whose rim edges already crosses another form's
    // edge is a meeting the input states, not a gap, and is left where it is
    if (targets.size() != 0) {
      tf::buffer<char> scan;
      scan.allocate(tables.points.size());
      tf::parallel_fill(scan, char(0));
      for (const auto &t : targets)
        scan[std::size_t(t.flat)] = char(1);
      tf::exact::door::exact_contacts<Index> contacts;
      tf::exact::door::gather_edge_contacts(tables, _converter, apply_to_form,
                                            n_tags, _vertex_offsets,
                                            face_offsets, scan, contacts);
      if (contacts.edges.size() != 0) {
        tf::buffer<char> crossing;
        crossing.allocate(tables.points.size());
        tf::parallel_fill(crossing, char(0));
        for (const auto &e : contacts.edges)
          for (const auto v : e)
            crossing[std::size_t(v)] = char(1);
        std::size_t kept = 0;
        for (std::size_t at = 0; at < targets.size(); ++at)
          if (crossing[std::size_t(targets[at].flat)])
            ++_census.crossing;
          else
            targets[kept++] = targets[at];
        targets.erase_till_end(targets.begin() + std::ptrdiff_t(kept));
      }
    }
    tf::exact::door::committed_planes<Index, Int> committed;
    tf::exact::door::commit_target_planes(tables, pools, targets,
                                          _tolerance_int, committed, _census);
    tf::exact::door::place_door_vertices(tables, committed, targets,
                                         _tolerance_int, _ticket, _moved,
                                         _edge_names, _census);
  }
  auto converter() const -> const converter_type & { return _converter; }
  /// What the band did: the planes its contacts committed, the vertices
  /// that moved, and every refusal. Zero throughout when no band ran.
  auto census() const -> const tf::exact::door::placement_census & {
    return _census;
  }
  auto n_tags() const -> Index { return _n_tags; }
  /// The door's band in lattice units; zero when no door ran.
  auto tolerance_int() const -> Int { return _tolerance_int; }
#ifdef TF_POOL_CENSUS
  /// The partition's own account of what it committed and what it spent.
  /// Nothing below the door reads either census; they are the surface a gate
  /// measures the pass on, so they exist only where the gate asked for them.
  auto election() const -> const tf::exact::door::pool::election_census & {
    return _election;
  }
  auto certificate() const
      -> const tf::exact::door::pool::certificate_census & {
    return _certificate;
  }
#endif
  auto vertex_offsets() const -> const tf::buffer<Index> & {
    return _vertex_offsets;
  }
  /// The vertices the band moved, in flat order — empty when it moved
  /// none, which is the band answering exactly as zero does.
  auto moved_points() const {
    return tf::make_range(_moved.begin(), _moved.end());
  }
  /// Per flat vertex its row in @ref moved_points, `-1` for a vertex the
  /// band did not move; empty when it moved none.
  auto moved_tickets() const {
    return tf::make_range(_ticket.begin(), _ticket.end());
  }

  /// The rim vertices the band landed on another form's edge, one row
  /// `{vertex, face, slot}` per landing — the flat vertex, the flat face,
  /// and the face's edge from corner `slot` to the next — in flat order of
  /// the vertex; empty when no band ran or nothing landed on an edge.
  auto edge_names() const {
    return tf::make_range(_edge_names.begin(), _edge_names.end());
  }
  /// The flat name of an original vertex — the index the moved table
  /// and every identity below are keyed by.
  auto flat_vertex(int tag, Index id) const -> Index {
    return _vertex_offsets[std::size_t(tag)] + id;
  }

  /// The pair search's gather. The caller offers the world coordinate it
  /// already holds, so a vertex the band did not move pays only the
  /// conversion, exactly as at zero.
  template <typename P>
  auto point(int tag, Index id, const P &world_point) const
      -> tf::point<Int, 3> {
    if (_ticket.size() != 0) {
      const auto row = _ticket[std::size_t(flat_vertex(tag, id))];
      if (row != Index(-1))
        return _moved[std::size_t(row)];
    }
    return _converter.convert(world_point);
  }

  /// The reader every consumer programs against, bound to the caller's
  /// own form access.
  template <typename ApplyToForm>
  auto reader(ApplyToForm apply_to_form) const
      -> tf::exact::input_lattice_reader<Index, RealType, Int, ApplyToForm> {
    return {moved_points(),
            moved_tickets(),
            tf::make_range(_vertex_offsets.begin(), _vertex_offsets.end()),
            _converter,
            std::move(apply_to_form),
            _tolerance_int};
  }

  /// @ref reader with the seats the caller states, by block per tag.
  template <typename ApplyToForm>
  auto tail_reader(
      ApplyToForm apply_to_form,
      tf::range<const Index *, tf::dynamic_size> tail_offsets,
      tf::range<const tf::point<Int, 3> *, tf::dynamic_size> tail_points) const
      -> tf::exact::input_lattice_tail_reader<Index, RealType, Int,
                                              ApplyToForm> {
    return {reader(std::move(apply_to_form)), tail_offsets, tail_points};
  }

private:
  converter_type _converter;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<Index> _ticket;
  tf::buffer<tf::point<Int, 3>> _moved;
  tf::buffer<std::array<Index, 3>> _edge_names;
  tf::exact::door::placement_census _census;
  Index _n_tags = Index(0);
  Int _tolerance_int = Int(0);
#ifdef TF_POOL_CENSUS
  tf::exact::door::pool::election_census _election;
  tf::exact::door::pool::certificate_census _certificate;
#endif
};

} // namespace tf::exact
