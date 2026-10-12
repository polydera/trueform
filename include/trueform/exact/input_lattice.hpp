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
#include "../core/range.hpp"
#include "./door/gather_vertex_contacts.hpp"
#include "./door/hold_exact_contacts.hpp"
#include "./door/placement_census.hpp"
#include "./door/placement_tables.hpp"
#include "./door/pool/discovery_resolution.hpp"
#include "./door/pool/pool_input_planes.hpp"
#include "./input_lattice_reader.hpp"
#include "./resolve_int_type.hpp"
#include "./vertex_converter.hpp"

#include <cstddef>
#include <limits>
#include <utility>

namespace tf::exact {

/// Where every original vertex of every operand stands on the exact
/// lattice, and the one producer of that fact for the whole pipeline:
/// the converter the whole run shares, the flat vertex space that names
/// a vertex `(tag, id)` by one integer, and — when a tolerance is given
/// — the sparse table of the vertices the band moved.
///
/// A TOLERANCE PROMISES EXACTLY TWO THINGS, through one shared object, the
/// COMMITTED PLANE, so both hold transitively:
///
/// 1. ALMOST-COPLANAR IS COPLANAR. Planes that are almost one plane pull
///    together into one committed plane — faces whose directions agree pool,
///    and a face a vertex of another form stands within the band of gathers
///    the faces of that vertex lying within the band of it.
/// 2. ALMOST-ON-PLANE IS ON-PLANE. A vertex within the band of a committed
///    plane is dropped onto it exactly — where the plane settled, not where
///    its face began.
///
/// FIRST THE PLANES PULL: a face commits to a plane only if its own corners
/// move within the band onto it; a plane they cannot follow is not changed
/// and stays the face's exact self. THEN THE VERTICES DROP into the
/// committed planes, bounded by the same certificate. The band bounds both
/// motions, and there is no special case: a vertex no committed plane
/// claims is not placed at all, and every tier reads it exactly as at zero.
/// A plane that could not commit and a vertex that could not drop are
/// counted in @ref census — the law refusing, visibly.
///
/// AN EXISTING IDENTIFICATION OUTRANKS A NEW ONE. Every exact contact the
/// input states between two forms survives the placement, in every class:
/// a vertex whose new pull would undo one refuses the pull and keeps its
/// input point. An add forgone is legal; a removal never is.
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
    _tolerance_int =
        tolerance > 0.0
            ? _converter.coords.convert_tolerance(RealType(tolerance))
            : Int(0);
    // The pair search grows one leaf box against another by twice the
    // band, so the doubled band is a lattice value like any other.
    const Int widest = std::numeric_limits<Int>::max() / Int(4);
    if (_tolerance_int > widest)
      _tolerance_int = widest;
    if (_tolerance_int <= Int(0)) {
      _tolerance_int = Int(0);
      return;
    }
    tf::exact::door::placement_tables<Index, Int, RealType> tables;
    tables.build(_converter, apply_to_form, n_tags, _vertex_offsets,
                 face_offsets);
    tf::exact::door::gather_vertex_contacts(tables, _converter, apply_to_form,
                                            n_tags, _vertex_offsets,
                                            face_offsets, _tolerance_int);
    // The partition states which exact planes the input's faces give up for
    // which others before a single vertex is placed, so a contact against a
    // pooled face lands on the plane the face will stand on.
    tf::exact::door::pool::input_pools<Int> pools;
    if (tables.triangles())
      tf::exact::door::pool::pool_input_planes(
          tables, _tolerance_int,
          tf::exact::door::pool::pool_discovery_resolution<Int>(_tolerance_int),
          pools);
#ifdef TF_POOL_CENSUS
    _election = pools.election;
    _certificate = pools.certificate;
#endif
    tf::exact::door::hold_exact_contacts(
        tables, pools, _converter, apply_to_form, n_tags, _vertex_offsets,
        face_offsets, _tolerance_int, _ticket, _moved, _census);
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

private:
  converter_type _converter;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<Index> _ticket;
  tf::buffer<tf::point<Int, 3>> _moved;
  tf::exact::door::placement_census _census;
  Index _n_tags = Index(0);
  Int _tolerance_int = Int(0);
#ifdef TF_POOL_CENSUS
  tf::exact::door::pool::election_census _election;
  tf::exact::door::pool::certificate_census _certificate;
#endif
};

} // namespace tf::exact
