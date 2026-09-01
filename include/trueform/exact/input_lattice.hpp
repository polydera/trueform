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
#include "./door/place_vertices.hpp"
#include "./door/placement_tables.hpp"
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
/// — the table the door placed every vertex into.
///
/// Each original vertex moves at most the tolerance, to a lattice
/// vertex; the arrangement is then the exact arrangement of the moved
/// mesh, computed at zero. A tolerance of zero is the identity: no
/// table is built, no face is read, no normal is taken, and the view is
/// the plain converter.
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
    _placed.clear();
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
                 face_offsets, _tolerance_int);
    tf::exact::door::place_vertices(tables, _tolerance_int, _placed);
  }

  auto converter() const -> const converter_type & { return _converter; }
  auto n_tags() const -> Index { return _n_tags; }
  /// The door's band in lattice units; zero when no door ran.
  auto tolerance_int() const -> Int { return _tolerance_int; }
  auto vertex_offsets() const -> const tf::buffer<Index> & {
    return _vertex_offsets;
  }
  /// The placed table by flat vertex id — empty when no door ran, which
  /// is how @ref tf::exact::input_lattice_reader knows the mode.
  auto placed_points() const {
    return tf::make_range(_placed.begin(), _placed.end());
  }

  /// The flat name of an original vertex — the index the placed table
  /// and every identity below are keyed by.
  auto flat_vertex(int tag, Index id) const -> Index {
    return _vertex_offsets[std::size_t(tag)] + id;
  }

  /// The pair search's gather. The caller offers the world coordinate it
  /// already holds, so a run with no door pays only the conversion and a
  /// run with one never touches the float at all.
  template <typename P>
  auto point(int tag, Index id, const P &world_point) const
      -> tf::point<Int, 3> {
    return _placed.size() != 0
               ? _placed[std::size_t(flat_vertex(tag, id))]
               : _converter.convert(world_point);
  }

  /// The reader every consumer programs against, bound to the caller's
  /// own form access.
  template <typename ApplyToForm>
  auto reader(ApplyToForm apply_to_form) const
      -> tf::exact::input_lattice_reader<Index, RealType, Int, ApplyToForm> {
    return {placed_points(),
            tf::make_range(_vertex_offsets.begin(), _vertex_offsets.end()),
            _converter, std::move(apply_to_form), _tolerance_int};
  }

private:
  converter_type _converter;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<tf::point<Int, 3>> _placed;
  Index _n_tags = Index(0);
  Int _tolerance_int = Int(0);
};

} // namespace tf::exact
