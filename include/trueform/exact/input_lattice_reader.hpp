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

#include "../core/frame_of.hpp"
#include "../core/point.hpp"
#include "../core/range.hpp"
#include "../core/transformed.hpp"
#include "./tag_of_flat_vertex.hpp"
#include "./vertex_converter.hpp"

#include <cstddef>

namespace tf::exact {

/// The one reader of an original vertex's position, bound to the
/// caller's own forms.
///
/// The mode lives here and nowhere else, and it is per vertex: a vertex the
/// band moved answers its row of the moved table, every other vertex the
/// converter over the input's own coordinate — exactly the tolerance-zero
/// answer. Every consumer that asks `(tag, id)` or a flat id gets the same
/// fact from the same producer, so one vertex answers one position in every
/// tier, and the originals of an output mesh and its created points cannot
/// disagree.
///
/// The real-valued reader is the export's: a moved vertex states its moved
/// lattice point, every other vertex hands back the input's own coordinate
/// untouched, because an export of what the band did not touch is a view of
/// the input and not a round trip through the lattice.
template <typename Index, typename RealType, typename Int, typename ApplyToForm>
struct input_lattice_reader {
  using converter_type = tf::exact::vertex_converter<Int, RealType, 3>;

  tf::range<const tf::point<Int, 3> *, tf::dynamic_size> moved;
  tf::range<const Index *, tf::dynamic_size> moved_tickets;
  tf::range<const Index *, tf::dynamic_size> vertex_offsets;
  converter_type converter;
  ApplyToForm apply_to_form;
  /// How far a vertex was allowed to move, in lattice units. A query
  /// pruned by a box built from the input's own coordinates grows by
  /// this much and by nothing else — the certificate makes it a bound
  /// and not an estimate.
  Int motion_bound = Int(0);

  /// The row of a flat vertex in the moved table, `-1` when the band did
  /// not move it.
  auto moved_row(Index flat) const -> Index {
    return moved_tickets.size() != 0 ? moved_tickets[std::size_t(flat)]
                                     : Index(-1);
  }

  /// The lattice point of a vertex of a form the caller already holds.
  template <typename Points, typename Frame>
  auto point_in(const Points &points, const Frame &frame, Index base,
                Index id) const -> tf::point<Int, 3> {
    const auto row = moved_row(base + id);
    return row != Index(-1)
               ? moved[std::size_t(row)]
               : converter.convert(tf::transformed(points[id], frame));
  }

  /// The same vertex as an export states it.
  template <typename Points, typename Frame>
  auto real_point_in(const Points &points, const Frame &frame, Index base,
                     Index id) const -> tf::point<RealType, 3> {
    const auto row = moved_row(base + id);
    return row != Index(-1)
               ? converter.deconvert(moved[std::size_t(row)])
               : tf::point<RealType, 3>(tf::transformed(points[id], frame));
  }

  auto operator()(int tag, Index id) const -> tf::point<Int, 3> {
    tf::point<Int, 3> out{};
    const auto base = vertex_offsets[std::size_t(tag)];
    apply_to_form(Index(tag), [&](const auto &form) {
      out = point_in(form.points(), tf::frame_of(form), base, id);
    });
    return out;
  }

  auto point_of_flat(Index flat) const -> tf::point<Int, 3> {
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    return (*this)(int(tag), flat - vertex_offsets[std::size_t(tag)]);
  }
};

} // namespace tf::exact
