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

#include "../core/point.hpp"
#include "../core/range.hpp"
#include "./input_lattice_reader.hpp"

#include <cstddef>

namespace tf::exact {

/// The reader of a record face's corners: an id past its form's own points
/// is a seat, answered from the caller's tail, every other id the one
/// reader's. A read through a form's own faces never names a seat.
template <typename Index, typename RealType, typename Int, typename ApplyToForm>
struct input_lattice_tail_reader {
  tf::exact::input_lattice_reader<Index, RealType, Int, ApplyToForm> reader;
  /// Per tag, its block of seats in `tail_points`.
  tf::range<const Index *, tf::dynamic_size> tail_offsets;
  tf::range<const tf::point<Int, 3> *, tf::dynamic_size> tail_points;

  auto operator()(int tag, Index id) const -> tf::point<Int, 3> {
    const auto n = reader.vertex_offsets[std::size_t(tag) + 1] -
                   reader.vertex_offsets[std::size_t(tag)];
    if (id >= n)
      return tail_points[std::size_t(tail_offsets[std::size_t(tag)] +
                                     (id - n))];
    return reader(tag, id);
  }
};

} // namespace tf::exact
