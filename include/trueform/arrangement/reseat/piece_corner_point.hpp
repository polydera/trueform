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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "./reseat_pieces.hpp"
#include <cstddef>

namespace tf::arrangement::reseat {

/// Where a carve corner stands: an original at `reader`, a seat in `seats`,
/// a created identity in `created`.
template <typename Index, typename Int, typename Reader, typename Created>
auto piece_corner_point(const piece_corner<Index> &corner, const Reader &reader,
                        const tf::buffer<tf::point<Int, 3>> &seats,
                        const Created &created) -> tf::point<Int, 3> {
  if (corner.tag >= 0) {
    const auto p = reader(int(corner.tag), corner.id);
    return tf::point<Int, 3>{p[0], p[1], p[2]};
  }
  if (corner.tag == seat_corner_tag)
    return seats[std::size_t(corner.id)];
  return created[std::size_t(corner.id)];
}

} // namespace tf::arrangement::reseat
