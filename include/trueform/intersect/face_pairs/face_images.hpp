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
#include "../../core/aabb.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../exact/vertex.hpp"
#include "../classify/face_plane_info.hpp"

#include <cstddef>

namespace tf::intersect {

/// What every pair a face is in reads of it: its corners on the lattice,
/// the exact box they span and its supporting plane — one row per face of
/// the form, by face id. A self search reaches every face that has a
/// neighbour, from every leaf pair its leaf is in, so the images are formed
/// once for the form and every block reads them.
template <typename Index, typename Int> struct face_images {
  tf::offset_block_buffer<Index, tf::exact::vertex<Index, Int>> corners;
  tf::buffer<tf::aabb<Int, 3>> boxes;
  tf::buffer<tf::exact::face_plane<Int>> planes;

  auto face(std::size_t f) const -> tf::exact::vertex_range<Index, Int> {
    const auto &offsets = corners.offsets_buffer();
    return tf::make_range(corners.data_buffer().begin() + offsets[f],
                          offsets[f + 1] - offsets[f]);
  }
};

} // namespace tf::intersect
