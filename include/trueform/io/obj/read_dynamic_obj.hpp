/*
 * Copyright (c) 2025 XLAB
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

#include "parse_obj_face_index.hpp"
#include "parse_obj_scalars.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/offset_block_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"

#include <cstddef>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Dims>
auto read_dynamic_obj(tf::range<const char *, tf::dynamic_size> input,
                      tf::points_buffer<RealT, Dims> &points,
                      tf::offset_block_buffer<Index, Index> &faces) -> bool {
  static_assert(Dims == 2 || Dims == 3);

  auto *cursor = input.begin();
  const auto *end = input.end();
  const auto estimate = input.size() / 28;
  points.reserve(estimate);
  auto &offsets = faces.offsets_buffer();
  auto &data = faces.data_buffer();
  if (offsets.size() == 0)
    offsets.push_back(0);
  offsets.reserve(estimate);
  data.reserve(estimate * 3);

  auto current_data_offset = offsets[offsets.size() - 1];
  while (cursor < end) {
    cursor = skip_obj_whitespace(cursor, end);
    if (cursor >= end)
      break;

    if (cursor[0] == 'v' && cursor + 1 < end &&
        (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      RealT x{};
      RealT y{};
      RealT z{};
      if (!parse_obj_scalars(cursor, end, x, y, z))
        return false;
      if constexpr (Dims == 3)
        points.emplace_back(x, y, z);
      else
        points.emplace_back(x, y);
    } else if (cursor[0] == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      std::size_t indices_in_face = 0;
      while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        cursor = skip_obj_whitespace(cursor, end);
        if (cursor >= end || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '#')
          break;
        int vertex_index = 0;
        if (!parse_obj_face_index(cursor, end, vertex_index) ||
            vertex_index <= 0)
          return false;
        data.push_back(static_cast<Index>(vertex_index - 1));
        ++indices_in_face;
      }
      if (indices_in_face >= 3) {
        current_data_offset += static_cast<Index>(indices_in_face);
        offsets.push_back(current_data_offset);
      }
    }
    cursor = skip_obj_line(cursor, end);
  }
  return true;
}

} // namespace tf::io::obj
