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
#include "validate_obj_face_corners.hpp"

#include "../../core/blocked_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"

#include <array>
#include <cstddef>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
auto read_fixed_obj(tf::range<const char *, tf::dynamic_size> data,
                    tf::points_buffer<RealT, Dims> &points,
                    tf::blocked_buffer<Index, Ngon> &faces) -> bool {
  static_assert(Dims == 2 || Dims == 3);
  static_assert(Ngon != tf::dynamic_size);

  auto *cursor = data.begin();
  const auto *end = data.end();
  const auto corner_base = faces.size() * Ngon;
  const auto estimate = data.size() / 28;
  points.reserve(points.size() + estimate);
  faces.reserve(faces.size() + estimate);

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
      if (!parse_obj_scalars(cursor, end, x, y, z)) {
        points.clear();
        faces.clear();
        return false;
      }
      if constexpr (Dims == 3)
        points.emplace_back(x, y, z);
      else
        points.emplace_back(x, y);
    } else if (cursor[0] == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      std::array<Index, Ngon> face{};
      std::size_t count = 0;
      while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        cursor = skip_obj_whitespace(cursor, end);
        if (cursor >= end || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '#')
          break;
        int vertex_index = 0;
        if (!parse_obj_face_index(cursor, end, vertex_index) ||
            vertex_index <= 0) {
          points.clear();
          faces.clear();
          return false;
        }
        if (count < Ngon)
          face[count] = static_cast<Index>(vertex_index - 1);
        ++count;
      }
      if (count != Ngon) {
        points.clear();
        faces.clear();
        return false;
      }
      faces.push_back(face);
    }
    cursor = skip_obj_line(cursor, end);
  }
  if (validate_obj_face_corners(
          tf::make_range(faces.data_buffer().begin() + corner_base,
                         faces.data_buffer().end()),
          points.size()))
    return true;
  points.clear();
  faces.clear();
  return false;
}

} // namespace tf::io::obj
