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

#include "count_complete_obj_records.hpp"
#include "obj_face_mode.hpp"
#include "obj_tables.hpp"
#include "parse_obj_face_triplet.hpp"
#include "parse_obj_scalars.hpp"
#include "parse_obj_token.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/static_size.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace tf::io::obj {

/// @brief Parses one line partition into the slices its bases reserved.
///
/// On success every table position between `base` and `limit` has been written
/// exactly once, so the partitions stay independent; a refusal leaves the
/// slice partly unwritten and the caller reads nothing. A face names the last
/// `g` / `o` of this partition, or `-1` when the partition has not seen one
/// yet. `file_mode` is what the file's first face stated, and a face that
/// references anything else is refused. A stated arity is a contract: a face
/// of any other corner count is refused rather than discovered.
template <typename Index, typename RealT, std::size_t Ngon>
auto parse_complete_obj_partition(const char *cursor, const char *end,
                                  const obj_complete_record_counts &base,
                                  const obj_complete_record_counts &limit,
                                  obj_face_mode file_mode,
                                  obj_tables<Index, RealT, Ngon> &tables)
    -> bool {
  const auto has_attributes = obj_face_mode_names_textures(file_mode) ||
                              obj_face_mode_names_normals(file_mode);
  auto positions = base.positions;
  auto textures = base.textures;
  auto normals = base.normals;
  auto faces = base.faces;
  auto corners = base.corners;
  auto groups = base.groups;
  auto objects = base.objects;
  auto current_group = Index(-1);
  auto current_object = Index(-1);

  while (cursor < end) {
    cursor = skip_obj_whitespace(cursor, end);
    if (cursor >= end)
      break;

    const auto first = cursor[0];
    if (first == 'v' && cursor + 1 < end &&
        (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      RealT x{};
      RealT y{};
      RealT z{};
      if (!parse_obj_scalars(cursor, end, x, y, z))
        return false;
      auto &position = tables.positions[positions++];
      position[0] = x;
      position[1] = y;
      position[2] = z;
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 't' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      cursor += 3;
      RealT u{};
      RealT v{};
      if (!parse_obj_scalars(cursor, end, u, v))
        return false;
      auto &texture = tables.textures[textures++];
      texture[0] = u;
      texture[1] = v;
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 'n' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      cursor += 3;
      RealT x{};
      RealT y{};
      RealT z{};
      if (!parse_obj_scalars(cursor, end, x, y, z))
        return false;
      auto &normal = tables.normals[normals++];
      normal[0] = x;
      normal[1] = y;
      normal[2] = z;
    } else if (first == 'g' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      auto name = parse_obj_token(cursor, end);
      if (name.empty())
        name = std::string_view{"default"};
      tables.group_names[groups] = std::string(name);
      current_group = static_cast<Index>(groups++);
    } else if (first == 'o' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      auto name = parse_obj_token(cursor, end);
      if (name.empty())
        name = std::string_view{"default"};
      tables.object_names[objects] = std::string(name);
      current_object = static_cast<Index>(objects++);
    } else if (first == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      int corner_count = 0;
      while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        cursor = skip_obj_whitespace(cursor, end);
        if (cursor >= end || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '#')
          break;
        std::array<int, 3> triplet{};
        if (!parse_obj_face_triplet(cursor, end, file_mode, triplet))
          return false;
        if (corners == limit.corners)
          return false;
        tables.corner_positions[corners] = triplet[0];
        if (has_attributes)
          tables.corner_attributes[corners] = {triplet[1], triplet[2]};
        ++corners;
        ++corner_count;
      }
      if constexpr (Ngon == tf::dynamic_size) {
        if (corner_count < 3)
          return false;
        tables.face_offsets[faces + 1] = static_cast<Index>(corners);
      } else {
        if (corner_count != int(Ngon))
          return false;
      }
      tables.face_groups[faces] = current_group;
      tables.face_objects[faces] = current_object;
      ++faces;
    }
    cursor = skip_obj_line(cursor, end);
  }
  return positions == limit.positions && textures == limit.textures &&
         normals == limit.normals && faces == limit.faces &&
         corners == limit.corners && groups == limit.groups &&
         objects == limit.objects;
}

} // namespace tf::io::obj
