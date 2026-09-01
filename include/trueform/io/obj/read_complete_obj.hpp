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

#include "obj_face_mode.hpp"
#include "parse_obj_face_triplet.hpp"
#include "parse_obj_scalars.hpp"
#include "parse_obj_token.hpp"
#include "resolve_obj_labels.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/index_map.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../obj_file.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::io::obj {

template <typename Index, typename RealT>
auto read_complete_obj(tf::range<const char *, tf::dynamic_size> input,
                       tf::obj_file<Index, RealT> &output) -> bool {
  static_assert(std::is_floating_point_v<RealT>,
                "obj_reader requires a floating-point Real type");
  output = tf::obj_file<Index, RealT>{};

  tf::buffer<tf::point<RealT, 3>> positions;
  tf::buffer<tf::point<RealT, 2>> textures;
  tf::buffer<tf::point<RealT, 3>> normals;
  tf::buffer<std::array<int, 3>> triplets;
  tf::buffer<Index> face_offsets;
  tf::buffer<Index> face_group_ids;
  tf::buffer<Index> face_object_ids;
  face_offsets.push_back(0);

  std::vector<std::string> group_names;
  std::vector<std::string> object_names;
  auto current_group = Index(-1);
  auto current_object = Index(-1);
  bool group_needs_default = false;
  bool object_needs_default = false;
  auto mode = obj_face_mode::unknown;

  auto *cursor = input.begin();
  const auto *end = input.end();
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
      positions.emplace_back(x, y, z);
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 't' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      cursor += 3;
      RealT u{};
      RealT v{};
      if (!parse_obj_scalars(cursor, end, u, v))
        return false;
      textures.emplace_back(u, v);
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 'n' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      cursor += 3;
      RealT x{};
      RealT y{};
      RealT z{};
      if (!parse_obj_scalars(cursor, end, x, y, z))
        return false;
      normals.emplace_back(x, y, z);
    } else if (first == 'g' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      auto name = parse_obj_token(cursor, end);
      if (name.empty())
        name = std::string_view{"default"};
      group_names.emplace_back(name);
      current_group = static_cast<Index>(group_names.size() - 1);
    } else if (first == 'o' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      auto name = parse_obj_token(cursor, end);
      if (name.empty())
        name = std::string_view{"default"};
      object_names.emplace_back(name);
      current_object = static_cast<Index>(object_names.size() - 1);
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
        if (!parse_obj_face_triplet(cursor, end, mode, triplet))
          return false;
        triplets.push_back(triplet);
        ++corner_count;
      }
      if (corner_count < 3)
        return false;
      face_offsets.push_back(face_offsets.back() +
                             static_cast<Index>(corner_count));
      face_group_ids.push_back(current_group);
      face_object_ids.push_back(current_object);
      group_needs_default |= current_group == Index(-1);
      object_needs_default |= current_object == Index(-1);
    }
    cursor = skip_obj_line(cursor, end);
  }

  if (positions.size() == 0 || triplets.size() == 0)
    return false;

  const auto n_positions = static_cast<int>(positions.size());
  const auto n_textures = static_cast<int>(textures.size());
  const auto n_normals = static_cast<int>(normals.size());
  const bool has_textures =
      mode == obj_face_mode::v_vt || mode == obj_face_mode::v_vt_vn;
  const bool has_normals =
      mode == obj_face_mode::v_vn || mode == obj_face_mode::v_vt_vn;
  auto valid_triplets = tf::make_mapped_range(triplets, [&](const auto &value) {
    if (value[0] < 0 || value[0] >= n_positions)
      return false;
    if (has_textures && (value[1] < 0 || value[1] >= n_textures))
      return false;
    if (has_normals && (value[2] < 0 || value[2] >= n_normals))
      return false;
    return true;
  });
  if (!tf::reduce(
          valid_triplets, [](bool left, bool right) { return left && right; },
          true, tf::checked))
    return false;

  tf::index_map_buffer<Index> index_map;
  tf::make_unique_index_map(triplets, index_map);
  auto &output_points = output.polygons.points_buffer();
  output_points.allocate(index_map.kept_ids().size());
  if (has_normals)
    output.normals.allocate(index_map.kept_ids().size());
  if (has_textures)
    output.textures.allocate(index_map.kept_ids().size());
  tf::parallel_for_each(
      tf::enumerate(index_map.kept_ids()),
      [&](auto pair) {
        auto &&[index, kept] = pair;
        const auto &triplet = triplets[kept];
        output_points[index] = positions[triplet[0]];
        if (has_normals) {
          const auto &normal = normals[triplet[2]];
          output.normals.data_buffer()[3 * index + 0] = normal[0];
          output.normals.data_buffer()[3 * index + 1] = normal[1];
          output.normals.data_buffer()[3 * index + 2] = normal[2];
        }
        if (has_textures) {
          const auto &texture = textures[triplet[1]];
          output.textures.data_buffer()[2 * index + 0] = texture[0];
          output.textures.data_buffer()[2 * index + 1] = texture[1];
        }
      },
      tf::checked);

  auto &output_faces = output.polygons.faces_buffer();
  output_faces.offsets_buffer() = std::move(face_offsets);
  output_faces.data_buffer().allocate(triplets.size());
  tf::parallel_for_each(
      tf::enumerate(index_map.f()),
      [&](auto pair) {
        auto &&[index, value] = pair;
        output_faces.data_buffer()[index] = static_cast<Index>(value);
      },
      tf::checked);

  resolve_obj_labels(face_group_ids, group_names, group_needs_default,
                     output.face_groups, output.group_names);
  resolve_obj_labels(face_object_ids, object_names, object_needs_default,
                     output.face_objects, output.object_names);
  return true;
}

} // namespace tf::io::obj
