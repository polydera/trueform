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
#include "for_each_obj_partition.hpp"
#include "inherit_obj_partition_labels.hpp"
#include "make_obj_line_partitions.hpp"
#include "make_obj_vertex_triplets.hpp"
#include "obj_face_mode.hpp"
#include "obj_partition_count.hpp"
#include "obj_tables.hpp"
#include "parse_complete_obj_partition.hpp"
#include "resolve_obj_labels.hpp"

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../obj_file.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Ngon>
auto read_complete_obj(tf::range<const char *, tf::dynamic_size> input,
                       tf::obj_file<Index, RealT, Ngon> &output) -> bool {
  static_assert(std::is_floating_point_v<RealT>,
                "obj_reader requires a floating-point Real type");
  static_assert(Ngon == tf::dynamic_size || Ngon >= 3,
                "a face needs at least three corners");
  output = tf::obj_file<Index, RealT, Ngon>{};

  const auto partition_count = obj_partition_count(input.size());
  auto boundaries =
      make_obj_line_partitions(input.begin(), input.size(), partition_count);
  tf::buffer<obj_complete_record_counts> bases;
  tf::buffer<obj_face_mode> modes;
  bases.allocate_and_initialize(partition_count + 1, {});
  modes.allocate_and_initialize(partition_count, obj_face_mode::unknown);
  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    bases[partition + 1] = count_complete_obj_records<Ngon>(
        input.begin() + boundaries[partition],
        input.begin() + boundaries[partition + 1], modes[partition]);
  });
  for (std::size_t partition = 0; partition < partition_count; ++partition) {
    const auto &previous = bases[partition];
    auto &current = bases[partition + 1];
    current.positions += previous.positions;
    current.textures += previous.textures;
    current.normals += previous.normals;
    current.faces += previous.faces;
    current.corners += previous.corners;
    current.groups += previous.groups;
    current.objects += previous.objects;
  }

  const auto totals = bases[partition_count];
  if (totals.positions == 0 || totals.corners == 0)
    return false;

  auto mode = obj_face_mode::unknown;
  for (auto partition_mode : modes) {
    if (partition_mode == obj_face_mode::unknown)
      continue;
    if (mode == obj_face_mode::unknown)
      mode = partition_mode;
    else if (mode != partition_mode)
      return false;
  }
  const bool has_textures = obj_face_mode_names_textures(mode);
  const bool has_normals = obj_face_mode_names_normals(mode);

  obj_tables<Index, RealT, Ngon> tables;
  tables.positions.allocate(totals.positions);
  tables.textures.allocate(totals.textures);
  tables.normals.allocate(totals.normals);
  tables.corner_positions.allocate(totals.corners);
  if (has_textures || has_normals)
    tables.corner_attributes.allocate(totals.corners);
  if constexpr (Ngon == tf::dynamic_size) {
    tables.face_offsets.allocate(totals.faces + 1);
    tables.face_offsets[0] = 0;
  }
  tables.face_groups.allocate(totals.faces);
  tables.face_objects.allocate(totals.faces);
  tables.group_names.resize(totals.groups);
  tables.object_names.resize(totals.objects);

  tf::buffer<unsigned char> valid;
  valid.allocate_and_initialize(partition_count, 0);
  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    valid[partition] = static_cast<unsigned char>(parse_complete_obj_partition(
        input.begin() + boundaries[partition],
        input.begin() + boundaries[partition + 1], bases[partition],
        bases[partition + 1], mode, tables));
  });
  if (std::find(valid.begin(), valid.end(), 0) != valid.end())
    return false;

  const auto n_positions = static_cast<int>(totals.positions);
  const auto n_textures = static_cast<int>(totals.textures);
  const auto n_normals = static_cast<int>(totals.normals);
  auto known_positions =
      tf::make_mapped_range(tables.corner_positions, [&](int position) {
        return position >= 0 && position < n_positions;
      });
  auto known_attributes =
      tf::make_mapped_range(tables.corner_attributes, [&](const auto &pair) {
        if (has_textures && (pair[0] < 0 || pair[0] >= n_textures))
          return false;
        if (has_normals && (pair[1] < 0 || pair[1] >= n_normals))
          return false;
        return true;
      });
  const auto both = [](bool left, bool right) { return left && right; };
  if (!tf::reduce(known_positions, both, true, tf::checked) ||
      !tf::reduce(known_attributes, both, true, tf::checked))
    return false;

  tf::buffer<Index> corner_vertices;
  tf::buffer<std::array<int, 3>> vertices;
  make_obj_vertex_triplets(tables.corner_positions, tables.corner_attributes,
                           totals.positions, partition_count, corner_vertices,
                           vertices);
  auto &output_points = output.polygons.points_buffer();
  output_points.allocate(vertices.size());
  if (has_normals)
    output.normals.allocate(vertices.size());
  if (has_textures)
    output.textures.allocate(vertices.size());
  tf::parallel_for_each(
      tf::enumerate(vertices),
      [&](auto pair) {
        auto &&[index, triplet] = pair;
        output_points[index] = tables.positions[triplet[0]];
        if (has_normals) {
          const auto &normal = tables.normals[triplet[2]];
          output.normals.data_buffer()[3 * index + 0] = normal[0];
          output.normals.data_buffer()[3 * index + 1] = normal[1];
          output.normals.data_buffer()[3 * index + 2] = normal[2];
        }
        if (has_textures) {
          const auto &texture = tables.textures[triplet[1]];
          output.textures.data_buffer()[2 * index + 0] = texture[0];
          output.textures.data_buffer()[2 * index + 1] = texture[1];
        }
      },
      tf::checked);

  auto &output_faces = output.polygons.faces_buffer();
  if constexpr (Ngon == tf::dynamic_size)
    output_faces.offsets_buffer() = std::move(tables.face_offsets);
  output_faces.data_buffer() = std::move(corner_vertices);

  auto face_bases = tf::make_mapped_range(
      bases, [](const auto &counts) { return counts.faces; });
  // A file that states no directive of a kind leaves every label of that kind
  // unclaimed, and `resolve_obj_labels` then emits none, so nothing reads what
  // the inheritance would resolve.
  if (totals.groups != 0) {
    inherit_obj_partition_labels(
        face_bases,
        tf::make_mapped_range(bases,
                              [](const auto &counts) { return counts.groups; }),
        tables.face_groups);
    // The faces preceding the file's first directive are a prefix of the face
    // stream, so the first face alone states whether a default label is needed.
    resolve_obj_labels(tables.face_groups, tables.group_names,
                       tables.face_groups[0] == Index(-1), output.face_groups,
                       output.group_names);
  }
  if (totals.objects != 0) {
    inherit_obj_partition_labels(
        face_bases,
        tf::make_mapped_range(
            bases, [](const auto &counts) { return counts.objects; }),
        tables.face_objects);
    resolve_obj_labels(tables.face_objects, tables.object_names,
                       tables.face_objects[0] == Index(-1),
                       output.face_objects, output.object_names);
  }
  return true;
}

} // namespace tf::io::obj
