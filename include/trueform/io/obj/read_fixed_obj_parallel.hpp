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

#include "count_obj_records.hpp"
#include "for_each_obj_partition.hpp"
#include "make_obj_line_partitions.hpp"
#include "obj_partition_count.hpp"
#include "parse_fixed_obj_partition.hpp"
#include "validate_obj_face_corners.hpp"

#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
auto read_fixed_obj_parallel(tf::range<const char *, tf::dynamic_size> input,
                             tf::points_buffer<RealT, Dims> &points,
                             tf::blocked_buffer<Index, Ngon> &faces) -> bool {
  static_assert(Dims == 2 || Dims == 3);
  static_assert(Ngon != tf::dynamic_size);

  const auto partition_count = obj_partition_count(input.size());
  auto boundaries =
      make_obj_line_partitions(input.begin(), input.size(), partition_count);
  tf::buffer<obj_record_counts> counts;
  counts.allocate_and_initialize(partition_count, {});
  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    counts[partition] =
        count_obj_records(input.begin() + boundaries[partition],
                          input.begin() + boundaries[partition + 1]);
  });

  tf::buffer<std::size_t> point_offsets;
  tf::buffer<std::size_t> face_offsets;
  point_offsets.allocate(partition_count + 1);
  face_offsets.allocate(partition_count + 1);
  point_offsets[0] = 0;
  face_offsets[0] = 0;
  for (std::size_t partition = 0; partition < partition_count; ++partition) {
    point_offsets[partition + 1] =
        point_offsets[partition] + counts[partition].points;
    face_offsets[partition + 1] =
        face_offsets[partition] + counts[partition].faces;
  }

  const auto points_base = points.size();
  const auto faces_base = static_cast<std::size_t>(faces.size());
  const auto output_points = points_base + point_offsets.back();
  const auto output_faces = faces_base + face_offsets.back();

  tf::buffer<unsigned char> valid;
  valid.allocate_and_initialize(partition_count, 0);
  points.reserve(output_points);
  faces.reserve(output_faces);
  points.reallocate(output_points);
  faces.reallocate(output_faces);

  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    valid[partition] = static_cast<unsigned char>(parse_fixed_obj_partition(
        input.begin() + boundaries[partition],
        input.begin() + boundaries[partition + 1], points,
        points_base + point_offsets[partition], counts[partition].points, faces,
        faces_base + face_offsets[partition], counts[partition].faces));
  });

  if (std::find(valid.begin(), valid.end(), 0) == valid.end() &&
      validate_obj_face_corners(
          tf::make_range(faces.data_buffer().begin() + faces_base * Ngon,
                         faces.data_buffer().end()),
          points.size()))
    return true;
  points.clear();
  faces.clear();
  return false;
}

} // namespace tf::io::obj
