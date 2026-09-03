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

#include "count_dynamic_obj_records.hpp"
#include "for_each_obj_partition.hpp"
#include "make_obj_line_partitions.hpp"
#include "obj_partition_count.hpp"
#include "parse_dynamic_obj_partition.hpp"
#include "validate_obj_face_corners.hpp"

#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Dims>
auto read_dynamic_obj(tf::range<const char *, tf::dynamic_size> input,
                      tf::points_buffer<RealT, Dims> &points,
                      tf::offset_block_buffer<Index, Index> &faces) -> bool {
  static_assert(Dims == 2 || Dims == 3);

  const auto partition_count = obj_partition_count(input.size());
  auto boundaries =
      make_obj_line_partitions(input.begin(), input.size(), partition_count);
  tf::buffer<obj_dynamic_record_counts> bases;
  bases.allocate_and_initialize(partition_count + 1, {});
  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    bases[partition + 1] =
        count_dynamic_obj_records(input.begin() + boundaries[partition],
                                  input.begin() + boundaries[partition + 1]);
  });
  for (std::size_t partition = 0; partition < partition_count; ++partition) {
    const auto &previous = bases[partition];
    auto &current = bases[partition + 1];
    current.points += previous.points;
    current.faces += previous.faces;
    current.corners += previous.corners;
  }

  const auto totals = bases[partition_count];
  auto &offsets = faces.offsets_buffer();
  auto &data = faces.data_buffer();
  if (offsets.size() == 0)
    offsets.push_back(0);
  const auto point_base = points.size();
  const auto face_base = offsets.size() - 1;
  const auto corner_base = data.size();
  points.reallocate(point_base + totals.points);
  offsets.reallocate(face_base + 1 + totals.faces);
  data.reallocate(corner_base + totals.corners);

  tf::buffer<unsigned char> valid;
  valid.allocate_and_initialize(partition_count, 0);
  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    valid[partition] = static_cast<unsigned char>(parse_dynamic_obj_partition(
        input.begin() + boundaries[partition],
        input.begin() + boundaries[partition + 1], bases[partition],
        bases[partition + 1], point_base, face_base, corner_base, points,
        faces));
  });
  if (std::find(valid.begin(), valid.end(), 0) == valid.end() &&
      validate_obj_face_corners(
          tf::make_range(data.begin() + corner_base, data.end()),
          points.size()))
    return true;
  points.clear();
  faces.clear();
  return false;
}

} // namespace tf::io::obj
