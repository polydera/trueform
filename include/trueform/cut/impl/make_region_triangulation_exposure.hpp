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

#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../face_regions.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto make_region_triangulation_exposure(
    const tf::face_regions<Index, Int> &fr, Index n_tags,
    const tf::buffer<std::array<Index, 2>> &ranges,
    const tf::buffer<char> &reversed,
    const tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    const tf::buffer<tf::intersect::graph::face_descriptor<Index>> &promoted,
    bool aliased, bool &owned,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        &exposed_triangles,
    tf::buffer<tf::intersect::graph::face_descriptor<Index>>
        &exposed_descriptors,
    tf::buffer<std::array<Index, 2>> &exposed_ranges,
    tf::buffer<Index> &face_offsets,
    tf::buffer<Index> &tag_offsets, tf::buffer<Index> &deleted,
    tf::buffer<Index> &deleted_offsets) -> void {
  face_offsets.clear();

  auto region_deleted_offsets = fr.deleted_offsets();
  deleted_offsets.allocate(region_deleted_offsets.size());
  for (std::size_t i = 0; i < region_deleted_offsets.size(); ++i)
    deleted_offsets[i] = region_deleted_offsets[i];
  deleted.allocate(region_deleted_offsets.size()
                       ? std::size_t(region_deleted_offsets.back())
                       : std::size_t(0));
  for (std::size_t tag = 0; tag + 1 < region_deleted_offsets.size();
       ++tag)
    tf::parallel_copy(
        fr.deleted(static_cast<Index>(tag)),
        tf::slice(tf::make_range(deleted),
                  std::size_t(region_deleted_offsets[tag]),
                  std::size_t(region_deleted_offsets[tag + 1])));

  auto descriptors = fr.descriptors();
  auto region_tag_offsets = fr.tag_offsets();
  const std::size_t n_regions = descriptors.size();
  const std::size_t n_loops = ranges.size();
  const Index tag_count = n_tags < 0 ? Index(0) : n_tags;
  tag_offsets.allocate(std::size_t(tag_count) + 1);
  owned = promoted.size() != 0 || aliased;

  if (!owned) {
    exposed_ranges.allocate(n_loops);
    tf::parallel_copy(tf::make_range(ranges),
                      tf::make_range(exposed_ranges));
    const Index n_triangles = static_cast<Index>(triangles.size());
    tag_offsets[0] = 0;
    for (Index tag = 0; tag < tag_count; ++tag) {
      const std::size_t end =
          std::size_t(region_tag_offsets[std::size_t(tag) + 1]);
      tag_offsets[std::size_t(tag) + 1] =
          end < n_loops ? ranges[end][0] : n_triangles;
    }
    tf::intersect::graph::face_descriptor<Index> previous{Index(-1),
                                                          Index(-1)};
    for (std::size_t loop = 0; loop < n_loops; ++loop) {
      if (ranges[loop][0] == ranges[loop][1])
        continue;
      const auto descriptor = descriptors[loop];
      if (descriptor.tag != previous.tag ||
          descriptor.object != previous.object) {
        face_offsets.push_back(ranges[loop][0]);
        previous = descriptor;
      }
    }
    face_offsets.push_back(n_triangles);
    exposed_descriptors.allocate(triangles.size());
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t loop) {
          const auto range = ranges[loop];
          const auto descriptor = descriptors[loop];
          for (Index triangle = range[0]; triangle < range[1]; ++triangle)
            exposed_descriptors[std::size_t(triangle)] = descriptor;
        });
    exposed_triangles.clear();
    return;
  }

  tf::buffer<Index> loop_order;
  loop_order.allocate(n_loops);
  tf::buffer<std::size_t> tag_ends;
  tag_ends.allocate(std::size_t(tag_count));
  {
    std::size_t position = 0;
    std::size_t promoted_index = 0;
    for (Index tag = 0; tag < tag_count; ++tag) {
      for (Index loop = region_tag_offsets[std::size_t(tag)];
           loop < region_tag_offsets[std::size_t(tag) + 1]; ++loop)
        loop_order[position++] = loop;
      while (promoted_index < promoted.size() &&
             promoted[promoted_index].tag == tag) {
        loop_order[position++] =
            static_cast<Index>(n_regions + promoted_index);
        ++promoted_index;
      }
      tag_ends[std::size_t(tag)] = position;
    }
  }

  tf::buffer<Index> loop_offsets;
  loop_offsets.allocate(n_loops + 1);
  Index offset = 0;
  tf::intersect::graph::face_descriptor<Index> previous{Index(-1), Index(-1)};
  for (std::size_t position = 0; position < n_loops; ++position) {
    const std::size_t loop = std::size_t(loop_order[position]);
    const auto descriptor =
        loop < n_regions ? descriptors[loop] : promoted[loop - n_regions];
    const Index count = ranges[loop][1] - ranges[loop][0];
    loop_offsets[position] = offset;
    if (count != 0 && (descriptor.tag != previous.tag ||
                       descriptor.object != previous.object)) {
      face_offsets.push_back(offset);
      previous = descriptor;
    }
    offset += count;
  }
  loop_offsets[n_loops] = offset;
  face_offsets.push_back(offset);
  tag_offsets[0] = 0;
  for (Index tag = 0; tag < tag_count; ++tag)
    tag_offsets[std::size_t(tag) + 1] =
        loop_offsets[tag_ends[std::size_t(tag)]];

  exposed_ranges.allocate(n_loops);
  tf::parallel_for_each(
      tf::make_sequence_range(n_loops), [&](std::size_t position) {
        exposed_ranges[std::size_t(loop_order[position])] = {
            loop_offsets[position], loop_offsets[position + 1]};
      });

  exposed_triangles.allocate(std::size_t(offset));
  exposed_descriptors.allocate(std::size_t(offset));
  tf::parallel_for_each(
      tf::make_sequence_range(n_loops), [&](std::size_t position) {
        const std::size_t loop = std::size_t(loop_order[position]);
        const auto range = ranges[loop];
        if (range[0] == range[1])
          return;
        const auto descriptor =
            loop < n_regions ? descriptors[loop] : promoted[loop - n_regions];
        const bool reverse =
            loop < reversed.size() && bool(reversed[loop]);
        std::size_t write = std::size_t(loop_offsets[position]);
        for (Index triangle = range[0]; triangle < range[1]; ++triangle) {
          const auto &input = triangles[std::size_t(triangle)];
          exposed_triangles[write] = {
              input[0], input[reverse ? 2 : 1], input[reverse ? 1 : 2]};
          exposed_descriptors[write] = descriptor;
          ++write;
        }
      });
}

} // namespace tf::cut
