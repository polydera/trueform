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

#include "for_each_obj_partition.hpp"

#include "../../core/buffer.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief Resolves the faces a line partition parsed before its first
/// directive.
///
/// A `g` / `o` directive claims every face from its own position to the next
/// one, so a partition's leading faces belong to the last directive stated
/// before the partition began. The parse pass marks them `-1`, and they are a
/// prefix of the partition because every later face already names a directive
/// of its own partition.
///
/// @param face_bases Per-partition face base, of size `partitions + 1`.
/// @param directive_bases Per-partition directive base, of size
/// `partitions + 1`.
/// @param face_labels The per-face directive ids, resolved in place.
template <typename FaceBases, typename DirectiveBases, typename Index>
auto inherit_obj_partition_labels(FaceBases face_bases,
                                  DirectiveBases directive_bases,
                                  tf::buffer<Index> &face_labels) -> void {
  const auto partition_count = face_bases.size() - 1;
  tf::buffer<Index> inherited;
  inherited.allocate(partition_count);
  auto carried = Index(-1);
  for (std::size_t partition = 0; partition < partition_count; ++partition) {
    inherited[partition] = carried;
    if (directive_bases[partition + 1] > directive_bases[partition])
      carried = static_cast<Index>(directive_bases[partition + 1] - 1);
  }

  for_each_obj_partition(partition_count, [&](std::size_t partition) {
    const auto value = inherited[partition];
    auto *label = face_labels.begin() + face_bases[partition];
    const auto *stop = face_labels.begin() + face_bases[partition + 1];
    while (label < stop && *label == Index(-1))
      *label++ = value;
  });
}

} // namespace tf::io::obj
