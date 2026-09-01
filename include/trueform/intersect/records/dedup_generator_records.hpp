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
#include "../../core/buffer.hpp"
#include "./tagged_intersection.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <tuple>

namespace tf::intersect {

/// A record's generators: the two tagged faces and the two topological
/// targets. This is the record's whole content once identity is a
/// canonical name — the point id is a function of it.
template <typename Index>
auto generator_key(const tf::intersect::tagged_intersection<Index> &rec) {
  return std::make_tuple(rec.tag, rec.object, rec.tag_other, rec.object_other,
                         rec.target, rec.target_other);
}

/// Sort by generator and collapse repeats. Two records naming the same
/// generators state the same fact, so the survivor loses nothing; the
/// resulting order is canonical, and its leading key is `key()`, so a
/// later grouping pass needs no sort of its own.
template <typename Index>
auto dedup_generator_records(
    tf::buffer<tf::intersect::tagged_intersection<Index>> &records) -> void {
  if (records.size() == 0)
    return;
  const auto by_key = [](const auto &a, const auto &b) {
    return generator_key(a) < generator_key(b);
  };
  // A small batch is sorted below the cost of entering the parallel
  // machinery.
  if (records.size() < 4096)
    std::sort(records.begin(), records.end(), by_key);
  else
    tbb::parallel_sort(records.begin(), records.end(), by_key);
  records.erase_till_end(std::unique(
      records.begin(), records.end(), [](const auto &a, const auto &b) {
        return generator_key(a) == generator_key(b);
      }));
}

} // namespace tf::intersect
