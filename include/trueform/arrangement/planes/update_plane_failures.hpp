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

#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// Publish the outstanding plane failures after one successful emission.
///
/// All three inputs are ascending-sorted unique sets. `produced` retires
/// standing failures, while `newly_failed` publishes this wave's failures.
/// Invalid inputs leave `current` unchanged.
template <typename Index>
auto update_plane_failures(tf::buffer<Index> &current,
                           const tf::buffer<Index> &produced,
                           const tf::buffer<Index> &newly_failed) -> bool {
  const auto is_sorted_unique = [](const tf::buffer<Index> &ids) {
    for (std::size_t i = 1; i < ids.size(); ++i)
      if (!(ids[i - 1] < ids[i]))
        return false;
    return true;
  };
  if (!is_sorted_unique(current) || !is_sorted_unique(produced) ||
      !is_sorted_unique(newly_failed))
    return false;

  std::size_t produced_at = 0;
  std::size_t newly_failed_at = 0;
  while (produced_at < produced.size() &&
         newly_failed_at < newly_failed.size()) {
    if (produced[produced_at] < newly_failed[newly_failed_at])
      ++produced_at;
    else if (newly_failed[newly_failed_at] < produced[produced_at])
      ++newly_failed_at;
    else
      return false;
  }

  tf::buffer<Index> standing;
  standing.reserve(current.size());
  std::size_t current_at = 0;
  produced_at = 0;
  while (current_at < current.size() && produced_at < produced.size()) {
    if (current[current_at] < produced[produced_at])
      standing.push_back(current[current_at++]);
    else if (produced[produced_at] < current[current_at])
      ++produced_at;
    else {
      ++current_at;
      ++produced_at;
    }
  }
  while (current_at < current.size())
    standing.push_back(current[current_at++]);

  tf::buffer<Index> next;
  next.reserve(standing.size() + newly_failed.size());
  std::size_t standing_at = 0;
  newly_failed_at = 0;
  while (standing_at < standing.size() &&
         newly_failed_at < newly_failed.size()) {
    if (standing[standing_at] < newly_failed[newly_failed_at])
      next.push_back(standing[standing_at++]);
    else if (newly_failed[newly_failed_at] < standing[standing_at])
      next.push_back(newly_failed[newly_failed_at++]);
    else {
      next.push_back(standing[standing_at++]);
      ++newly_failed_at;
    }
  }
  while (standing_at < standing.size())
    next.push_back(standing[standing_at++]);
  while (newly_failed_at < newly_failed.size())
    next.push_back(newly_failed[newly_failed_at++]);

  current = std::move(next);
  return true;
}

} // namespace tf::arrangement
