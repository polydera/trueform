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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "./name_plane_carriers.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <cstddef>

namespace tf::intersect::graph {

/// THE SEATING LAW, for every tier that admits faces: an entrant names the
/// plane it stands on by the law every cut face does (@ref
/// tf::intersect::graph::name_plane_carriers). A name a standing plane holds
/// seats the entrant on that plane, equal names among the rest are one new
/// plane, and a void name is a plane of its own. The new planes are numbered
/// past the `n_existing` standing ones in the order of their smallest
/// entrant.
///
/// `existing_names[p]` is standing plane `p`'s name. Per entrant,
/// `plane_of` is its plane; per new plane, `new_plane_first` its smallest
/// entrant. Returns the number of new planes.
template <typename Index, typename Int, typename ExistingNames>
auto seat_named_planes(
    const tf::buffer<tf::exact::canonical_plane<Int>> &names,
    const ExistingNames &existing_names, Index n_existing,
    tf::buffer<Index> &plane_of, tf::buffer<Index> &new_plane_first)
    -> Index {
  using name_t = tf::exact::canonical_plane<Int>;
  const auto n_entrants = Index(names.size());
  // the standing planes by name, so a name they hold is one binary search
  tf::buffer<Index> by_name;
  by_name.allocate(std::size_t(n_existing));
  for (Index p = 0; p < n_existing; ++p)
    by_name[std::size_t(p)] = p;
  tbb::parallel_sort(by_name.begin(), by_name.end(), [&](Index a, Index b) {
    return existing_names[std::size_t(a)] < existing_names[std::size_t(b)];
  });
  plane_of.allocate(std::size_t(n_entrants));
  tf::parallel_for_each(
      tf::make_sequence_range(n_entrants),
      [&](Index e) {
        const auto &name = names[std::size_t(e)];
        Index plane = Index(-1);
        if (!is_plane_name_void<Int>(name)) {
          const auto at = std::lower_bound(
              by_name.begin(), by_name.end(), name,
              [&](Index p, const name_t &value) {
                return existing_names[std::size_t(p)] < value;
              });
          if (at != by_name.end() && existing_names[std::size_t(*at)] == name)
            plane = *at;
        }
        plane_of[std::size_t(e)] = plane;
      },
      tf::checked);

  // the rest found planes of their own: equal names one run, a run
  // numbered by its smallest entrant
  tf::buffer<Index> unseated;
  for (Index e = 0; e < n_entrants; ++e)
    if (plane_of[std::size_t(e)] == Index(-1))
      unseated.push_back(e);
  std::sort(unseated.begin(), unseated.end(), [&](Index a, Index b) {
    if (names[std::size_t(a)] != names[std::size_t(b)])
      return names[std::size_t(a)] < names[std::size_t(b)];
    return a < b;
  });
  tf::buffer<Index> run_first;
  tf::buffer<Index> run_of;
  run_of.allocate(unseated.size());
  for (std::size_t i = 0; i < unseated.size(); ++i) {
    const bool joins =
        i != 0 && !is_plane_name_void<Int>(names[std::size_t(unseated[i])]) &&
        names[std::size_t(unseated[i])] == names[std::size_t(unseated[i - 1])];
    if (!joins)
      run_first.push_back(unseated[i]);
    run_of[i] = Index(run_first.size()) - Index(1);
  }
  tf::buffer<Index> order;
  order.allocate(run_first.size());
  for (std::size_t r = 0; r < run_first.size(); ++r)
    order[r] = Index(r);
  std::sort(order.begin(), order.end(), [&](Index a, Index b) {
    return run_first[std::size_t(a)] < run_first[std::size_t(b)];
  });
  tf::buffer<Index> run_rank;
  run_rank.allocate(run_first.size());
  new_plane_first.allocate(run_first.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    run_rank[std::size_t(order[k])] = Index(k);
    new_plane_first[k] = run_first[std::size_t(order[k])];
  }
  for (std::size_t i = 0; i < unseated.size(); ++i)
    plane_of[std::size_t(unseated[i])] =
        n_existing + run_rank[std::size_t(run_of[i])];
  return Index(run_first.size());
}

} // namespace tf::intersect::graph
