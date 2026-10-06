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
#include "../../core/range.hpp"
#include "./hole_segments_meet.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::fill {

/// One cycle edge as the span it occupies on the sweep axis, which is both
/// the key the sweep sorts by and the end the active list retires on.
template <typename Index, typename Int> struct hole_cycle_span {
  Int lo;
  Int hi;
  Index edge;
};

/// The scratch one simplicity sweep walks on, reused across the rims a
/// worker takes.
template <typename Index, typename Int> struct hole_cycle_sweep {
  tf::buffer<tf::fill::hole_cycle_span<Index, Int>> spans;
  tf::buffer<tf::fill::hole_cycle_span<Index, Int>> active;
};

/// Whether a closed cycle of distinct lattice points is simple: its edges
/// meet only where consecutive ones share their endpoint.
///
/// `offending` takes the lowest edge position of any pair that meets
/// otherwise. The pairs are found by one sweep along the cycle's widest
/// axis, so the quadratic comparison never happens on a rim whose edges are
/// spread out.
template <typename Iterator, typename Index, typename Int>
auto hole_cycle_is_simple(const tf::range<Iterator, tf::dynamic_size> &cycle,
                          tf::fill::hole_cycle_sweep<Index, Int> &sweep,
                          Index &offending) -> bool {
  const Index n = Index(cycle.size());
  const auto next = [n](Index k) { return Index(k + 1 == n ? 0 : k + 1); };

  int sweep_axis = 0;
  Int widest = 0;
  for (int axis = 0; axis < 3; ++axis) {
    Int lo = cycle[0][axis], hi = cycle[0][axis];
    for (Index k = 1; k < n; ++k) {
      lo = std::min(lo, cycle[std::size_t(k)][axis]);
      hi = std::max(hi, cycle[std::size_t(k)][axis]);
    }
    if (hi - lo > widest) {
      widest = hi - lo;
      sweep_axis = axis;
    }
  }

  sweep.spans.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k) {
    const Int a = cycle[std::size_t(k)][sweep_axis];
    const Int b = cycle[std::size_t(next(k))][sweep_axis];
    sweep.spans[std::size_t(k)] = {std::min(a, b), std::max(a, b), k};
  }
  std::sort(sweep.spans.begin(), sweep.spans.end(),
            [](const tf::fill::hole_cycle_span<Index, Int> &a,
               const tf::fill::hole_cycle_span<Index, Int> &b) {
              return a.lo != b.lo ? a.lo < b.lo : a.edge < b.edge;
            });

  sweep.active.clear();
  offending = Index(-1);
  for (const auto &span : sweep.spans) {
    std::size_t kept = 0;
    for (std::size_t i = 0; i < sweep.active.size(); ++i)
      if (sweep.active[i].hi >= span.lo)
        sweep.active[kept++] = sweep.active[i];
    sweep.active.erase_till_end(sweep.active.begin() + kept);

    for (const auto &other : sweep.active) {
      const Index i = std::min(span.edge, other.edge);
      const Index j = std::max(span.edge, other.edge);
      bool meets = false;
      if (j == i + 1)
        meets = tf::fill::hole_segments_overrun(
            cycle[std::size_t(j)], cycle[std::size_t(i)],
            cycle[std::size_t(next(j))]);
      else if (i == 0 && j == n - 1)
        meets = tf::fill::hole_segments_overrun(cycle[0], cycle[1],
                                                cycle[std::size_t(j)]);
      else
        meets = tf::fill::hole_segments_meet(
            cycle[std::size_t(i)], cycle[std::size_t(next(i))],
            cycle[std::size_t(j)], cycle[std::size_t(next(j))]);
      if (meets && (offending == Index(-1) || i < offending))
        offending = i;
    }
    sweep.active.push_back(span);
  }
  return offending == Index(-1);
}

} // namespace tf::fill
