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
#include "./meet_tet_subsegment.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Read the row `row` of the complex `dt` against the cells `star` around its
/// first end, reached at the cell `hint`, which the row keeps as its next
/// hint: present when its second end is a corner of the star; a missing row
/// reads what it meets along the cells it pierces (@ref meet_tet_subsegment).
template <typename DT, typename Star, typename Row, typename Index,
          typename Query>
auto read_tet_subsegment(const DT &dt, const Star &star, Row &row, Index hint,
                         Query &query) -> void {
  const auto cells = dt.tets();
  query.on = Index(-1);
  query.encroacher = Index(-1);
  query.cosine = 0;
  query.hint = hint;
  row.hint = hint;
  query.present = false;
  query.encroached = false;
  for (auto at = star.begin(); at != star.end() && !query.present; ++at)
    for (auto corner : cells[std::size_t(*at)])
      query.present = query.present || corner == row.b;
  if (query.present)
    return;
  meet_tet_subsegment(dt, star, row, query);
}

} // namespace tf::topology::cdt::dt3
