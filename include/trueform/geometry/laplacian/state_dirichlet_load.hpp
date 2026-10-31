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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "./laplacian_form.hpp"
#include <cstddef>

namespace tf::geometry {

/// The load the fixed rows state into the free ones, `rhs = -A_FK x_K`, for
/// `r` fields at once.
///
/// `held` spans the whole system, entry `(row, c)` at `row r + c`, and is
/// read on the fixed rows only; `rhs` receives `r` lanes per free row.
template <typename Index>
auto state_dirichlet_load(const tf::geometry::laplacian_form<Index> &form,
                          const tf::buffer<double> &held, std::size_t r,
                          tf::buffer<double> &rhs) -> void {
  const std::size_t n = form.free.size();
  rhs.allocate(n * r);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i)
          for (std::size_t c = 0; c < r; ++c) {
            double value = 0.0;
            for (Index at = form.fixed_offsets[i];
                 at < form.fixed_offsets[i + 1]; ++at)
              value -=
                  form.fixed_values[std::size_t(at)] *
                  held[std::size_t(form.fixed_columns[std::size_t(at)]) * r +
                       c];
            rhs[i * r + c] = value;
          }
      },
      tf::checked);
}

} // namespace tf::geometry
