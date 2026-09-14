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

#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./intercept_ladder.hpp"
#include "./ladder_gap.hpp"
#include "./line_cells.hpp"
#include "./pool_records.hpp"
#include "./wide_to_double.hpp"

#include "../../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef TF_POOL_CENSUS
#include "../../../core/algorithm/reduce.hpp"
#include "../../../core/views/mapped_range.hpp"
#endif

namespace tf::exact::door::pool {

/// One member on its cell's line, before the order exists.
struct intercept_record {
  double height = 0.0;
  double slack = 0.0;
  int name = -1;
  int at = -1;
};

/// One block of cells' worth of ladder, in the block's own slot numbering.
///
/// A cell is an independent piece of the ladder — its members, its order and
/// its cuts are stated by its own line and by nothing outside it — so a
/// block builds a whole ladder of the cells it was handed and the aggregate
/// rebases it. `run_start` holds the block-local slot each run begins at,
/// which the append turns into global positions with one running base.
template <typename Int> struct intercept_block {
  tf::buffer<int> name;
  tf::buffer<int> cell_of_slot;
  tf::buffer<typename exact_lane<Int>::coefficient_type> residual;
  tf::buffer<typename exact_lane<Int>::product_type> reach;
  tf::buffer<double> height;
  tf::buffer<double> slack;
  tf::buffer<int> run_start;
  tf::buffer<intercept_record> record;
  tf::buffer<typename exact_lane<Int>::coefficient_type> scratch_residual;
  tf::buffer<typename exact_lane<Int>::product_type> scratch_reach;
  election_census census;
};

/// The ladder: every member of every cell described by that cell's line,
/// sorted along it, and cut into discovery runs wherever the physical gap
/// exceeds the proposal band.
///
/// The band here is `2T` and not `T`: the certificate still judges at `T`,
/// but a pair whose supports are within `T` of a common plane can stand up
/// to twice that apart on the line, so a `T` cut would refuse pairs before
/// asking. A wider band costs exposure, never correctness.
///
/// The sort is exact and its comparator is filtered: the double heights
/// decide every pair they can bound, and the cross product `A_i B_j`
/// against `A_j B_i` decides the rest. Equal intercepts fall to the
/// canonical name, so the whole order is a pure function of the geometry.
///
/// Only a SPEAKING cell is laddered. A cell with no witnessed member asks
/// its order no question and its names cannot be reached from any other
/// cell, so its arithmetic would have no consumer.
///
/// THE CELL IS THE PARALLEL GRAIN and the aggregation order IS the ladder:
/// a block builds the cells it was handed in its own slot numbering, and
/// appending the blocks in cell order rebases their runs with one running
/// base. Nothing is sorted across cells, because a cell's line is not a
/// coordinate any other cell shares.
template <typename Int>
auto make_intercept_ladder(const pool_names<Int> &names,
                           const line_cells<Int> &cells,
                           const tf::buffer<char> &speaks, Int band,
                           election_census &census,
                           intercept_ladder<Int> &ladder) -> void {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;

  const auto count = names.plane.size();
  ladder.name.clear();
  ladder.cell_of_slot.clear();
  ladder.residual.clear();
  ladder.reach.clear();
  ladder.height.clear();
  ladder.slack.clear();
  ladder.run_offsets.clear();
  ladder.slot_of_name.allocate(count);
  tf::parallel_fill(ladder.slot_of_name, -1);
  ladder.run_offsets.push_back(0);
  if (count == 0)
    return;

  ladder.name.reserve(count);
  ladder.cell_of_slot.reserve(count);
  ladder.residual.reserve(count);
  ladder.reach.reserve(count);
  ladder.height.reserve(count);
  ladder.slack.reserve(count);

  const product_type proposal(product_type(2) * product_type(band));

  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(cells.key.size()), ladder, intercept_block<Int>{},
      [&names, &cells, &speaks, &proposal](auto block,
                                           intercept_block<Int> &local) {
        local.name.clear();
        local.cell_of_slot.clear();
        local.residual.clear();
        local.reach.clear();
        local.height.clear();
        local.slack.clear();
        local.run_start.clear();
        for (const auto c : block) {
          if (!speaks[std::size_t(c)])
            continue;
          const auto &line = cells.line[std::size_t(c)];
          const auto &at = cells.support[std::size_t(c)];
          const auto &square_length = cells.square_length[std::size_t(c)];
          const double along = std::sqrt(wide_to_double(square_length));

          local.record.clear();
          local.scratch_residual.clear();
          local.scratch_reach.clear();
          for (int k = cells.offsets[std::size_t(c)];
               k < cells.offsets[std::size_t(c) + 1]; ++k) {
            const auto name = std::size_t(cells.member[std::size_t(k)]);
            const auto &plane = names.plane[name];
            const coefficient_type sign(cells.sign_of_name[name]);
            const std::array<coefficient_type, 3> normal{
                sign * plane[0], sign * plane[1], sign * plane[2]};
            coefficient_type stated = sign * plane[3];
            for (std::size_t axis = 0; axis < 3; ++axis)
              stated = stated - normal[axis] * coefficient_type(at[axis]);
            const product_type against = exact_dot<Int>(normal, line);
            if (against <= product_type(0))
              continue;
            intercept_record made;
            made.name = int(name);
            made.at = int(local.scratch_residual.size());
            made.height =
                along * (wide_to_double(stated) / wide_to_double(against));
            // three wide-to-double truncations at `2^-52` each (@ref
            // tf::exact::door::pool::wide_to_double), a square root that
            // halves one of them, and two roundings: under `2^-49` relative,
            // and the relative term alone bounds the whole error. Both terms
            // are therefore slack the screen does not need — and a screen
            // that doubts too much only hands the pair to the exact
            // comparison, which is the answer either way.
            made.slack = std::fabs(made.height) * 0x1p-40 + 0x1p-16;
            local.scratch_residual.push_back(stated);
            local.scratch_reach.push_back(against);
            local.record.push_back(made);
          }

          std::sort(local.record.begin(), local.record.end(),
                    [&local](const intercept_record &a,
                             const intercept_record &b) {
                      if (std::fabs(a.height - b.height) > a.slack + b.slack)
                        return a.height < b.height;
                      const product_type left =
                          product_type(
                              local.scratch_residual[std::size_t(a.at)]) *
                          local.scratch_reach[std::size_t(b.at)];
                      const product_type right =
                          product_type(
                              local.scratch_residual[std::size_t(b.at)]) *
                          local.scratch_reach[std::size_t(a.at)];
                      if (left != right)
                        return left < right;
                      return a.name < b.name;
                    });

          for (std::size_t k = 0; k < local.record.size(); ++k) {
            const auto &made = local.record[k];
            const auto slot = int(local.name.size());
            if (k == 0)
              local.run_start.push_back(slot);
            else {
              const auto &earlier = local.record[k - 1];
              const ladder_point<Int> before{
                  local.scratch_residual[std::size_t(earlier.at)],
                  local.scratch_reach[std::size_t(earlier.at)], earlier.height,
                  earlier.slack};
              const ladder_point<Int> here{
                  local.scratch_residual[std::size_t(made.at)],
                  local.scratch_reach[std::size_t(made.at)], made.height,
                  made.slack};
              if (ladder_gap_exceeds<Int>(before, here, square_length,
                                          proposal, local.census))
                local.run_start.push_back(slot);
            }
            local.name.push_back(made.name);
            local.cell_of_slot.push_back(int(c));
            local.residual.push_back(
                local.scratch_residual[std::size_t(made.at)]);
            local.reach.push_back(local.scratch_reach[std::size_t(made.at)]);
            local.height.push_back(made.height);
            local.slack.push_back(made.slack);
          }
        }
      },
      [&census](const intercept_block<Int> &local,
                intercept_ladder<Int> &into) {
        const int base = int(into.name.size());
        for (const auto start : local.run_start)
          if (start + base != 0)
            into.run_offsets.push_back(start + base);
        // the block's slots are already contiguous, so the append is one
        // growth and a copy per carrier, never a growth check per slot
        const auto held = into.name.size();
        const auto added = local.name.size();
        into.name.reallocate(held + added);
        into.cell_of_slot.reallocate(held + added);
        into.residual.reallocate(held + added);
        into.reach.reallocate(held + added);
        into.height.reallocate(held + added);
        into.slack.reallocate(held + added);
        std::copy(local.name.begin(), local.name.end(),
                  into.name.begin() + held);
        std::copy(local.cell_of_slot.begin(), local.cell_of_slot.end(),
                  into.cell_of_slot.begin() + held);
        std::copy(local.residual.begin(), local.residual.end(),
                  into.residual.begin() + held);
        std::copy(local.reach.begin(), local.reach.end(),
                  into.reach.begin() + held);
        std::copy(local.height.begin(), local.height.end(),
                  into.height.begin() + held);
        std::copy(local.slack.begin(), local.slack.end(),
                  into.slack.begin() + held);
        merge_election_census(local.census, census);
      });

  ladder.run_offsets.push_back(int(ladder.name.size()));

  census.runs = std::int64_t(ladder.run_offsets.size()) - 1;

  tf::parallel_for_each(tf::make_sequence_range(ladder.name.size()),
                        [&ladder](std::size_t slot) {
                          ladder.slot_of_name[std::size_t(
                              ladder.name[slot])] = int(slot);
                        },
                        tf::checked);

#ifdef TF_POOL_CENSUS
  census.gap_cuts = census.runs - census.speaking_cells;
  census.longest_run = tf::reduce(
      tf::make_mapped_range(
          tf::make_sequence_range(std::size_t(census.runs)),
          [&ladder](std::size_t r) {
            return std::int64_t(ladder.run_offsets[r + 1] -
                                ladder.run_offsets[r]);
          }),
      [](std::int64_t held, std::int64_t length) {
        return held < length ? length : held;
      },
      std::int64_t(0));
#endif
}

} // namespace tf::exact::door::pool
