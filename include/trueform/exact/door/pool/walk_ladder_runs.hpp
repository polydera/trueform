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

#include "./certificate_census.hpp"
#include "./certificate_lane.hpp"
#include "./certify_name.hpp"
#include "./exact_lane.hpp"
#include "./intercept_ladder.hpp"
#include "./ladder_gap.hpp"
#include "./line_cells.hpp"
#include "./make_line_plane.hpp"
#include "./pool_records.hpp"
#include "./reachable_steps.hpp"
#include "./tally_census.hpp"

#include "../../../core/algorithm/block_reduce.hpp"
#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../meta.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::exact::door::pool {

/// What a block of cells tallied while it walked them.
struct walk_tally {
  election_census census;
  certificate_census certificates;
};

/// A block's scratch: the claim marks of the core it is certifying and that
/// core's own offsets, in slot order and sorted.
template <typename Int> struct walk_block {
  tf::buffer<char> carried;
  tf::buffer<typename tf::exact::meta<Int>::T2> offset, sorted;
  walk_tally tally;
};

/// THE WALK: ONE PASS over a run's sorted intercepts, every entry getting
/// exactly one look and at most one certificate. The members of a cell
/// already lie in order along its one line, so the nearest member of any
/// entry is the next one. GATHER forward from the first unconsumed entry
/// while the next stands within the proposal band of it. COMMIT the plane of
/// the gathered core's MEDIAN member. CERTIFY the core against it, once
/// each. EXTEND past the core while certificates keep succeeding.
///
/// THE ONE-T LAW IS INPUT-RELATIVE: a vertex's whole constraint is against
/// its own input position, and no relation between the planes this tier
/// commits is required or checked. So a run consumes its gathered interval
/// whatever the certificates said — a member that refuses stays where the
/// input put it, never re-anchors, and is never asked a second question.
/// Nothing is re-scanned, so the cost is linear in the entries.
///
/// The one entry a run does not consume is the first EXTENSION refusal: it
/// was never inside the gathered interval, so it becomes the next run's
/// origin.
///
/// THE MEDIAN COMMITS. A core spanning at most 2T from its first entry puts
/// every member within about T of its middle one, where the first member's
/// plane would leave the far end at 2T. The even count takes the lower of
/// the two middles, so the choice is a function of the geometry alone.
///
/// THE BAND IS 2T AND THE COMMITMENT IS T. The ladder distance is measured
/// AT THE CELL'S ELECTED SUPPORT POINT, so 2T bounds certifiability only for
/// members parallel to the median. For a tilted member it is a proposal and
/// not a theorem, and the walk cuts there anyway: a range honest about tilt
/// would grow with the support's extent and bound nothing. Admission is
/// always the certificate at T.
///
/// THE PLANE NEVER MOVES DURING AN EXTENSION, so a member cannot carry the
/// run further than its own certificate reaches.
///
/// NO PARTNER, NO MOTION. A name commits only where it welds with another —
/// an entry the band reaches nobody from, or whose reached partners all
/// refuse, keeps the plane it arrived with and moves no vertex.
///
/// A name no face witnesses cannot originate a run, and a core whose median
/// witnesses no wholly realizable face commits nothing: every name that
/// face's corners target has to certify onto the plane. That is what keeps a
/// curved surface from welding — its names certify onto their neighbours one
/// at a time and no face of it is ever realizable.
template <typename Int>
auto walk_ladder_runs(const pool_names<Int> &names,
                      const line_cells<Int> &cells,
                      const intercept_ladder<Int> &ladder, Int tolerance,
                      const tf::buffer<int> &witness_offsets,
                      tf::buffer<certificate_lane<Int>> &lane,
                      certificate_census &certificates, election_census &census,
                      tf::buffer<int> &owner_of_name) -> void {
  using product_type = typename exact_lane<Int>::product_type;

  owner_of_name.allocate(names.plane.size());
  tf::parallel_fill(owner_of_name, -1);
  if (ladder.run_offsets.size() < 2)
    return;

  // the runs of one cell are contiguous, and a cell's line is what a
  // certificate frame is built for, so the cell is the grain
  const auto runs = ladder.run_offsets.size() - 1;
  tf::buffer<int> group;
  for (std::size_t run = 0; run < runs; ++run) {
    const int from = ladder.run_offsets[run];
    if (from == ladder.run_offsets[run + 1])
      continue;
    const int cell = ladder.cell_of_slot[std::size_t(from)];
    if (group.size() == 0 ||
        ladder.cell_of_slot[std::size_t(
            ladder.run_offsets[std::size_t(group[group.size() - 1])])] != cell)
      group.push_back(int(run));
  }
  if (group.size() == 0)
    return;
  group.push_back(int(runs));

  const product_type proposal(product_type(2) * product_type(tolerance));
  walk_tally whole;

  tf::blocked_reduce(
      tf::make_sequence_range(group.size() - 1), whole, walk_block<Int>{},
      [&](auto block, walk_block<Int> &local) {
        for (const auto at : block) {
          const int first = group[std::size_t(at)];
          const int last = group[std::size_t(at) + 1];
          // every run of the group stands on the one cell the group is, so
          // its line, its frame and its reachable-offset set are stated once
          const auto cell = std::size_t(
              ladder.cell_of_slot[std::size_t(
                  ladder.run_offsets[std::size_t(first)])]);
          const auto &square_length = cells.square_length[cell];
          const auto &elected = names.plane[std::size_t(cells.elected[cell])];
          auto &speaks = lane[cell];
          const auto steps = make_reachable_steps<Int>(elected, tolerance);
          for (int run = first; run < last; ++run) {
            const int from = ladder.run_offsets[std::size_t(run)];
            const int to = ladder.run_offsets[std::size_t(run) + 1];
            if (from == to)
              continue;

            bool anchored = false;

            for (int slot = from; slot < to;) {
              const auto name = std::size_t(ladder.name[std::size_t(slot)]);
              if (witness_offsets[name] == witness_offsets[name + 1]) {
                tally_census(local.tally.census.anchors_without_witness);
                ++slot;
                continue;
              }
              tally_census(local.tally.census.anchors);

              // GATHER: forward from this entry while the next stands inside
              // the proposal band of it
              int high = slot;
              for (int ahead = slot + 1; ahead < to; ++ahead) {
                if (ladder_gap_exceeds<Int>(ladder_point_at(ladder, slot),
                                            ladder_point_at(ladder, ahead),
                                            square_length, proposal,
                                            local.tally.census))
                  break;
                high = ahead;
              }
              if (high == slot) {
                tally_census(local.tally.census.partnerless_anchors);
                ++slot;
                continue;
              }

              // A member is ADMISSIBLE when it witnesses a face whose every
              // named target stands within the proposal band of its own
              // offset — a fact about offsets and not a certificate, so
              // choosing the plane costs no certificates at all. A target is
              // judged by REACH and not by membership: a witness names the
              // planes its corners aim at, and those may sit outside the
              // interval this run happened to gather.
              const auto offset_of_name = [&](int wanted) {
                return make_line_plane<Int>(
                    elected, names.support_point[std::size_t(wanted)])[3];
              };
              // Demanding that a single lattice step land exactly on the
              // target's offset would retire every curved surface whose
              // facets sit well inside the band; that question belongs to the
              // election below, asked of the candidates.
              const auto target_stands = [&](int wanted, product_type from) {
                const auto gap = product_type(offset_of_name(wanted)) - from;
                return gap * gap <= proposal * proposal * square_length;
              };
              const auto admissible = [&](int at2) {
                const auto held = std::size_t(ladder.name[std::size_t(at2)]);
                const auto here = product_type(offset_of_name(int(held)));
                for (int w = witness_offsets[held];
                     w < witness_offsets[held + 1]; ++w) {
                  bool whole = true;
                  for (const auto stated : names.witness[std::size_t(w)].name)
                    if (stated >= 0 && !target_stands(stated, here))
                      whole = false;
                  if (whole)
                    return true;
                }
                return false;
              };
              const auto middle = slot + (high - slot) / 2;
              // THE REACHABLE-SET ELECTION: the admissible candidate whose
              // own offset the most members can be reached from by one
              // lattice step, ties broken by centrality and then by the lower
              // index. Reachability is arithmetic and the certificate still
              // owns every commitment.
              //
              // IT IS COUNTED FROM THE SET, NOT FROM THE MEMBERS. The
              // reachable offsets are few and a core can be thousands, so a
              // candidate's count is one lookup per reachable step into the
              // core's own sorted offsets. Where the ball was too large to
              // enumerate, every offset is reachable, every candidate carries
              // the whole core, and the order falls back to centrality with
              // no offsets to build.
              const auto span = std::size_t(high - slot + 1);
              if (steps.enumerated) {
                local.offset.allocate(span);
                for (int at2 = slot; at2 <= high; ++at2)
                  local.offset[std::size_t(at2 - slot)] =
                      offset_of_name(ladder.name[std::size_t(at2)]);
                local.sorted.allocate(span);
                for (std::size_t k = 0; k < span; ++k)
                  local.sorted[k] = local.offset[k];
                std::sort(local.sorted.begin(), local.sorted.end());
              }

              int elect = -1, best = -1, closest = 0;
              for (int at2 = slot; at2 <= high; ++at2) {
                if (!admissible(at2))
                  continue;
                int carried_by = int(span);
                if (steps.enumerated) {
                  const auto here = local.offset[std::size_t(at2 - slot)];
                  carried_by = 0;
                  for (const auto step : steps.increment) {
                    const auto wanted = here + step;
                    const auto from = std::lower_bound(
                        local.sorted.begin(), local.sorted.end(), wanted);
                    const auto to = std::upper_bound(
                        local.sorted.begin(), local.sorted.end(), wanted);
                    carried_by += int(to - from);
                  }
                }
                const int apart =
                    at2 > middle ? at2 - middle : middle - at2;
                if (carried_by > best ||
                    (carried_by == best && apart < closest)) {
                  best = carried_by;
                  closest = apart;
                  elect = at2;
                }
              }
              if (elect < 0) {
                // no member of this core witnesses a face the core could
                // realize, so there was never a plane to weld onto and the
                // interval is consumed with no certificate spent
                tally_census(local.tally.census.cores_without_admissible);
                slot = high + 1;
                continue;
              }
              const auto centre = std::size_t(ladder.name[std::size_t(elect)]);
              const auto plane =
                  make_line_plane<Int>(elected, names.support_point[centre]);

              // CERTIFY the core, once each
              local.carried.allocate(std::size_t(high - slot + 1));
              for (int at2 = slot; at2 <= high; ++at2) {
                const auto held = std::size_t(ladder.name[std::size_t(at2)]);
                tally_census(local.tally.census.certificate_queries);
                if (names.support_offsets[held] ==
                    names.support_offsets[held + 1])
                  tally_census(local.tally.census.empty_supports);
                local.carried[std::size_t(at2 - slot)] =
                    char(certify_name<Int>(names, int(held), plane, tolerance,
                                           speaks, local.tally.certificates));
              }

              const auto committed = product_type(offset_of_name(int(centre)));
              const auto certified_name = [&](int wanted) {
                const int at2 = ladder.slot_of_name[std::size_t(wanted)];
                if (at2 >= slot && at2 <= high)
                  return local.carried[std::size_t(at2 - slot)] != 0;
                // a target this run never gathered was never asked, so what
                // stands for it is the same reach the election used
                return target_stands(wanted, committed);
              };
              bool realizes = false;
              for (int w = witness_offsets[centre];
                   w < witness_offsets[centre + 1] && !realizes; ++w) {
                bool whole = true;
                for (const auto stated : names.witness[std::size_t(w)].name)
                  if (stated >= 0 && !certified_name(stated))
                    whole = false;
                realizes = whole;
              }
              if (!realizes) {
                tally_census(local.tally.census.anchors_realizing_nothing);
                slot = high + 1;
                continue;
              }

              // EXTEND against the SAME committed plane; the first refusal
              // is not consumed and becomes the next run's origin
              int next = high + 1;
              for (; next < to; ++next) {
                const auto held = std::size_t(ladder.name[std::size_t(next)]);
                tally_census(local.tally.census.certificate_queries);
                if (names.support_offsets[held] ==
                    names.support_offsets[held + 1])
                  tally_census(local.tally.census.empty_supports);
                if (!certify_name<Int>(names, int(held), plane, tolerance,
                                       speaks, local.tally.certificates))
                  break;
                tally_census(local.tally.census.extended_members);
              }

              // NO PARTNER, NO MOTION: a name that welded with nobody is not
              // committed at all
              int joined = next - (high + 1);
              for (int at2 = slot; at2 <= high; ++at2)
                joined += local.carried[std::size_t(at2 - slot)] ? 1 : 0;
              if (joined < 2) {
                tally_census(local.tally.census.partnerless_anchors);
                slot = next;
                continue;
              }

              for (int at2 = slot; at2 <= high; ++at2)
                if (local.carried[std::size_t(at2 - slot)])
                  owner_of_name[std::size_t(ladder.name[std::size_t(at2)])] =
                      int(centre);
                else
                  tally_census(local.tally.census.consumed_refusals);
              for (int at2 = high + 1; at2 < next; ++at2)
                owner_of_name[std::size_t(ladder.name[std::size_t(at2)])] =
                    int(centre);

              anchored = true;
              slot = next;
            }
            if (!anchored)
              tally_census(local.tally.census.runs_without_anchor);
          }
        }
      },
      [](const walk_block<Int> &local, walk_tally &into) {
        merge_election_census(local.tally.census, into.census);
        merge_certificate_census(local.tally.certificates, into.certificates);
      },
      tf::checked);

  merge_election_census(whole.census, census);
  merge_certificate_census(whole.certificates, certificates);
}

} // namespace tf::exact::door::pool
