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
#include "uniform_draw.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/cdt/dt3/admitted_tet_edge.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/compact_tet_dead_rows.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_segments.hpp>
#include <trueform/topology/cdt/dt3/recovery/regions/run_tet_region_round.hpp>
#include <trueform/topology/cdt/dt3/recovery/regions/tet_round_workspace.hpp>
#include <trueform/topology/cdt/dt3/recovery/regions/tet_row_admits.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_refusing_cut.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_recovery.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_status.hpp>
#include <trueform/topology/cdt/dt3/tet_execution_tuning.hpp>
#include <trueform/topology/cdt/dt3/tet_flip_scratch.hpp>
#include <trueform/topology/cdt/dt3/tet_owner_complex.hpp>
#include <trueform/topology/cdt/dt3/tet_round_state.hpp>
#include <trueform/topology/cdt/dt3/tet_row_pool.hpp>
#include <trueform/topology/cdt/dt3/tet_site_position.hpp>
#include <trueform/topology/cdt/dt3/tet_state.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tbb/task_arena.h>
#include <utility>
#include <vector>

namespace {

namespace adm_dt3 = tf::topology::cdt::dt3;
using adm_coord = std::int64_t;
using adm_owner_t =
    adm_dt3::tetrahedralization_owner<int, adm_coord, tf::exact::int32>;
using adm_complex = adm_dt3::tet_owner_complex<false, adm_owner_t, tf::none_t>;
using adm_recovery = adm_dt3::tet_segment_recovery<int, adm_coord>;
using adm_status = adm_dt3::tet_segment_status;

/// A cloud on the int32 lattice, stored scaled by 2^scale.
auto adm_cloud(std::uint32_t seed, std::size_t n, int span, int scale)
    -> std::vector<adm_coord> {
  std::mt19937 rng(seed);
  std::vector<adm_coord> flat;
  for (std::size_t i = 0; i < 3 * n; ++i)
    flat.push_back(adm_coord(tf::test::uniform_int(rng, -span, span)) *
                   (adm_coord(1) << scale));
  return flat;
}

auto adm_build(adm_owner_t &owner, const std::vector<adm_coord> &flat) -> bool {
  return adm_dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat)));
}

/// The patch of every site under a partition into `parts` by site ranges,
/// the first `strays` sites belonging to no region — the lowest ids, which
/// a cell names as its first corner.
auto adm_site_patches(const adm_owner_t &owner, std::size_t parts,
                      std::size_t strays) -> tf::buffer<int> {
  const std::size_t sites = adm_dt3::tet_site_count(owner);
  const std::size_t owned = sites > strays ? sites - strays : 0;
  const std::size_t each = owned / parts + 1;
  tf::buffer<int> patches;
  patches.allocate(sites);
  for (std::size_t site = 0; site < sites; ++site)
    patches[site] = site < strays ? -1 : int((site - strays) / each);
  return patches;
}

auto adm_view(const tf::buffer<int> &patches)
    -> tf::range<const int *, tf::dynamic_size> {
  const int *first = patches.begin();
  return tf::make_range(first, first + patches.size());
}

auto adm_live_rows(const adm_owner_t &owner) -> std::vector<int> {
  std::vector<int> rows;
  for (std::size_t row = 0; row < std::size_t(owner._corners.size()); ++row)
    if (owner._states[row] == adm_dt3::tet_state::live)
      rows.push_back(int(row));
  return rows;
}

/// The admission's verdict for every live cell, keyed by the cell itself.
auto adm_verdicts(const adm_owner_t &owner, const tf::buffer<int> &patches,
                  int patch) -> std::vector<std::pair<std::array<int, 4>, int>> {
  std::vector<std::pair<std::array<int, 4>, int>> verdicts;
  for (auto row : adm_live_rows(owner)) {
    std::array<int, 4> cell{};
    for (std::size_t k = 0; k < 4; ++k)
      cell[k] = owner._corners[std::size_t(row)][k];
    std::sort(cell.begin(), cell.end());
    verdicts.push_back(
        {cell, int(adm_dt3::tet_row_admits(patches, owner, row, patch))});
  }
  std::sort(verdicts.begin(), verdicts.end());
  return verdicts;
}

/// The segments of a cloud: pairs of sites far enough apart to cross cells.
auto adm_segments(const adm_owner_t &owner, std::uint32_t seed, std::size_t n)
    -> std::vector<std::array<int, 2>> {
  std::mt19937 rng(seed);
  const int sites = int(adm_dt3::tet_site_count(owner));
  std::vector<std::array<int, 2>> segments;
  for (std::size_t i = 0; i < n; ++i) {
    const int a = int(rng() % std::uint32_t(sites));
    int b = int(rng() % std::uint32_t(sites));
    if (b == a)
      b = (b + 1) % sites;
    segments.push_back({a, b});
  }
  return segments;
}

/// The cells as published, the digest a round's product is read by.
auto adm_cells_of(const adm_owner_t &owner) -> std::vector<std::array<int, 4>> {
  std::vector<std::array<int, 4>> cells;
  for (auto row : adm_live_rows(owner)) {
    std::array<int, 4> cell{};
    for (std::size_t k = 0; k < 4; ++k)
      cell[k] = owner._corners[std::size_t(row)][k];
    std::sort(cell.begin(), cell.end());
    cells.push_back(cell);
  }
  std::sort(cells.begin(), cells.end());
  return cells;
}

} // namespace

TEST_CASE("tet admission: a row's region is the patch of its first corner",
          "[topology][delaunay3][regions]") {
  const auto flat = adm_cloud(5101, 300, 1 << 16, 10);
  adm_owner_t owner;
  REQUIRE(adm_build(owner, flat));
  const auto patches = adm_site_patches(owner, 4, 3);
  const auto rows = adm_live_rows(owner);
  REQUIRE(rows.size() > 0);
  adm_dt3::tet_round_state<int> closed;
  for (auto row : rows)
    CHECK(adm_dt3::admit_tet_row(owner, closed, row));
  CHECK(!closed.stopped);
  adm_dt3::tet_round_state<int> round;
  round.rows = adm_dt3::tet_row_pool{10, 20};
  round.sites = adm_view(patches);
  std::size_t admitted = 0;
  std::size_t foreign = 0;
  std::size_t nobodys = 0;
  for (auto row : rows) {
    const int site = owner._corners[std::size_t(row)][0];
    const int patch = site >= 0 && std::size_t(site) < patches.size()
                          ? patches[std::size_t(site)]
                          : -1;
    adm_dt3::open_tet_task(round, 0);
    const bool takes = adm_dt3::admit_tet_row(owner, round, row);
    CHECK(takes == (patch == 0));
    CHECK(round.stopped == !takes);
    if (takes)
      ++admitted;
    else if (patch < 0)
      ++nobodys;
    else
      ++foreign;
    // the row's own site answers for it, and for every row a flip makes on it
    adm_dt3::open_tet_task(round, 0);
    CHECK(adm_dt3::admit_tet_site_patch(round, site) == takes);
  }
  CHECK(admitted > 0);
  CHECK(foreign > 0);
  CHECK(nobodys > 0);
  // a link across the hull names no row, and no task stops on it
  adm_dt3::open_tet_task(round, 0);
  CHECK(adm_dt3::admit_tet_row(owner, round, adm_owner_t::none));
  CHECK(!round.stopped);
}

TEST_CASE("tet admission: a compaction carries nothing, the mark being no "
          "table over the rows",
          "[topology][delaunay3][regions]") {
  const auto flat = adm_cloud(5102, 400, 1 << 16, 10);
  adm_owner_t owner;
  REQUIRE(adm_build(owner, flat));
  const auto patches = adm_site_patches(owner, 5, 0);
  const auto before = adm_verdicts(owner, patches, 2);
  adm_dt3::compact_tetrahedralization<true, true>(owner);
  REQUIRE(adm_dt3::check_tetrahedralization_validity(owner));
  CHECK(adm_verdicts(owner, patches, 2) == before);
  CHECK(before.size() > 0);
}

TEST_CASE("tet admission: a flip is refused unless its whole write set is "
          "the task's",
          "[topology][delaunay3][regions]") {
  const auto flat = adm_cloud(5103, 300, 1 << 16, 10);
  adm_owner_t owner;
  REQUIRE(adm_build(owner, flat));
  auto patches = adm_site_patches(owner, 1, 0);
  adm_dt3::tet_round_state<int> round;
  round.rows = adm_dt3::tet_row_pool{10, 20};
  round.sites = adm_view(patches);
  const auto names_site = [&owner, &patches](int row) {
    const int site = owner._corners[std::size_t(row)][0];
    return site >= 0 && std::size_t(site) < patches.size();
  };
  int interior = adm_owner_t::none;
  int away = adm_owner_t::none;
  for (auto row : adm_live_rows(owner)) {
    bool whole = names_site(row);
    int apart = adm_owner_t::none;
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const int mate = owner._neighbors[std::size_t(row)][slot];
      whole = whole && mate >= 0 &&
              owner._states[std::size_t(mate)] == adm_dt3::tet_state::live &&
              names_site(mate);
      // the mate's own first corner, where it is not the cell's own: the one
      // site that marks the mate another region's and leaves the cell this
      // task's
      if (whole && owner._corners[std::size_t(mate)][0] !=
                       owner._corners[std::size_t(row)][0])
        apart = owner._corners[std::size_t(mate)][0];
    }
    if (whole && apart != adm_owner_t::none) {
      interior = row;
      away = apart;
      break;
    }
  }
  REQUIRE(interior != adm_owner_t::none);
  adm_dt3::open_tet_task(round, 0);
  CHECK(adm_dt3::admit_tet_flip_closure(owner, round, interior));
  CHECK(!round.stopped);
  // one mate of another region refuses the flip before it writes, while the
  // cell itself is still the task's
  patches[std::size_t(away)] = 1;
  adm_dt3::open_tet_task(round, 0);
  CHECK(adm_dt3::admit_tet_row(owner, round, interior));
  CHECK(!adm_dt3::admit_tet_flip_closure(owner, round, interior));
  CHECK(round.stopped);
}

TEST_CASE("tet admission: a round runs its own regions and leaves the rest "
          "in due order at any worker count",
          "[topology][delaunay3][regions]") {
  const auto flat = adm_cloud(5104, 300, 1 << 16, 10);
  adm_owner_t owner;
  REQUIRE(adm_build(owner, flat));
  const auto patches = adm_site_patches(owner, 4, 5);
  const auto live = adm_live_rows(owner);
  REQUIRE(live.size() > 0);
  const std::size_t due = adm_dt3::tet_execution_tuning::region_round_rows + 64;
  std::vector<int> rows;
  for (std::size_t at = 0; at < due; ++at)
    rows.push_back(live[at % live.size()]);
  const auto patch_of_row = [&owner, &patches](int row) {
    const int site = owner._corners[std::size_t(row)][0];
    return site >= 0 && std::size_t(site) < patches.size()
               ? patches[std::size_t(site)]
               : -1;
  };
  const auto patch_of = [&patch_of_row, &rows](std::size_t at) {
    return patch_of_row(rows[at]);
  };
  // every seventh task reaches one shared row as well, which is its own
  // region's alone
  int shared = adm_owner_t::none;
  for (auto row : live)
    if (patch_of_row(row) >= 0) {
      shared = row;
      break;
    }
  REQUIRE(shared != adm_owner_t::none);
  const int shared_patch = patch_of_row(shared);
  std::vector<int> expected;
  for (std::size_t at = 0; at < due; ++at)
    if (patch_of(at) < 0 || (at % 7 == 0 && patch_of(at) != shared_patch))
      expected.push_back(int(at));
  const auto cells = adm_cells_of(owner);
  std::vector<int> product;
  for (int workers : {1, 4}) {
    adm_owner_t own;
    REQUIRE(adm_build(own, flat));
    adm_dt3::tet_flip_scratch<int> scratch;
    adm_dt3::tet_round_workspace<int> work;
    tf::buffer<int> left;
    const auto task = [&own, &rows, shared](adm_dt3::tet_flip_scratch<int> &mine,
                                            int at) {
      if (!adm_dt3::admit_tet_row(own, mine.round, rows[std::size_t(at)]))
        return false;
      if (at % 7 == 0 && !adm_dt3::admit_tet_row(own, mine.round, shared))
        return false;
      return true;
    };
    tbb::task_arena arena(workers);
    arena.execute([&] {
      REQUIRE(adm_dt3::run_tet_region_round(
          own, adm_dt3::tet_standing_partition{4}, patches, due, 0, patch_of,
          scratch, work, task, left, scratch));
    });
    adm_dt3::compact_tetrahedralization<true, true>(own);
    REQUIRE(adm_dt3::check_tetrahedralization_validity(own));
    CHECK(adm_cells_of(own) == cells);
    std::vector<int> stopped(left.begin(), left.end());
    if (workers == 1)
      product = stopped;
    else
      CHECK(stopped == product);
    CHECK(stopped == expected);
  }
  CHECK(product.size() > 0);
}

TEST_CASE("tet admission: a partition no round of the phase meets states "
          "nothing and changes nothing",
          "[topology][delaunay3][regions]") {
  const auto flat = adm_cloud(5105, 400, 1 << 16, 12);
  const int depth = 12;
  std::vector<std::array<int, 4>> product;
  std::vector<int> statuses;
  for (int stated = 0; stated < 2; ++stated) {
    adm_owner_t owner;
    REQUIRE(adm_build(owner, flat));
    tf::none_t work{};
    adm_complex dt{owner, work};
    const auto segments = adm_segments(owner, 5106, 32);
    adm_recovery recovery;
    if (stated) {
      const auto patches = adm_site_patches(owner, 4, 0);
      recovery.site_patches.allocate(patches.size());
      for (std::size_t site = 0; site < patches.size(); ++site)
        recovery.site_patches[site] = patches[site];
      recovery.n_regions = 4;
    }
    adm_dt3::compact_tet_dead_rows<true>(owner);
    REQUIRE(adm_dt3::recover_tet_segments(
        dt, segments, 0, depth, recovery, adm_dt3::tet_refusing_cut{},
        adm_dt3::tet_standing_partition{recovery.n_regions}));
    adm_dt3::compact_tetrahedralization<true, true>(owner);
    REQUIRE(adm_dt3::check_tetrahedralization_validity(owner));
    for (std::size_t s = 0; s < segments.size(); ++s) {
      CAPTURE(s, int(recovery.status[s]));
      CHECK((recovery.status[s] == adm_status::recovered ||
             recovery.status[s] == adm_status::depth));
    }
    // the phase's rounds are far below the threshold, so none engaged
    CHECK(recovery.region_tasks == 0);
    const auto cells = adm_cells_of(owner);
    std::vector<int> status;
    for (auto s : recovery.status)
      status.push_back(int(s));
    if (!stated) {
      product = cells;
      statuses = status;
    } else {
      CHECK(cells == product);
      CHECK(status == statuses);
    }
  }
  CHECK(product.size() > 0);
}
