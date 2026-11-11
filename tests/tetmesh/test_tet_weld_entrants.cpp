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
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/intersect/graph/plane_uncut_entrants.hpp>
#include <trueform/tetmesh/cut/tet_slot_holders.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

TEST_CASE("weld entrants: the drain's holders through the arrangement's "
          "discovery equal the drain's own walk",
          "[tetmesh][entrants]") {
  // ten slot triangles over eight slots: the first eight stated as the
  // standing membership, the last two appended past it
  std::vector<std::array<int, 3>> faces{
      {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6},
      {5, 6, 7}, {0, 2, 4}, {1, 3, 5}, {2, 5, 7}, {0, 5, 6}};
  tf::tetmesh::tet_slot_holders<int> holders;
  const std::vector<std::array<int, 3>> standing(faces.begin(),
                                                 faces.begin() + 8);
  tf::tetmesh::state_tet_slot_holders(standing, 8, holders);
  tf::tetmesh::state_tet_slot_holders(faces, 8, holders);
  REQUIRE(holders.covered == 8);
  REQUIRE(holders.appended.size() == 6);

  // slots 2 and 5 retired; triangle 3 dead, triangle 1 already changed
  const std::vector<int> retired{2, 5};
  const auto live = [](int t) { return t != 3; };
  const std::vector<int> changed{1};
  const auto is_changed = [&changed](int t) {
    return std::binary_search(changed.begin(), changed.end(), t);
  };

  // the drain's applier: a slot's standing row, then its appended run
  const auto drain_holders = [&holders](int slot, const auto &f) {
    if (std::size_t(slot) < holders.slots)
      for (auto u : holders.standing[std::size_t(slot)])
        f(int(u), int(u));
    for (auto at = std::lower_bound(holders.appended.begin(),
                                    holders.appended.end(),
                                    std::array<int, 2>{slot, -1});
         at != holders.appended.end() && (*at)[0] == slot; ++at)
      f((*at)[1], (*at)[1]);
  };
  const auto is_named = [&live, &is_changed](int t) {
    return !live(t) || is_changed(t);
  };
  tf::buffer<int> entrants;
  tf::intersect::graph::discover_weld_entrants<int>(retired, drain_holders,
                                                    is_named, entrants);
  std::sort(entrants.begin(), entrants.end());
  entrants.erase_till_end(std::unique(entrants.begin(), entrants.end()));

  // the drain's own walk, as enter_tet_retired_holders states it
  std::vector<int> walked;
  for (auto slot : retired)
    tf::tetmesh::for_each_tet_slot_holder(holders, slot, [&](int u) {
      if (live(u) && !is_changed(u))
        walked.push_back(u);
    });
  std::sort(walked.begin(), walked.end());
  walked.erase(std::unique(walked.begin(), walked.end()), walked.end());

  const std::vector<int> expected{0, 2, 4, 5, 6, 7, 8, 9};
  CHECK(std::vector<int>(entrants.begin(), entrants.end()) == expected);
  CHECK(walked == expected);
}
