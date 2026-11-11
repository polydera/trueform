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
#include <trueform/exact/int32.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/tetmesh/cut/count_tet_drain_member_openness.hpp>
#include <trueform/tetmesh/cut/tet_changed_triangle.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace {

using mo_point = tf::exact::pt3<tf::exact::int32>;

/// A member of a drain: its corners, the points the drain states on each of
/// its sides, and the pieces it gives.
struct mo_member {
  std::array<int, 3> corners;
  std::array<std::vector<int>, 3> sides;
  std::vector<std::array<int, 3>> pieces;
};

/// The drain the readers answer from: where every point stands, where the
/// wave leaves it, and the members.
struct mo_drain {
  std::vector<mo_point> points;
  std::vector<int> speaks;
  std::vector<mo_member> members;
};

auto mo_openness(const mo_drain &drain) -> std::size_t {
  tf::buffer<tf::tetmesh::tet_changed_triangle<int>> changed;
  tf::buffer<std::array<int, 3>> pieces;
  for (std::size_t t = 0; t < drain.members.size(); ++t) {
    const int first = int(pieces.size());
    for (const auto &piece : drain.members[t].pieces)
      pieces.push_back(piece);
    changed.push_back({int(t), first, int(pieces.size())});
  }
  return tf::tetmesh::count_tet_drain_member_openness<tf::exact::int32>(
      changed, pieces,
      [&drain](int t) { return drain.members[std::size_t(t)].corners; },
      [&drain](int t, std::size_t k, const auto &visit) {
        for (auto id : drain.members[std::size_t(t)].sides[k])
          visit(id);
      },
      [&drain](int id) { return drain.points[std::size_t(id)]; },
      [&drain](int id) { return drain.speaks[std::size_t(id)]; });
}

/// A member whose two sides from one corner carry one identity: the wave
/// welds the point on each into the same one, so the corner's run cancels
/// and the triangle beyond it runs the rest of the boundary.
auto mo_tip() -> mo_member {
  mo_member member;
  member.corners = {0, 1, 2};
  member.sides[0] = {3};
  member.sides[2] = {4};
  member.pieces = {{3, 1, 2}};
  return member;
}

/// A member whose sides all carry one point inside it: its one piece sheds
/// the corner's two sides, which no weld retires.
auto mo_shed() -> mo_member {
  mo_member member;
  member.corners = {5, 6, 7};
  member.sides[0] = {8};
  member.sides[1] = {8};
  member.sides[2] = {8};
  member.pieces = {{6, 7, 8}};
  return member;
}

/// A member one of whose corners the drain states on its opposite side, a
/// point on another side welded into that corner: a line, whose pieces all
/// dropped.
auto mo_line() -> mo_member {
  mo_member member;
  member.corners = {0, 1, 2};
  member.sides[0] = {3};
  member.sides[1] = {0};
  return member;
}

auto mo_scene() -> mo_drain {
  mo_drain drain;
  drain.points = {{0, 0, 0}, {100, 0, 0}, {0, 100, 0}, {1, 0, 0}, {0, 1, 0},
                  {0, 0, 0}, {100, 0, 0}, {0, 100, 0}, {2, 2, 0}};
  drain.speaks = {0, 1, 2, 3, 3, 5, 6, 7, 8};
  return drain;
}

} // namespace

TEST_CASE("count_tet_drain_member_openness: a member's pieces run its own "
          "boundary once",
          "[tetmesh][cut]") {
  SECTION("a tip whose two sides carry one identity is closed") {
    auto drain = mo_scene();
    drain.members = {mo_tip()};
    CHECK(mo_openness(drain) == 0);
  }
  SECTION("a member whose piece sheds a tip is open") {
    auto drain = mo_scene();
    drain.members = {mo_shed()};
    CHECK(mo_openness(drain) == 1);
  }
  SECTION("a line every piece of which dropped runs nothing") {
    auto drain = mo_scene();
    drain.speaks[3] = 0;
    drain.members = {mo_line()};
    CHECK(mo_openness(drain) == 0);
  }
  SECTION("the count is per member") {
    auto drain = mo_scene();
    drain.members = {mo_tip(), mo_shed()};
    CHECK(mo_openness(drain) == 1);
  }
}
