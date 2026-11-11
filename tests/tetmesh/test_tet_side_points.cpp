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
#include <trueform/core/memory.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/tetmesh/cut/count_tet_drain_t_junctions.hpp>
#include <trueform/tetmesh/cut/state_tet_lane_occurrences.hpp>
#include <trueform/tetmesh/cut/state_tet_side_parameters.hpp>
#include <trueform/tetmesh/cut/weld_tet_side_points.hpp>
#include <trueform/tetmesh/identity/tet_identity_authority.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace {

using sp_int = tf::exact::int32;
using sp_pt = tf::exact::pt3<sp_int>;
using sp_param = tf::exact::meta<sp_int>::param_type;
using sp_authority = tf::tetmesh::tet_identity_authority<int, sp_int>;

auto sp_point(sp_int x, sp_int y, sp_int z) -> sp_pt {
  sp_pt p;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  return p;
}

/// One side `(0, 1)` from the origin to `(1024, 0, 0)` and the points the
/// drain states on it: the originals 0, 1 (the ends) and 2 on the side, then
/// the created points in id order 3, 4, 5, 6, 7 — the id order is not the
/// order along the side.
struct sp_side {
  sp_authority identities;
  tf::buffer<std::array<int, 3>> rows;
  tf::buffer<sp_param> t;

  sp_side() {
    const tf::core::std_vector<sp_pt> originals{
        sp_point(0, 0, 0), sp_point(1024, 0, 0), sp_point(700, 0, 0)};
    identities.name_originals(originals, tf::core::std_vector<int>{0, 1, 2});
    tf::buffer<int> ids;
    identities.name_positions(
        tf::core::std_vector<sp_pt>{sp_point(513, 2, 0), sp_point(256, 5, 0),
                                    sp_point(511, -3, 0), sp_point(1023, 1, 0),
                                    sp_point(702, 1, 0)},
        tf::core::std_vector<std::array<int, 2>>(5, {-1, -1}), ids);
    for (int x : {2, 3, 4, 5, 6, 7})
      rows.push_back({0, 1, x});
    tf::tetmesh::state_tet_side_parameters<sp_int>(
        rows, [this](int slot) { return identities.position_of(slot); },
        [this](int id) { return identities.position_of(id); }, t);
  }
  auto row_of(int x) const -> std::size_t {
    for (std::size_t r = 0; r < rows.size(); ++r)
      if (rows[r][2] == x)
        return r;
    return rows.size();
  }
  auto weld(sp_int quantum) -> void {
    tf::buffer<std::size_t> order;
    tf::tetmesh::weld_tet_side_points<sp_int>(
        rows, t, [this](int slot) { return identities.position_of(slot); },
        [](int slot) { return slot; }, quantum, identities, order);
  }
};

constexpr sp_param sp_full = sp_param(1) << tf::exact::meta<sp_int>::param_bits;

} // namespace

TEST_CASE("tet side points: a point stands on a side at its own dyadic "
          "parameter, its identity and position untouched",
          "[tetmesh][cut]") {
  sp_side side;
  // 256 of 1024 along the side, whatever the point's offset off it
  CHECK(side.t[side.row_of(4)] == sp_full / 4);
  CHECK(side.identities.position_of(4) == sp_point(256, 5, 0));
  CHECK(side.identities.n_created() == 5);
}

TEST_CASE("tet side points: a side's points are ordered by their parameter",
          "[tetmesh][cut]") {
  sp_side side;
  tf::buffer<std::size_t> order;
  for (std::size_t r = 0; r < side.rows.size(); ++r)
    order.push_back(r);
  std::sort(order.begin(), order.end(), [&side](std::size_t x, std::size_t y) {
    return side.t[x] < side.t[y];
  });
  tf::buffer<int> along;
  for (auto r : order)
    along.push_back(side.rows[r][2]);
  CHECK(along[0] == 4);
  CHECK(along[1] == 5);
  CHECK(along[2] == 3);
  CHECK(along[3] == 2);
  CHECK(along[4] == 7);
  CHECK(along[5] == 6);
}

TEST_CASE("tet side points: two points within the quantum along a side are "
          "one identity, adjacent in the order whatever their ids",
          "[tetmesh][cut]") {
  sp_side side;
  side.weld(4);
  side.identities.close();
  // 3 and 5 stand 2 units apart along the side, ids not adjacent
  CHECK(side.identities.standing(3) == side.identities.standing(5));
  // 4 stands far from every other point
  CHECK(side.identities.standing(4) == 4);
  // a point keeps its position when it is welded
  CHECK(side.identities.position_of(5) == sp_point(511, -3, 0));
}

TEST_CASE("tet side points: a point within a lattice unit of a side's end is "
          "the end",
          "[tetmesh][cut]") {
  sp_side side;
  side.weld(0);
  side.identities.close();
  CHECK(side.identities.standing(6) == 1);
}

TEST_CASE("tet side points: an original on a side survives, a created point "
          "welded to it retires into it",
          "[tetmesh][cut]") {
  sp_side side;
  side.weld(4);
  side.identities.close();
  CHECK(side.identities.standing(2) == 2);
  CHECK(side.identities.standing(7) == 2);
  CHECK(side.identities.position_of(2) == sp_point(700, 0, 0));
}

TEST_CASE("tet side points: an exact coincidence across names is one "
          "identity by the election",
          "[tetmesh][cut]") {
  sp_side side;
  tf::buffer<int> again;
  side.identities.name_positions(
      tf::core::std_vector<sp_pt>{sp_point(256, 5, 0), sp_point(700, 0, 0)},
      tf::core::std_vector<std::array<int, 2>>(2, {-1, -1}), again);
  CHECK(again[0] == 4);
  CHECK(again[1] == 2);
  CHECK(side.identities.n_created() == 5);
}

namespace {

/// The lane occurrences of `rows` over the one-lane members `members`,
/// each member its own triangle, `positions` read by identity and slot.
auto sp_lane(const tf::core::std_vector<sp_pt> &positions,
             const tf::buffer<std::array<int, 3>> &members,
             const tf::buffer<std::array<int, 3>> &rows)
    -> tf::tetmesh::tet_lane_occurrences<int, sp_int> {
  const auto at = [&positions](int id) { return positions[std::size_t(id)]; };
  tf::buffer<sp_param> t;
  tf::tetmesh::state_tet_side_parameters<sp_int>(rows, at, at, t);
  tf::buffer<int> lanes, triangle;
  lanes.push_back(0);
  lanes.push_back(int(members.size()));
  for (std::size_t f = 0; f < members.size(); ++f)
    triangle.push_back(int(f));
  const auto rows_on_edge = [&](int f, std::size_t k, auto &&visit) {
    const auto &m = members[std::size_t(f)];
    const int a = m[k], b = m[(k + 1) % 3];
    for (std::size_t r = 0; r < rows.size(); ++r)
      if (rows[r][0] == std::min(a, b) && rows[r][1] == std::max(a, b))
        visit(r);
  };
  tf::tetmesh::tet_lane_occurrences<int, sp_int> lane;
  tf::tetmesh::state_tet_lane_occurrences<sp_int>(
      lanes, members, triangle, rows_on_edge, rows, t, at, lane);
  return lane;
}

} // namespace

TEST_CASE("tet side points: within one lane a point stated on two crossing "
          "sides is charted at its own position, one point for both",
          "[tetmesh][cut]") {
  // one member (0, 1, 2); the created point 3 near corner 1 stands on its
  // sides (0, 1) and (1, 2), which cross there
  const tf::core::std_vector<sp_pt> positions{
      sp_point(0, 0, 0), sp_point(1024, 0, 0), sp_point(1024, 1024, 0),
      sp_point(1023, 2, 0)};
  tf::buffer<std::array<int, 3>> members, rows;
  members.push_back({0, 1, 2});
  rows.push_back({0, 1, 3});
  rows.push_back({0, 2, 1});
  rows.push_back({1, 2, 3});
  const auto lane = sp_lane(positions, members, rows);
  CHECK(lane.position.size() == 0);
  CHECK(lane.chart(0, 3, positions[3]) == positions[3]);
  // a corner of the lane stated on another side is at its own position
  CHECK(lane.chart(0, 1, positions[1]) == positions[1]);
}

TEST_CASE("tet side points: within one lane a point stated on collinear "
          "sides is one occurrence, evaluated on its first side",
          "[tetmesh][cut]") {
  // members (0, 1, 2) and (1, 4, 2) in one lane; their sides (0, 1) and
  // (1, 4) lie on one line, and the point 5 stands on both
  const tf::core::std_vector<sp_pt> positions{
      sp_point(0, 0, 0), sp_point(1024, 0, 0), sp_point(1024, 1024, 0),
      sp_point(0, 0, 0), sp_point(2048, 0, 0), sp_point(1023, 2, 0)};
  tf::buffer<std::array<int, 3>> members, rows;
  members.push_back({0, 1, 2});
  members.push_back({1, 4, 2});
  rows.push_back({0, 1, 5});
  rows.push_back({1, 4, 5});
  const auto lane = sp_lane(positions, members, rows);
  REQUIRE(lane.position.size() == 1);
  CHECK(lane.position[0] == sp_point(1023, 0, 0));
  CHECK(lane.chart(0, 5, positions[5]) == sp_point(1023, 0, 0));
  // another lane charts it at its own position
  CHECK(lane.chart(1, 5, positions[5]) == positions[5]);
}

namespace {

/// The T-junctions of three triangles over slots `0..4`: the holder
/// `(0, 1, 2)` and its neighbours `(1, 0, 3)` across `(0, 1)` and
/// `(0, 2, 4)` across `(0, 2)`, the points the rows `rows` state,
/// each triangle changed into `pieces` (`offsets` per triangle).
auto sp_t_junctions(const tf::buffer<std::array<int, 3>> &rows,
                    const tf::buffer<std::array<int, 3>> &pieces,
                    const std::array<int, 4> &offsets) -> std::size_t {
  tf::buffer<std::array<int, 3>> faces;
  faces.push_back({0, 1, 2});
  faces.push_back({1, 0, 3});
  faces.push_back({0, 2, 4});
  tf::tetmesh::tet_slot_holders<int> holders;
  tf::tetmesh::state_tet_slot_holders<int>(faces, 7, holders);
  tf::buffer<tf::tetmesh::tet_changed_triangle<int>> changed;
  for (int t = 0; t < 3; ++t)
    changed.push_back(
        {t, offsets[std::size_t(t)], offsets[std::size_t(t) + 1]});
  const auto live = [](std::size_t) { return true; };
  const auto holds = [&faces](int t, int slot) {
    const auto &f = faces[std::size_t(t)];
    return f[0] == slot || f[1] == slot || f[2] == slot;
  };
  const auto corners_of = [&faces](int t) { return faces[std::size_t(t)]; };
  const auto edge_key = [&faces](int t, std::size_t k) {
    const int a = faces[std::size_t(t)][k];
    const int b = faces[std::size_t(t)][(k + 1) % 3];
    return std::array<int, 2>{std::min(a, b), std::max(a, b)};
  };
  const auto speaks = [](int id) { return id; };
  return tf::tetmesh::count_tet_drain_t_junctions(rows, holders, live, holds,
                                                  corners_of, edge_key, speaks,
                                                  changed, pieces);
}

} // namespace

TEST_CASE("tet drain t-junctions: a point two sides of its holder state "
          "is held by the holder's line portion",
          "[tetmesh][cut]") {
  // 5 then 6 stand on (0, 1) and on (0, 2): from corner 0 the two sides
  // are one edge as far as 6, a line holding 5, so the holder's pieces are
  // the triangle beyond 6 and the neighbours hold (0, 5, 6) once each way
  tf::buffer<std::array<int, 3>> rows, pieces;
  rows.push_back({0, 1, 5});
  rows.push_back({0, 1, 6});
  rows.push_back({0, 2, 5});
  rows.push_back({0, 2, 6});
  pieces.push_back({6, 1, 2});
  pieces.push_back({1, 6, 3});
  pieces.push_back({6, 5, 3});
  pieces.push_back({5, 0, 3});
  pieces.push_back({0, 5, 4});
  pieces.push_back({5, 6, 4});
  pieces.push_back({6, 2, 4});
  CHECK(sp_t_junctions(rows, pieces, {0, 1, 4, 7}) == 0);
}

TEST_CASE("tet drain t-junctions: a point one side states and no piece "
          "holds is a T-junction",
          "[tetmesh][cut]") {
  // 5 stands on (0, 1) alone, and the holder's pieces drop it
  tf::buffer<std::array<int, 3>> rows, pieces;
  rows.push_back({0, 1, 5});
  pieces.push_back({0, 1, 2});
  pieces.push_back({1, 5, 3});
  pieces.push_back({5, 0, 3});
  pieces.push_back({0, 2, 4});
  CHECK(sp_t_junctions(rows, pieces, {0, 1, 3, 4}) == 1);
}
