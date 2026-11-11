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
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/tetmesh/cut/land_tet_reach_points.hpp>
#include <trueform/tetmesh/cut/tet_slot_holders.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace {

using lr_int = tf::exact::int32;

/// The drain the landing reads: its points and the slot triangles, the
/// slots being the point ids.
struct lr_drain {
  tf::points_buffer<lr_int, 3> points;
  tf::core::std_vector<std::array<int, 3>> faces;
  tf::tetmesh::tet_slot_holders<int> holders;

  auto state(const tf::core::std_vector<std::array<lr_int, 3>> &positions,
             const tf::core::std_vector<std::array<int, 3>> &triangles)
      -> void {
    points.allocate(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
      points[i] = tf::exact::pt3<lr_int>{positions[i][0], positions[i][1],
                                         positions[i][2]};
    faces = triangles;
    tf::tetmesh::state_tet_slot_holders<int>(faces, positions.size(), holders);
  }
};

/// The rows on `edge_points` after the landing over `drain`.
auto lr_land(lr_drain &drain,
             const tf::core::std_vector<std::array<int, 3>> &rows, lr_int reach)
    -> tf::buffer<std::array<int, 3>> {
  tf::buffer<std::array<int, 3>> edge_points;
  for (const auto &row : rows)
    edge_points.push_back(row);
  std::sort(edge_points.begin(), edge_points.end());
  tf::tetmesh::tet_reach_landing_workspace<int> work;
  tf::tetmesh::land_tet_reach_points<lr_int>(
      drain.points, edge_points, drain.holders,
      [](std::size_t) { return true; },
      [&drain](int t, int slot) {
        const auto &f = drain.faces[std::size_t(t)];
        return f[0] == slot || f[1] == slot || f[2] == slot;
      },
      [&drain](int t) { return drain.faces[std::size_t(t)]; },
      [&drain](int t, std::size_t k) {
        const int a = drain.faces[std::size_t(t)][k];
        const int b = drain.faces[std::size_t(t)][(k + 1) % 3];
        return std::array<int, 2>{std::min(a, b), std::max(a, b)};
      },
      reach, work);
  return edge_points;
}

/// A face of the sub-unit fixture's first tetrahedron moved by 20 lattice
/// units on every axis, so it and its point stand in one cell of a reach of
/// 32, the quantum the box of half extent 2^28 makes.
auto lr_tet_face() -> tf::core::std_vector<std::array<lr_int, 3>> {
  return {{23, 15, 12}, {15, 16, 27}, {13, 26, 20}, {20, 20, 20}};
}

/// A needle whose long side runs 4000 units, past a reach of 128.
auto lr_needle(lr_int y) -> tf::core::std_vector<std::array<lr_int, 3>> {
  return {{0, 0, 0}, {4000, 0, 0}, {4000, 40, 0}, {1000, y, 0}};
}

} // namespace

TEST_CASE("land_tet_reach_points: a member inside one reach cell is an "
          "ordinary member, its point stated on its other sides",
          "[tetmesh][cut]") {
  lr_drain drain;
  drain.state(lr_tet_face(), {{0, 1, 2}});
  const auto rows = lr_land(drain, {{0, 1, 3}}, 32);
  CHECK(rows.size() > 1);
  for (const auto &row : rows)
    CHECK(row[2] == 3);
}

TEST_CASE("land_tet_reach_points: a point sharing its cell with the lattice "
          "point of another side nearest it stands on that side as itself",
          "[tetmesh][cut]") {
  SECTION("the point is stated on the second side as itself, nothing "
          "minted and nothing moved") {
    lr_drain drain;
    drain.state(lr_needle(20), {{0, 1, 2}});
    const auto rows = lr_land(drain, {{0, 2, 3}}, 128);
    REQUIRE(rows.size() == 2);
    const std::array<int, 3> landed{0, 1, 3};
    CHECK(std::binary_search(rows.begin(), rows.end(), landed));
    CHECK(drain.points.size() == 4);
    const auto p = drain.points[3];
    CHECK(p[0] == 1000);
    CHECK(p[1] == 20);
    CHECK(p[2] == 0);
  }
  SECTION("an original on a side is stated there at its own position") {
    lr_drain drain;
    drain.state(lr_needle(0), {{0, 1, 2}});
    const auto rows = lr_land(drain, {{0, 2, 3}}, 128);
    const std::array<int, 3> landed{0, 1, 3};
    CHECK(std::binary_search(rows.begin(), rows.end(), landed));
    const auto p = drain.points[3];
    CHECK(p[0] == 1000);
    CHECK(p[1] == 0);
  }
  SECTION("a point in another cell lands nowhere") {
    lr_drain drain;
    drain.state(lr_needle(200), {{0, 1, 2}});
    const auto rows = lr_land(drain, {{0, 2, 3}}, 128);
    CHECK(rows.size() == 1);
  }
}
