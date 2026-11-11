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
#include <trueform/arrangement/identity/identity_substrate.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/intersect/graph/plane_tables.hpp>
#include <trueform/tetmesh/cut/plane_drain_world.hpp>
#include <trueform/tetmesh/cut/state_tet_lane_occurrences.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

using dn_int = tf::exact::int32;

/// The wave over one lane of one member `(0, 1, 2)`, the drain's points
/// `positions` of which the first `n_originals` are originals: side `(0, 1)`
/// split at 4, side `(1, 2)` at 5, and the chords `(3, 1)` and `(4, 5)`
/// crossing at a point the input quantum `reach` holds in one box with 3.
/// Returns the identities the wave minted and whether a piece names 3.
auto dn_wave(const tf::core::std_vector<std::array<int, 3>> &positions,
             int n_originals, dn_int reach) -> std::array<int, 2> {
  tf::points_buffer<dn_int, 3> points;
  points.allocate(positions.size());
  for (std::size_t p = 0; p < positions.size(); ++p)
    for (std::size_t d = 0; d < 3; ++d)
      points[p][d] = dn_int(positions[p][d]);
  const tf::tetmesh::tet_drain_names<int> names{
      n_originals, int(positions.size()) - n_originals};
  tf::buffer<int> member_offsets;
  member_offsets.push_back(0);
  member_offsets.push_back(1);
  tf::buffer<std::array<int, 3>> members;
  members.push_back({0, 1, 2});
  const auto side_points = [](std::size_t, std::size_t k,
                              tf::buffer<int> &out) {
    out.clear();
    if (k == 0)
      out.push_back(4);
    if (k == 1)
      out.push_back(5);
  };
  const auto chord_pieces = [](std::size_t,
                               tf::buffer<std::array<int, 2>> &out) {
    out.clear();
    out.push_back({3, 1});
    out.push_back({4, 5});
  };
  tf::buffer<int> face_def_offsets, emission_to_canon, run;
  tf::buffer<std::array<int, 2>> pieces;
  tf::intersect::graph::plane_tables<int, dn_int> tables;
  tf::tetmesh::state_tet_drain_tables<int, dn_int>(
      names, member_offsets, members, side_points, chord_pieces,
      face_def_offsets, emission_to_canon, run, pieces, tables);
  tf::tetmesh::tet_lane_occurrences<int, dn_int> lanes;
  tf::tetmesh::plane_drain_world<int, dn_int, tf::points_buffer<dn_int, 3>>
      world(points, member_offsets, members, tables, lanes);
  tf::buffer<int> created_name;
  const auto read = [&](std::int16_t tag, int id) {
    return world.point(names.point(tag, id, created_name));
  };
  tf::buffer<int> vertex_offsets;
  tf::arrangement::identity_substrate<int, dn_int> identities;
  tf::arrangement::plane_arrangement<int, dn_int> arrangement;
  tf::tetmesh::arrange_tet_drain(world, names, reach, read, vertex_offsets,
                                 identities, arrangement);
  REQUIRE(arrangement.failed().size() == 0);
  int names_3 = 0;
  for (const auto &triangle : arrangement.triangles())
    for (auto corner : triangle)
      names_3 += corner == 3;
  return {int(identities.n_plane_created()), names_3 != 0};
}

} // namespace

TEST_CASE("tet drain names: a wave's mint within the window of a created "
          "identity is that identity, the standing one surviving",
          "[tetmesh][cut]") {
  // the chords cross at (309, 296), nine units from the created point 3 at
  // (300, 300) and in its box of side 32
  const tf::core::std_vector<std::array<int, 3>> positions{
      {0, 0, 0},     {1000, 0, 0}, {0, 1000, 0},
      {300, 300, 0}, {309, 0, 0},  {309, 691, 0}};
  const auto created = dn_wave(positions, 3, dn_int(32));
  CHECK(created[0] == 0);
  CHECK(created[1] == 1);
}

TEST_CASE("tet drain names: an original keeps the exact law in the window",
          "[tetmesh][cut]") {
  // the same lane with every point an original: the crossing mints
  const tf::core::std_vector<std::array<int, 3>> positions{
      {0, 0, 0},     {1000, 0, 0}, {0, 1000, 0},
      {300, 300, 0}, {309, 0, 0},  {309, 691, 0}};
  const auto created = dn_wave(positions, 6, dn_int(32));
  CHECK(created[0] == 1);
}

TEST_CASE("tet drain names: a created identity is the arrangement's base "
          "created identity of its number",
          "[tetmesh][cut]") {
  const tf::tetmesh::tet_drain_names<int> names{10, 4};
  tf::buffer<int> created_name;
  created_name.push_back(20);
  CHECK(names.name(3) == std::pair<std::int16_t, int>{0, 3});
  CHECK(names.name(12) == std::pair<std::int16_t, int>{-1, 2});
  CHECK(names.point(std::int16_t(-1), 2, created_name) == 12);
  CHECK(names.point(std::int16_t(-1), 4, created_name) == 20);
  CHECK(names.point_of_flat(12, created_name) == 12);
  CHECK(names.point_of_flat(14, created_name) == 20);
}
