/**
 * @file plane_arrangement_generators.hpp
 * @brief Worlds and scenes for tf::arrangement::plane_arrangement testing
 *
 * Two kinds of input reach a plane arrangement, and this header states both:
 *
 * - hand-built `plane_fixture_world` worlds, whose definition tables are
 *   written out row by row so a single structural case can be stated exactly:
 *   the shared boundary, the endpoint weld, the shared split rings and the
 *   Steiner retry they carry, and the recovery worlds (bent child, multi
 *   parent, uncut neighbor, mixed tier);
 * - meshes whose local arrangement builds the world, tagged the way the
 *   pipeline tags them (tree, face membership, manifold edge link).
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "plane_fixture_world.hpp"
#include "tagged_operand.hpp"

#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/make_supported_plane_frame.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::test {

using plane_index_t = int;
using plane_def_t = tf::intersect::graph::plane_edge_def<plane_index_t>;

/// One hand-built world: the input tables, the lattice positions its
/// definitions name (originals first, then the created identities a previous
/// round minted), and the offsets that flatten a `{tag, id}` identity.
template <typename Int> struct plane_hand_world {
  plane_fixture_world<plane_index_t, Int> input;
  tf::buffer<tf::point<Int, 3>> points;
  tf::buffer<tf::point<Int, 3>> created;
  tf::buffer<plane_index_t> vertex_offsets;
  tf::buffer<plane_index_t> face_offsets;
};

/// A mesh carrying the membership and manifold links a form must publish for
/// the arrangement to walk its base loops.
template <typename Real, std::size_t Ngon> struct plane_source_form {
  using buffer_t = tf::polygons_buffer<plane_index_t, Real, 3, Ngon>;

  buffer_t mesh;
  decltype(tf::make_face_membership(
      std::declval<buffer_t &>().polygons())) membership;
  decltype(tf::make_manifold_edge_link(
      std::declval<buffer_t &>().polygons())) manifold;

  explicit plane_source_form(buffer_t input)
      : mesh(std::move(input)),
        membership(tf::make_face_membership(mesh.polygons())),
        manifold(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() {
    return mesh.polygons() | tf::tag(membership) | tf::tag(manifold);
  }
};

/// Group the raw definitions the way a prepared world is grouped: canon-major
/// in endpoint-key order, with one edge block per plane.
template <typename Input>
auto fill_plane_tables(Input &input, tf::buffer<plane_def_t> raw,
                       plane_index_t n_planes) -> void {
  std::sort(raw.begin(), raw.end(),
            [](const plane_def_t &x, const plane_def_t &y) {
              if (tf::intersect::graph::plane_def_key_less(x, y))
                return true;
              if (tf::intersect::graph::plane_def_key_less(y, x))
                return false;
              return std::tie(x.face, x.ordinal, x.side) <
                     std::tie(y.face, y.ordinal, y.side);
            });
  const auto same_key = [](const plane_def_t &x, const plane_def_t &y) {
    return !tf::intersect::graph::plane_def_key_less(x, y) &&
           !tf::intersect::graph::plane_def_key_less(y, x);
  };
  auto &tables = input.tables();
  tables.def_offsets().push_back(0);
  plane_index_t group = -1;
  for (std::size_t row = 0; row < raw.size(); ++row) {
    if (row == 0 || !same_key(raw[row - 1], raw[row])) {
      if (row != 0)
        tables.def_offsets().push_back(plane_index_t(row));
      ++group;
    }
    raw[row].id = group;
    tables.defs().push_back(raw[row]);
  }
  tables.def_offsets().push_back(plane_index_t(raw.size()));
  tables.n_canon() = group + 1;
  tables.edges().offsets_buffer().push_back(0);
  for (plane_index_t plane = 0; plane < n_planes; ++plane) {
    for (std::size_t row = 0; row < raw.size(); ++row)
      if (raw[row].face == plane)
        tables.edges().data_buffer().push_back(plane_index_t(row));
    tables.edges().offsets_buffer().push_back(
        plane_index_t(tables.edges().data_buffer().size()));
  }
}

template <typename Input>
auto fill_single_member_planes(Input &input, plane_index_t n_planes,
                               int orientation = 1) -> void {
  input.plane_members_data().offsets_buffer().push_back(0);
  for (plane_index_t plane = 0; plane < n_planes; ++plane) {
    input.plane_members_data().data_buffer().push_back(plane);
    input.plane_members_data().offsets_buffer().push_back(plane + 1);
    input.descriptor_data().push_back({0, plane});
    input.face_planes().push_back(plane);
    input.face_orientations().push_back(orientation);
  }
}

template <typename Int, typename Coordinates>
auto fill_plane_points(tf::buffer<tf::point<Int, 3>> &points,
                       const Coordinates &coordinates, Int scale) -> void {
  for (std::size_t point = 0; point < coordinates.size(); ++point) {
    tf::point<Int, 3> position;
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      position[int(coordinate)] = Int(coordinates[point][coordinate]) * scale;
    points.push_back(position);
  }
}

// =============================================================================
// The shared boundary: two faces meeting along one edge
// =============================================================================

/// Two triangles sharing the side (0,1), on two planes of their own.
template <typename Real> auto make_plane_shared_boundary_mesh() {
  const auto scale = Real(plane_index_t(1) << 20);
  tf::polygons_buffer<plane_index_t, Real, 3, 3> mesh;
  mesh.points_buffer().emplace_back(Real(-8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(4) * scale, Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(4) * scale);
  mesh.faces_buffer().emplace_back(plane_index_t(0), plane_index_t(1),
                                   plane_index_t(2));
  mesh.faces_buffer().emplace_back(plane_index_t(1), plane_index_t(0),
                                   plane_index_t(3));
  return mesh;
}

/// The world of that mesh: the shared side is one canonical group carried by
/// both planes, so a split of it must reach both.
template <typename Int>
auto make_plane_shared_boundary_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 4> coordinates{{
      {{-8, 0, 0}},
      {{8, 0, 0}},
      {{0, 4, 0}},
      {{0, 0, 4}},
  }};
  fill_plane_points(world.points, coordinates, Int(plane_index_t(1) << 20));

  const auto whole = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, 0, 0, whole});
  raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, 1, 1, whole});
  raw.push_back(
      {0, 2, -1, 0, -1, 0, 0, -1, 2, 2, std::uint8_t(whole | reversed)});
  raw.push_back(
      {0, 1, -1, 1, -1, 0, 0, -1, 0, 0, std::uint8_t(whole | reversed)});
  raw.push_back({0, 3, -1, 1, -1, 0, 0, -1, 1, 1, whole});
  raw.push_back(
      {1, 3, -1, 1, -1, 0, 0, -1, 2, 2, std::uint8_t(whole | reversed)});
  fill_plane_tables(world.input, std::move(raw), 2);
  world.input.source_tables() = world.input.tables();

  world.input.frames().allocate(2);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {0, 0, 1};
  world.input.frames()[0].plane_d = 0;
  world.input.frames()[0].plane_fallback = 0;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 2;
  world.input.frames()[1].plane_n = {0, 1, 0};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = 0;

  world.input.plane_members_data().offsets_buffer().push_back(0);
  world.input.plane_members_data().offsets_buffer().push_back(1);
  world.input.plane_members_data().offsets_buffer().push_back(2);
  world.input.plane_members_data().data_buffer().push_back(0);
  world.input.plane_members_data().data_buffer().push_back(1);
  world.input.descriptor_data().push_back({0, 0});
  world.input.descriptor_data().push_back({0, 1});
  world.input.face_planes().push_back(0);
  world.input.face_planes().push_back(1);
  world.input.face_orientations().push_back(1);
  world.input.face_orientations().push_back(-1);

  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  world.face_offsets.push_back(0);
  world.face_offsets.push_back(2);
  return world;
}

// =============================================================================
// The endpoint weld: one plane whose quad face carries a projected twin
// =============================================================================

template <typename Real> auto make_plane_endpoint_weld_mesh() {
  const auto scale = Real(plane_index_t(1) << 20);
  tf::polygons_buffer<plane_index_t, Real, 3, tf::dynamic_size> mesh;
  mesh.points_buffer().emplace_back(Real(-8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(4) * scale, Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(4) * scale);
  mesh.points_buffer().emplace_back(Real(0), Real(1) * scale, Real(0));
  mesh.faces_buffer().push_back(
      {plane_index_t(0), plane_index_t(1), plane_index_t(2)});
  mesh.faces_buffer().push_back(
      {plane_index_t(1), plane_index_t(0), plane_index_t(3), plane_index_t(4)});
  return mesh;
}

/// One plane stating one triangle, whose neighbor's fourth vertex projects
/// onto the shared side: a refinement split of that side lands on a vertex
/// the plane already has.
template <typename Int>
auto make_plane_endpoint_weld_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 5> coordinates{{
      {{-8, 0, 0}},
      {{8, 0, 0}},
      {{0, 4, 0}},
      {{0, 0, 4}},
      {{0, 1, 0}},
  }};
  fill_plane_points(world.points, coordinates, Int(plane_index_t(1) << 20));

  const auto whole = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, 0, 0, whole});
  raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, 1, 1, whole});
  raw.push_back(
      {0, 2, -1, 0, -1, 0, 0, -1, 2, 2, std::uint8_t(whole | reversed)});
  fill_plane_tables(world.input, std::move(raw), 1);
  world.input.source_tables() = world.input.tables();

  world.input.frames().allocate(1);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {0, 0, 1};
  world.input.frames()[0].plane_d = 0;
  world.input.frames()[0].plane_fallback = 0;
  world.input.plane_members_data().offsets_buffer().push_back(0);
  world.input.plane_members_data().offsets_buffer().push_back(1);
  world.input.plane_members_data().data_buffer().push_back(0);
  world.input.descriptor_data().push_back({0, 0});
  world.input.face_planes().push_back(0);
  world.input.face_orientations().push_back(1);

  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  world.face_offsets.push_back(0);
  world.face_offsets.push_back(2);
  return world;
}

// =============================================================================
// The shared split rings and the Steiner retry they carry
// =============================================================================

template <typename Real> auto make_plane_steiner_retry_mesh() {
  const auto scale = Real(plane_index_t(1) << 20);
  tf::polygons_buffer<plane_index_t, Real, 3, 4> mesh;
  mesh.points_buffer().emplace_back(Real(-8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(8) * scale, Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(8) * scale, Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(-8) * scale, Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(8) * scale);
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(-8) * scale);
  mesh.points_buffer().emplace_back(Real(-4) * scale, Real(3) * scale, Real(0));
  mesh.points_buffer().emplace_back(Real(4) * scale, Real(3) * scale, Real(0));
  mesh.faces_buffer().emplace_back(plane_index_t(0), plane_index_t(3),
                                   plane_index_t(2), plane_index_t(4));
  mesh.faces_buffer().emplace_back(plane_index_t(0), plane_index_t(5),
                                   plane_index_t(2), plane_index_t(6));
  return mesh;
}

/// Two planes crossing along the x axis, each carrying a ring of boundary
/// sides plus the two interior cuts that split the shared root twice.
template <typename Int>
auto make_plane_split_rings_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 9> coordinates{{
      {{-8, 0, 0}},
      {{0, 0, 0}},
      {{8, 0, 0}},
      {{0, 8, 0}},
      {{0, -8, 0}},
      {{0, 0, 8}},
      {{0, 0, -8}},
      {{-4, 3, 0}},
      {{4, 3, 0}},
  }};
  const std::array<std::array<int, 3>, 2> created_coordinates{{
      {{-4, 0, 0}},
      {{4, 0, 0}},
  }};
  fill_plane_points(world.points, coordinates, Int(1));
  fill_plane_points(world.created, created_coordinates, Int(1));

  const auto boundary = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  raw.push_back({0, 7, -1, 0, -1, -1, 0, -1, -1, -1, 0});
  raw.push_back({1, 8, -1, 0, -1, -1, 0, -1, -1, -1, 0});
  raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({0, 1, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({0, 3, -1, 0, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {0, 4, -1, 0, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({0, 5, -1, 1, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {0, 6, -1, 1, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({1, 2, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back(
      {2, 3, -1, 0, -1, 0, 0, -1, 1, 1, std::uint8_t(boundary | reversed)});
  raw.push_back({2, 4, -1, 0, -1, 0, 0, -1, 2, 2, boundary});
  raw.push_back(
      {2, 5, -1, 1, -1, 0, 0, -1, 1, 1, std::uint8_t(boundary | reversed)});
  raw.push_back({2, 6, -1, 1, -1, 0, 0, -1, 2, 2, boundary});
  fill_plane_tables(world.input, std::move(raw), 2);

  world.input.frames().allocate(2);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {0, 0, 1};
  world.input.frames()[0].plane_d = 0;
  world.input.frames()[0].plane_fallback = 0;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 2;
  world.input.frames()[1].plane_n = {0, 1, 0};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = 0;
  for (const auto offset : std::array<plane_index_t, 3>{{0, 1, 2}})
    world.input.plane_members_data().offsets_buffer().push_back(offset);
  world.input.plane_members_data().data_buffer().push_back(0);
  world.input.plane_members_data().data_buffer().push_back(1);
  world.input.descriptor_data().push_back({0, 0});
  world.input.descriptor_data().push_back({0, 1});
  world.input.face_planes().push_back(0);
  world.input.face_planes().push_back(1);
  world.input.face_orientations().push_back(-1);
  world.input.face_orientations().push_back(-1);

  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  return world;
}

/// The split-rings world scaled onto the refinement lattice. With
/// `live_landings` the created identities sit on the shared root, so a
/// refinement split lands on a standing identity; without them they sit off
/// it and the round must mint its own.
template <typename Int>
auto make_plane_steiner_retry_world(bool live_landings)
    -> plane_hand_world<Int> {
  auto world = make_plane_split_rings_world<Int>();
  const auto scale = Int(plane_index_t(1) << 20);
  for (auto &point : world.points)
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      point[coordinate] *= scale;
  for (auto &point : world.created)
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      point[coordinate] *= scale;
  if (!live_landings)
    for (auto &point : world.created)
      point[1] = scale;
  world.input.source_tables() = world.input.tables();
  world.face_offsets.push_back(0);
  world.face_offsets.push_back(2);
  return world;
}

// =============================================================================
// The line carrier
// =============================================================================

/// A LINE CARRIER beside an area plane that shares one of its spans.
///
///   A(0) --- B(1) --- C(2) --- D(3)          apex E(4)
///
/// Plane 0 is the area triangle (A, C, E). Plane 1 holds the collinear
/// (A, C) and (B, D), whose overlap only IT can see, and (A, C) is one
/// canonical group carried by both planes, so a split of it must reach
/// plane 0. `axis` names the coordinate the collinear run varies along;
/// `twin` gives the carrier a second identity F at B's exact position, so
/// the line states one coincidence that is true.
///
/// With `presplit` the tables arrive as an upstream producer that already
/// stated the landing leaves them: the shared span is its two abutting
/// pieces on BOTH carriers, which is what the plane tier is handed when the
/// local arrangement saw the overlap first.
///
/// Both frames come from the frame producer, offered the points of the
/// carrier they belong to.
template <typename Int>
auto make_plane_line_carrier_world(int axis, bool twin = false,
                                   bool presplit = false)
    -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const int up = axis == 1 ? 2 : 1;
  const auto at = [&](int along, int away) {
    tf::point<Int, 3> position{Int(0), Int(0), Int(0)};
    position[axis] = Int(along);
    position[up] = Int(away);
    return position;
  };
  world.points.push_back(at(0, 0));
  world.points.push_back(at(10, 0));
  world.points.push_back(at(20, 0));
  world.points.push_back(at(30, 0));
  world.points.push_back(at(10, 10));
  if (twin)
    world.points.push_back(at(10, 0));

  const auto whole = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  if (presplit) {
    raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, 0, 0, whole});
    raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, 0, 0, whole});
  } else
    raw.push_back({0, 2, -1, 0, -1, 0, 0, -1, 0, 0, whole});
  raw.push_back({2, 4, -1, 0, -1, 0, 0, -1, 1, 1, whole});
  raw.push_back(
      {0, 4, -1, 0, -1, 0, 0, -1, 2, 2, std::uint8_t(whole | reversed)});
  if (presplit) {
    raw.push_back({0, 1, -1, 1, -1, 0, 0, -1, 0, 0, whole});
    raw.push_back({1, 2, -1, 1, -1, 0, 0, -1, 0, 0, whole});
    raw.push_back({2, 3, -1, 1, -1, 0, 0, -1, 1, 1, whole});
  } else {
    raw.push_back({0, 2, -1, 1, -1, 0, 0, -1, 0, 0, whole});
    raw.push_back({1, 3, -1, 1, -1, 0, 0, -1, 1, 1, whole});
  }
  if (twin)
    raw.push_back(
        {3, 5, -1, 1, -1, 0, 0, -1, 2, 2, std::uint8_t(whole | reversed)});
  fill_plane_tables(world.input, std::move(raw), 2);
  world.input.source_tables() = world.input.tables();

  world.input.frames().allocate(2);
  world.input.frames()[0] = tf::exact::make_supported_plane_frame<Int>(
      [&](const auto &consider) {
        for (const auto corner : {0, 2, 4})
          consider(world.points[std::size_t(corner)]);
      });
  world.input.frames()[1] = tf::exact::make_supported_plane_frame<Int>(
      [&](const auto &consider) {
        for (const auto corner : {0, 1, 2, 3})
          consider(world.points[std::size_t(corner)]);
      });

  fill_single_member_planes(world.input, 2);
  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  world.face_offsets.push_back(0);
  world.face_offsets.push_back(2);
  return world;
}

// =============================================================================
// The recovery worlds
// =============================================================================

/// A quad whose interior cut is crossed by a segment, beside a second plane
/// whose own crossing is born of the first round's identity. With
/// `landing_world` the second crossing lands on that identity; without it the
/// wave must mint again.
template <typename Int>
auto make_plane_bent_child_world(bool landing_world) -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  std::array<std::array<int, 3>, 14> coordinates{{
      {{-20, -80, -15}},
      {{15, -80, 10}},
      {{27, 80, 22}},
      {{-8, 80, -3}},
      {{0, 0, 1}},
      {{7, 0, 6}},
      {{2, -20, 2}},
      {{5, 20, 5}},
      {{3, -9, 3}},
      {{3, -11, 3}},
      {{-30, -30, -30}},
      {{30, -30, 30}},
      {{30, 30, 30}},
      {{-30, 30, -30}},
  }};
  if (landing_world) {
    // the midpoint of the child (2,-20,2)-(4,0,4), which the round-1 identity
    // (the blend of (0,0,1) and (7,0,6) at one half) closes
    coordinates[8] = {3, -10, 3};
    coordinates[9] = {20, -10, 20};
  }
  fill_plane_points(world.points, coordinates, Int(1));

  const auto boundary = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {0, 3, -1, 0, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, 1, 1, boundary});
  raw.push_back({2, 3, -1, 0, -1, 0, 0, -1, 2, 2, boundary});
  raw.push_back({4, 5, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({6, 7, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({6, 7, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({8, 9, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({10, 11, -1, 1, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {10, 13, -1, 1, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({11, 12, -1, 1, -1, 0, 0, -1, 1, 1, boundary});
  raw.push_back({12, 13, -1, 1, -1, 0, 0, -1, 2, 2, boundary});
  fill_plane_tables(world.input, std::move(raw), 2);

  world.input.frames().allocate(2);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {-100, -3, 140};
  world.input.frames()[0].plane_d = 140;
  world.input.frames()[0].plane_fallback = -15;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 1;
  world.input.frames()[1].plane_n = {-1, 0, 1};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = -30;
  fill_single_member_planes(world.input, 2);
  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  return world;
}

/// THE EQUAL CHILDREN WORLD PLUS A CARRIER THAT HOLDS BOTH PARENTS, and an
/// existing identity sitting on the fused child. One face therefore states the
/// same provenance twice the moment the two pieces coincide, which is the
/// union law's own case: one row survives and both slots read it.
template <typename Int>
auto make_plane_multi_parent_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 20> coordinates{{
      {{0, 0, 0}},        {{100, 4, 0}},     {{100, -4, 0}},
      {{8, -1, -33}},     {{11, 1, 14}},     {{8, 1, -33}},
      {{11, -1, 14}},     {{30, -1, 0}},     {{-20, -20, -480}},
      {{120, -20, -620}}, {{120, 20, 380}},  {{-20, 20, 520}},
      {{-20, -20, 520}},  {{120, -20, 380}}, {{120, 20, -620}},
      {{-20, 20, -480}},  {{-20, -20, 0}},   {{120, -20, 0}},
      {{120, 20, 0}},     {{-20, 20, 0}},
  }};
  fill_plane_points(world.points, coordinates, Int(1));
  world.created.push_back(tf::point<Int, 3>(Int(5), Int(0), Int(0)));

  tf::buffer<plane_def_t> raw;
  const auto add = [&](std::array<plane_index_t, 2> from,
                       std::array<plane_index_t, 2> to, plane_index_t face,
                       std::int16_t ordinal, std::int16_t side, bool boundary) {
    std::uint8_t flags = boundary
                             ? tf::intersect::graph::plane_edge_whole_side_flag
                             : std::uint8_t(0);
    if (to < from) {
      std::swap(from, to);
      flags =
          std::uint8_t(flags | tf::intersect::graph::plane_edge_reversed_flag);
    }
    raw.push_back({from[1], to[1], -1, face, -1, std::int16_t(from[0]),
                   std::int16_t(to[0]), -1, ordinal, side, flags});
  };
  const auto add_boundary = [&](plane_index_t face, plane_index_t base) {
    add({0, base}, {0, base + 1}, face, 0, 0, true);
    add({0, base + 1}, {0, base + 2}, face, 1, 1, true);
    add({0, base + 2}, {0, base + 3}, face, 2, 2, true);
    add({0, base + 3}, {0, base}, face, 3, 3, true);
  };
  add_boundary(0, 8);
  add_boundary(1, 12);
  add_boundary(2, 16);
  add({0, 0}, {0, 1}, 0, -1, -1, false);
  add({0, 0}, {0, 1}, 2, -1, -1, false);
  add({0, 0}, {0, 2}, 1, -1, -1, false);
  add({0, 0}, {0, 2}, 2, -1, -1, false);
  add({0, 3}, {0, 4}, 0, -1, -1, false);
  add({0, 5}, {0, 6}, 1, -1, -1, false);
  add({-1, 0}, {0, 7}, 2, -1, -1, false);
  fill_plane_tables(world.input, std::move(raw), 3);

  world.input.frames().allocate(3);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {1, -25, 1};
  world.input.frames()[0].plane_d = 0;
  world.input.frames()[0].plane_fallback = -480;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 1;
  world.input.frames()[1].plane_n = {1, 25, 1};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = 520;
  world.input.frames()[2].ax0 = 0;
  world.input.frames()[2].ax1 = 1;
  world.input.frames()[2].plane_n = {0, 0, 1};
  world.input.frames()[2].plane_d = 0;
  world.input.frames()[2].plane_fallback = 0;
  fill_single_member_planes(world.input, 3);
  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  return world;
}

/// THE UNCUT NEIGHBOR. The equal-children world plus a third plane that
/// carries one of the two parents and NOTHING else — no crosser, no weld, no
/// statement of its own. Its whole reason to enter the wave is that a split
/// lands on an edge whose span reaches it, which is the ownership law's own
/// case: every plane carrying any dirty edge becomes local in the same wave,
/// whole block, triangulation kept.
template <typename Int>
auto make_plane_uncut_neighbor_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 20> coordinates{{
      {{0, 0, 0}},        {{100, 4, 0}},     {{100, -4, 0}},
      {{8, -1, -33}},     {{11, 1, 14}},     {{8, 1, -33}},
      {{11, -1, 14}},     {{30, -1, 0}},     {{-20, -20, -480}},
      {{120, -20, -620}}, {{120, 20, 380}},  {{-20, 20, 520}},
      {{-20, -20, 520}},  {{120, -20, 380}}, {{120, 20, -620}},
      {{-20, 20, -480}},  {{-20, -20, 0}},   {{120, -20, 0}},
      {{120, 20, 0}},     {{-20, 20, 0}},
  }};
  fill_plane_points(world.points, coordinates, Int(1));

  tf::buffer<plane_def_t> raw;
  const auto add = [&](std::array<plane_index_t, 2> from,
                       std::array<plane_index_t, 2> to, plane_index_t face,
                       std::int16_t ordinal, std::int16_t side, bool boundary) {
    std::uint8_t flags = boundary
                             ? tf::intersect::graph::plane_edge_whole_side_flag
                             : std::uint8_t(0);
    if (to < from) {
      std::swap(from, to);
      flags =
          std::uint8_t(flags | tf::intersect::graph::plane_edge_reversed_flag);
    }
    raw.push_back({from[1], to[1], -1, face, -1, std::int16_t(from[0]),
                   std::int16_t(to[0]), -1, ordinal, side, flags});
  };
  const auto add_boundary = [&](plane_index_t face, plane_index_t base) {
    add({0, base}, {0, base + 1}, face, 0, 0, true);
    add({0, base + 1}, {0, base + 2}, face, 1, 1, true);
    add({0, base + 2}, {0, base + 3}, face, 2, 2, true);
    add({0, base + 3}, {0, base}, face, 3, 3, true);
  };
  add_boundary(0, 8);
  add_boundary(1, 12);
  add_boundary(2, 16);
  add({0, 0}, {0, 1}, 0, -1, -1, false);
  add({0, 0}, {0, 1}, 2, -1, -1, false);
  add({0, 0}, {0, 2}, 1, -1, -1, false);
  add({0, 3}, {0, 4}, 0, -1, -1, false);
  add({0, 5}, {0, 6}, 1, -1, -1, false);
  fill_plane_tables(world.input, std::move(raw), 3);

  world.input.frames().allocate(3);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {1, -25, 1};
  world.input.frames()[0].plane_d = 0;
  world.input.frames()[0].plane_fallback = -480;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 1;
  world.input.frames()[1].plane_n = {1, 25, 1};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = 520;
  world.input.frames()[2].ax0 = 0;
  world.input.frames()[2].ax1 = 1;
  world.input.frames()[2].plane_n = {0, 0, 1};
  world.input.frames()[2].plane_d = 0;
  world.input.frames()[2].plane_fallback = 0;
  fill_single_member_planes(world.input, 3);
  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  return world;
}

/// THE MIXED TIER. The bent-child world plus a slender bystander far away on
/// its own lattice scale: the stock build leaves the two crossing planes local
/// and that one still reading the world table, so a refinement round must
/// reach both tiers at once.
template <typename Int>
auto make_plane_mixed_tier_world() -> plane_hand_world<Int> {
  plane_hand_world<Int> world;
  const std::array<std::array<int, 3>, 18> coordinates{{
      {{-20, -80, -15}},
      {{15, -80, 10}},
      {{27, 80, 22}},
      {{-8, 80, -3}},
      {{0, 0, 1}},
      {{7, 0, 6}},
      {{2, -20, 2}},
      {{5, 20, 5}},
      {{3, -9, 3}},
      {{3, -11, 3}},
      {{-30, -30, -30}},
      {{30, -30, 30}},
      {{30, 30, 30}},
      {{-30, 30, -30}},
      {{200, 200, 0}},
      {{600, 200, 0}},
      {{600, 240, 0}},
      {{200, 240, 0}},
  }};
  for (std::size_t point = 0; point < coordinates.size(); ++point) {
    tf::point<Int, 3> position;
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      position[int(coordinate)] =
          Int(coordinates[point][coordinate]) *
          (point < std::size_t(14) ? Int(1) : Int(plane_index_t(1) << 20));
    world.points.push_back(position);
  }

  const auto boundary = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  tf::buffer<plane_def_t> raw;
  raw.push_back({0, 1, -1, 0, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {0, 3, -1, 0, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({1, 2, -1, 0, -1, 0, 0, -1, 1, 1, boundary});
  raw.push_back({2, 3, -1, 0, -1, 0, 0, -1, 2, 2, boundary});
  raw.push_back({4, 5, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({6, 7, -1, 0, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({6, 7, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({8, 9, -1, 1, -1, 0, 0, -1, -1, -1, 0});
  raw.push_back({10, 11, -1, 1, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {10, 13, -1, 1, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({11, 12, -1, 1, -1, 0, 0, -1, 1, 1, boundary});
  raw.push_back({12, 13, -1, 1, -1, 0, 0, -1, 2, 2, boundary});
  raw.push_back({14, 15, -1, 2, -1, 0, 0, -1, 0, 0, boundary});
  raw.push_back(
      {14, 17, -1, 2, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)});
  raw.push_back({15, 16, -1, 2, -1, 0, 0, -1, 1, 1, boundary});
  raw.push_back({16, 17, -1, 2, -1, 0, 0, -1, 2, 2, boundary});
  fill_plane_tables(world.input, std::move(raw), 3);

  world.input.frames().allocate(3);
  world.input.frames()[0].ax0 = 0;
  world.input.frames()[0].ax1 = 1;
  world.input.frames()[0].plane_n = {-100, -3, 140};
  world.input.frames()[0].plane_d = 140;
  world.input.frames()[0].plane_fallback = -15;
  world.input.frames()[1].ax0 = 0;
  world.input.frames()[1].ax1 = 1;
  world.input.frames()[1].plane_n = {-1, 0, 1};
  world.input.frames()[1].plane_d = 0;
  world.input.frames()[1].plane_fallback = -30;
  world.input.frames()[2].ax0 = 0;
  world.input.frames()[2].ax1 = 1;
  world.input.frames()[2].plane_n = {0, 0, 1};
  world.input.frames()[2].plane_d = 0;
  world.input.frames()[2].plane_fallback = 0;
  fill_single_member_planes(world.input, 3);
  world.vertex_offsets.push_back(0);
  world.vertex_offsets.push_back(plane_index_t(world.points.size()));
  return world;
}

// =============================================================================
// The boundary-dominant crease
// =============================================================================

template <typename Index, typename Real> struct plane_crease_meshes {
  tf::polygons_buffer<Index, Real, 3, 3> crease;
  tf::polygons_buffer<Index, Real, 3, 3> cutter;
};

/// An open crease whose shared edge lies strictly inside the cutter face.
///
/// The edge is a base-loop side of both crease faces and an interior cut of
/// the cutter's plane, so all three definitions fuse onto one canonical root.
/// That root therefore expects one triangle slot on each crease plane and two
/// on the cutter plane: the boundary-dominant arm of the incidence rule.
template <typename Index, typename Real>
auto make_plane_crease_meshes() -> plane_crease_meshes<Index, Real> {
  plane_crease_meshes<Index, Real> out;
  const std::array<std::array<Real, 3>, 4> crease_points{{
      {{Real(-1), Real(0), Real(0)}},
      {{Real(1), Real(0), Real(0)}},
      {{Real(0), Real(2), Real(1)}},
      {{Real(0), Real(2), Real(-1)}},
  }};
  const std::array<std::array<Index, 3>, 2> crease_faces{{
      {{Index(0), Index(1), Index(2)}},
      {{Index(1), Index(0), Index(3)}},
  }};
  const std::array<std::array<Real, 3>, 3> cutter_points{{
      {{Real(-3), Real(-3), Real(0)}},
      {{Real(3), Real(-3), Real(0)}},
      {{Real(0), Real(3), Real(0)}},
  }};

  out.crease.points_buffer().allocate(crease_points.size());
  out.crease.faces_buffer().allocate(crease_faces.size());
  for (std::size_t point = 0; point < crease_points.size(); ++point)
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      out.crease.points()[point][coordinate] = crease_points[point][coordinate];
  for (std::size_t face = 0; face < crease_faces.size(); ++face)
    out.crease.faces()[face] = crease_faces[face];

  out.cutter.points_buffer().allocate(cutter_points.size());
  out.cutter.faces_buffer().allocate(1);
  for (std::size_t point = 0; point < cutter_points.size(); ++point)
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      out.cutter.points()[point][coordinate] = cutter_points[point][coordinate];
  out.cutter.faces()[0] = std::array<Index, 3>{{Index(0), Index(1), Index(2)}};
  return out;
}

} // namespace tf::test
