/**
 * @file test_plane_collision_parity.cpp
 * @brief Slot parity across a constraint collision whose two inputs belong to
 *        different canonical groups.
 *
 * One square face carries two interior cuts that are distinct in exact 3D and
 * identical in this plane's projection. The triangulation welds their
 * endpoints, both constraints claim one edge, and the two producers name
 * different members of that coincidence class: the plain CDT keeps the later
 * input's provenance, the refinement states its alias representative, and
 * tf::arrangement::union_plane_constraint_collisions maps the second name onto
 * the first. The piece a triangle slot publishes must be the same either way.
 *
 * Refinement is a no-op on this world on purpose (min_quality 0 with frozen
 * boundaries returns early), so the refined arm must reproduce the stock
 * product exactly.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_fixture_world.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/arrangement/planes/union_plane_constraint_collisions.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/topology/cdt_refine_config.hpp>
#include <trueform/topology/cdt_refiner.hpp>
#include <trueform/topology/cdt_region_mode.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {

using collision_index_t = int;
using collision_def_t = tf::intersect::graph::plane_edge_def<collision_index_t>;

template <typename Int> struct fixture_t {
  tf::test::plane_fixture_world<collision_index_t, Int> input;
  std::array<tf::point<Int, 3>, 8> points{};
  tf::buffer<collision_index_t> vertex_offsets;
};

/// One square face on z = 0 with two interior cuts that project onto the same
/// span and sit at different heights in exact 3D.
template <typename Int> auto make_fixture() -> fixture_t<Int> {
  fixture_t<Int> fixture;
  const std::array<std::array<int, 3>, 8> coordinates{{
      {{0, 0, 0}},
      {{100, 0, 0}},
      {{100, 100, 0}},
      {{0, 100, 0}},
      {{20, 20, 0}},
      {{80, 20, 0}},
      {{20, 20, 7}},
      {{80, 20, 7}},
  }};
  for (std::size_t point = 0; point < fixture.points.size(); ++point)
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      fixture.points[point][coordinate] =
          Int(coordinates[point][std::size_t(coordinate)]);

  const auto boundary = tf::intersect::graph::plane_edge_whole_side_flag;
  const auto reversed = tf::intersect::graph::plane_edge_reversed_flag;
  std::array<collision_def_t, 6> raw{{
      {0, 1, -1, 0, -1, 0, 0, -1, 0, 0, boundary},
      {1, 2, -1, 0, -1, 0, 0, -1, 1, 1, boundary},
      {2, 3, -1, 0, -1, 0, 0, -1, 2, 2, boundary},
      {0, 3, -1, 0, -1, 0, 0, -1, 3, 3, std::uint8_t(boundary | reversed)},
      {4, 5, -1, 0, -1, 0, 0, -1, -1, -1, 0},
      {6, 7, -1, 0, -1, 0, 0, -1, -1, -1, 0},
  }};
  std::sort(raw.begin(), raw.end(),
            [](const collision_def_t &x, const collision_def_t &y) {
              if (tf::intersect::graph::plane_def_key_less(x, y))
                return true;
              if (tf::intersect::graph::plane_def_key_less(y, x))
                return false;
              return std::tie(x.face, x.ordinal, x.side) <
                     std::tie(y.face, y.ordinal, y.side);
            });
  const auto same_key = [](const collision_def_t &x, const collision_def_t &y) {
    return !tf::intersect::graph::plane_def_key_less(x, y) &&
           !tf::intersect::graph::plane_def_key_less(y, x);
  };
  auto &tables = fixture.input.tables();
  tables.defs().allocate(raw.size());
  tables.def_offsets().push_back(0);
  collision_index_t group = -1;
  for (std::size_t row = 0; row < raw.size(); ++row) {
    if (row == 0 || !same_key(raw[row - 1], raw[row])) {
      if (row != 0)
        tables.def_offsets().push_back(collision_index_t(row));
      ++group;
    }
    raw[row].id = group;
    tables.defs()[row] = raw[row];
  }
  tables.def_offsets().push_back(collision_index_t(raw.size()));
  tables.n_canon() = group + 1;
  tables.edges().offsets_buffer().push_back(0);
  for (std::size_t row = 0; row < raw.size(); ++row)
    tables.edges().data_buffer().push_back(collision_index_t(row));
  tables.edges().offsets_buffer().push_back(
      collision_index_t(tables.edges().data_buffer().size()));

  fixture.input.frames().allocate(1);
  fixture.input.frames()[0].ax0 = 0;
  fixture.input.frames()[0].ax1 = 1;
  fixture.input.frames()[0].plane_n = {0, 0, 1};
  fixture.input.frames()[0].plane_d = 0;
  fixture.input.frames()[0].plane_fallback = 0;
  fixture.input.plane_members_data().offsets_buffer().allocate(2);
  fixture.input.plane_members_data().offsets_buffer()[0] = 0;
  fixture.input.plane_members_data().offsets_buffer()[1] = 1;
  fixture.input.plane_members_data().data_buffer().allocate(1);
  fixture.input.plane_members_data().data_buffer()[0] = 0;
  fixture.input.descriptor_data().allocate(1);
  fixture.input.descriptor_data()[0] = {0, 0};
  fixture.input.face_planes().allocate(1);
  fixture.input.face_planes()[0] = 0;
  fixture.input.face_orientations().allocate(1);
  fixture.input.face_orientations()[0] = 1;

  fixture.vertex_offsets.allocate(2);
  fixture.vertex_offsets[0] = 0;
  fixture.vertex_offsets[1] = collision_index_t(fixture.points.size());
  return fixture;
}

/// One emitted triangle, keyed by the exact positions of its corners so the
/// two arms compare on identity-independent ground, with the slot each edge
/// publishes.
using record_t = std::array<std::int64_t, 15>;

template <typename Int, typename Arrangement>
auto records_of(const Arrangement &arrangement, const fixture_t<Int> &fixture)
    -> std::vector<record_t> {
  const auto get_point = [&](std::int16_t, collision_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  const auto n_flat = arrangement.n_flat_points();
  const auto position = [&](collision_index_t flat) -> tf::point<Int, 3> {
    return flat < n_flat ? fixture.points[std::size_t(flat)]
                         : arrangement.resolve_created_point(
                               flat - n_flat, get_point, get_point);
  };
  std::vector<record_t> records;
  const auto triangles = arrangement.triangles();
  const auto slots = arrangement.slot_parents();
  for (std::size_t row = 0; row < triangles.size(); ++row) {
    std::array<std::array<std::int64_t, 3>, 3> corners{};
    for (int corner = 0; corner < 3; ++corner) {
      const auto point = position(triangles[row][std::size_t(corner)]);
      corners[std::size_t(corner)] = {std::int64_t(point[0]),
                                      std::int64_t(point[1]),
                                      std::int64_t(point[2])};
    }
    int start = 0;
    for (int corner = 1; corner < 3; ++corner)
      if (corners[std::size_t(corner)] < corners[std::size_t(start)])
        start = corner;
    record_t record{};
    for (int corner = 0; corner < 3; ++corner) {
      const auto &key = corners[std::size_t((start + corner) % 3)];
      for (int field = 0; field < 3; ++field)
        record[std::size_t(3 * corner + field)] = key[std::size_t(field)];
      record[std::size_t(9 + corner)] =
          slots[row][std::size_t((start + corner) % 3)];
    }
    record[12] = std::int64_t(arrangement.stacked()[row]);
    record[13] =
        arrangement.coplanar_of()[row] == collision_index_t(-1) ? 0 : 1;
    record[14] = 0;
    records.push_back(record);
  }
  std::sort(records.begin(), records.end());
  return records;
}

/// The square with the same side stated three times: three inputs claim two
/// spans, which is the collision the resolution rule owns.
struct collision_scene_t {
  std::size_t compared = 0;
  std::size_t raw_disagreements = 0;
  std::size_t resolved_disagreements = 0;
  std::size_t collisions = 0;
};

template <typename Int> auto run_collision_scene() -> collision_scene_t {
  using coord_t = Int;
  using plain_t =
      tf::constrained_delaunay_triangulator<collision_index_t, coord_t, Int>;
  using refined_t = tf::cdt_refiner<collision_index_t, coord_t, Int>;

  tf::points_buffer<coord_t, 2> points;
  points.push_back(tf::make_point(coord_t(0), coord_t(0)));
  points.push_back(tf::make_point(coord_t(1024), coord_t(0)));
  points.push_back(tf::make_point(coord_t(1024), coord_t(1024)));
  points.push_back(tf::make_point(coord_t(0), coord_t(1024)));
  const std::array<std::array<collision_index_t, 2>, 6> raw{
      {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}, {{0, 1}}, {{3, 2}}}};
  tf::buffer<collision_index_t> constraints;
  tf::buffer<char> is_boundary;
  for (const auto &edge : raw) {
    constraints.push_back(edge[0]);
    constraints.push_back(edge[1]);
    is_boundary.push_back(char(0));
  }
  const auto edges =
      tf::make_edges(tf::make_blocked_range<2>(tf::make_range(constraints)));

  plain_t plain;
  plain.always_track_constraint_owners();
  REQUIRE(plain.build(points.points(), edges, tf::make_range(is_boundary),
                      false, tf::cdt_region_mode::nesting));
  refined_t refiner;
  refiner.always_track_constraint_owners();
  tf::cdt_refine_config config;
  config.min_quality = 0.0f;
  config.split_encroached = false;
  REQUIRE(refiner.build(points.points(), edges, tf::make_range(is_boundary),
                        config, tf::cdt_region_mode::nesting));

  tf::buffer<collision_index_t> roots;
  tf::arrangement::union_plane_constraint_collisions(
      refiner.constraint_collisions(), collision_index_t(raw.size()), roots);

  collision_scene_t scene;
  scene.collisions = refiner.constraint_collisions().size();
  const auto plain_faces = plain.make_faces();
  const auto refined_faces = refiner.make_faces();
  for (std::size_t face = 0; face < std::size_t(refined_faces.size()); ++face) {
    const auto corners = refined_faces[face];
    const auto owners = refiner.face_constraint_owners(collision_index_t(face));
    for (int edge = 0; edge < 3; ++edge) {
      if (owners[std::size_t(edge)].input_id == collision_index_t(-1))
        continue;
      const collision_index_t a = corners[edge];
      const collision_index_t b = corners[(edge + 1) % 3];
      for (std::size_t other = 0; other < std::size_t(plain_faces.size());
           ++other) {
        const auto plain_corners = plain_faces[other];
        const auto plain_owners =
            plain.face_constraint_owners(collision_index_t(other));
        for (int plain_edge = 0; plain_edge < 3; ++plain_edge) {
          const collision_index_t c = plain_corners[plain_edge];
          const collision_index_t d = plain_corners[(plain_edge + 1) % 3];
          if (std::min(a, b) != std::min(c, d) ||
              std::max(a, b) != std::max(c, d))
            continue;
          const auto expected = plain_owners[std::size_t(plain_edge)].input_id;
          if (expected == collision_index_t(-1))
            continue;
          ++scene.compared;
          scene.raw_disagreements +=
              std::size_t(owners[std::size_t(edge)].input_id != expected);
          scene.resolved_disagreements += std::size_t(
              roots[std::size_t(owners[std::size_t(edge)].input_id)] !=
              expected);
        }
      }
    }
  }
  return scene;
}

} // namespace

TEMPLATE_TEST_CASE("plane constraint collisions: the resolution rule",
                   "[cut][planes][collision]", tf::exact::int32,
                   tf::exact::int64) {
  const auto scene = run_collision_scene<TestType>();

  REQUIRE(scene.collisions != 0);
  REQUIRE(scene.compared != 0);
  // the counterfactual the rule exists for: naming the refinement's input
  // verbatim disagrees with the plain triangulation
  REQUIRE(scene.raw_disagreements != 0);
  REQUIRE(scene.resolved_disagreements == 0);
}

TEMPLATE_TEST_CASE("plane constraint collisions: slot parity across the arms",
                   "[cut][planes][collision]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;

  auto fixture = make_fixture<Int>();
  const auto get_point = [&](std::int16_t, collision_index_t point) {
    return fixture.points[std::size_t(point)];
  };

  tf::arrangement::plane_arrangement<collision_index_t, Int> stock;
  stock.record_triangle_arrangement();
  stock.build(fixture.input, collision_index_t(0), get_point, get_point,
              fixture.vertex_offsets);
  tf::cdt_refine_config config;
  config.min_quality = 0.0f;
  config.split_encroached = false;
  tf::arrangement::plane_arrangement<collision_index_t, Int> refined;
  refined.record_triangle_arrangement();
  refined.build_refined(fixture.input, collision_index_t(0), get_point,
                        get_point, fixture.vertex_offsets, config);

  REQUIRE(stock.n_planes() == collision_index_t(1));
  REQUIRE(stock.triangles().size() != 0);
  REQUIRE(stock.failed().size() == 0);
  REQUIRE(refined.failed().size() == 0);
  REQUIRE(records_of(refined, fixture) == records_of(stock, fixture));
}
