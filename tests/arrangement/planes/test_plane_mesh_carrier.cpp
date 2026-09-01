/**
 * @file test_plane_mesh_carrier.cpp
 * @brief One face, one plane carrier: the mesh as a world of the plane tier
 *
 * The plane arrangement's closed-world entry takes a prepared world and
 * nothing from a local arrangement. A MESH is such a world — one face is one
 * plane carrier, its own boundary is its constraint set, and a shared mesh
 * edge is one canonical group. The laws read here are per face, from the
 * published product only:
 *
 *   C   THE COUNT — a simple loop of n corners triangulates into n - 2
 *       triangles, and no carrier refuses.
 *   P   THE POINT LAW — every corner is one of the face's own input ids: the
 *       tier creates nothing for a mesh that needs no resolution.
 *   A   THE AREA LAW — the emitted triangles sum, exactly on the lattice, to
 *       the face's own signed area in the face's own frame.
 *   W   THE WINDING LAW — every emitted triangle turns the way its face does.
 *
 * The diagonal is deliberately NOT pinned: two triangulators of one simple
 * loop may cut it differently and both be right.
 *
 * The instrument is proven able to fail: a face whose loop crosses itself has
 * no such triangulation, so the count law must break on it.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_mesh_fixture.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient2d.hpp>
#include <trueform/exact/signed_area.hpp>
#include <trueform/exact/vertex.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using Index = tf::test::plane_index_t;

/// The real type each lattice width is quantized from.
template <typename Int> struct mesh_carrier_real_of;
template <> struct mesh_carrier_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct mesh_carrier_real_of<tf::exact::int64> {
  using type = double;
};

struct mesh_report_t {
  std::size_t faces = 0;
  std::size_t triangles = 0;
  std::size_t failed = 0;
  std::size_t created = 0;
  std::size_t foreign_corners = 0;
  std::size_t count_defects = 0;
  std::size_t corner_defects = 0;
  std::size_t area_defects = 0;
  std::size_t winding_defects = 0;
};

auto sorted_unique(std::vector<Index> ids) -> std::vector<Index> {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

/// The closed-world build of a prepared mesh world, read face by face. The
/// lattice the identities name is the world's own.
template <typename Int, typename World, typename Policy>
auto evaluate_mesh_world(World &world, const tf::buffer<Index> &vertex_offsets,
                         const tf::polygons<Policy> &polygons)
    -> mesh_report_t {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto get_point = [&](std::int16_t, Index id) {
    return world.point(id);
  };
  tf::arrangement::plane_arrangement<Index, Int> product;
  product.build(world, Index(0), get_point, get_point, vertex_offsets);

  const auto magnitude = [](T2 value) {
    return value < T2(0) ? -value : value;
  };
  const auto n_points = world.n_points();
  mesh_report_t report;
  report.faces = polygons.faces().size();
  report.triangles = product.triangles().size();
  report.failed = product.failed().size();
  report.created = std::size_t(product.n_created());
  for (Index f = 0; f < Index(polygons.faces().size()); ++f) {
    const auto face = polygons.faces()[std::size_t(f)];
    const auto span = product.face_range(f);
    const auto count = std::size_t(span[1] - span[0]);
    if (count + 2 != face.size())
      ++report.count_defects;

    const auto frame = world.frame(world.plane_of_face(f));
    const auto at = [&](Index flat) -> tf::exact::pt2<Int> {
      const auto q = world.point(flat);
      return {q[frame.ax0], q[frame.ax1]};
    };
    const auto winding = int(world.face_orientation(f));

    std::vector<Index> emitted;
    T2 area = 0;
    for (Index t = span[0]; t < span[1]; ++t) {
      const auto &triangle = product.triangles()[std::size_t(t)];
      bool own = true;
      for (const auto corner : triangle) {
        if (corner >= n_points) {
          ++report.foreign_corners;
          own = false;
          continue;
        }
        emitted.push_back(corner);
      }
      if (!own)
        continue;
      const auto turn = tf::exact::orient2d(at(triangle[0]), at(triangle[1]),
                                           at(triangle[2]));
      area += turn;
      if (turn != T2(0) && ((turn > T2(0)) != (winding > 0)))
        ++report.winding_defects;
    }
    const auto face_area = tf::exact::signed_area_2x(
        face, [&](auto corner) { return at(Index(corner)); });
    if (magnitude(area) != magnitude(face_area))
      ++report.area_defects;

    std::vector<Index> own_ids;
    for (const auto corner : face)
      own_ids.push_back(Index(corner));
    if (sorted_unique(std::move(emitted)) != sorted_unique(std::move(own_ids)))
      ++report.corner_defects;
  }
  return report;
}

template <typename Int, typename Policy>
auto check_mesh(const tf::polygons<Policy> &polygons) -> mesh_report_t {
  auto fixture = tf::test::make_plane_mesh_fixture<Int>(polygons);
  return evaluate_mesh_world<Int>(fixture.input, fixture.vertex_offsets,
                                  polygons);
}

/// This tier's own laws, read against the face itself. A set of same-winding
/// triangles over the face's own corners whose signed areas sum EXACTLY to the
/// face's own is a triangulation of that face: overlapping would sum to more,
/// a gap to less.
auto expect_own_laws(const mesh_report_t &report) -> void {
  CHECK(report.failed == 0);
  CHECK(report.created == 0);
  CHECK(report.foreign_corners == 0);
  CHECK(report.count_defects == 0);
  CHECK(report.corner_defects == 0);
  CHECK(report.area_defects == 0);
  CHECK(report.winding_defects == 0);
}

} // namespace

TEMPLATE_TEST_CASE("plane mesh carrier: a convex face is its own world",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto quad = tf::test::make_mesh_convex_quad<Real>();
  const auto report = check_mesh<Int>(quad.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: the frame is the face's own",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto tilted = tf::test::make_mesh_tilted_quad<Real>();
  const auto report = check_mesh<Int>(tilted.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: a reflex corner is not a refusal",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto shape = tf::test::make_mesh_non_convex_l<Real>();
  const auto report = check_mesh<Int>(shape.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 4);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: a collinear leading run is not a line",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto shape = tf::test::make_mesh_collinear_run<Real>();
  const auto report = check_mesh<Int>(shape.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 4);
  // this tier scans for its supporting triple, so the carrier bounds area and
  // states the face's whole triangulation
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: a shared edge is one canonical group",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_shared_edge<Real>();
  auto fixture = tf::test::make_plane_mesh_fixture<Int>(mesh.polygons());
  // the tables are the price of resolution: a world states none until it
  // needs one, and this asks it to state them so the group space can be read
  CHECK(!fixture.input.materialized());
  fixture.input.materialize();
  // four vertices, five distinct mesh edges, one of them carried twice
  CHECK(fixture.input.n_canon() == 5);
  CHECK(fixture.input.edge_defs().size() == 6);
  CHECK(fixture.input.canon_group(0).size() == 2);
  const auto report = evaluate_mesh_world<Int>(
      fixture.input, fixture.vertex_offsets, mesh.polygons());
  CHECK(report.faces == 2);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: a closed box is every face at once",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_box<Real>();
  const auto report = check_mesh<Int>(mesh.polygons());
  CHECK(report.faces == 12);
  CHECK(report.triangles == 12);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("plane mesh carrier: the instrument can fail",
                   "[arrangement][planes][mesh_carrier]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_carrier_real_of<Int>::type;
  auto bow_tie = tf::test::make_mesh_bow_tie<Real>();
  const auto report = check_mesh<Int>(bow_tie.polygons());
  // a crossing loop has no triangulation on its own boundary: whatever the
  // tier answers, it is not the face's own area over the face's own points
  CHECK(report.area_defects == 1);
  CHECK(report.created + report.failed > 0);
}
