/*
 * Copyright (c) 2025 XLAB
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
#include "worker_ladder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <tetmesh_builders.hpp>
#include <trueform/core/constants.hpp>
#include <trueform/core/faces.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_cylinder_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/io/read_obj.hpp>
#include <trueform/io/read_stl.hpp>
#include <trueform/reindex/concatenated.hpp>
#include <trueform/tetmesh/entrance/count_tet_merged_open_edges.hpp>
#include <trueform/tetmesh/entrance/merge_tet_facets.hpp>
#include <trueform/tetmesh/entrance/tet_entrance_budget.hpp>
#include <trueform/tetmesh/entrance/tet_entrance_rung_types.hpp>
#include <trueform/tetmesh/entrance/tet_facet_workspace.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>
#include <trueform/topology/cdt/dt3/tet_deep_lattice.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/has_flaps.hpp>
#include <trueform/topology/has_holes.hpp>
#include <trueform/topology/manifold_edge_link.hpp>

#include <tbb/task_arena.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using tm_mesh = tf::polygons_buffer<int, float, 3, 3>;

/// Whether `T` is a complete type, an instantiation that is defined.
template <typename T, typename = void> struct tm_complete : std::false_type {};
template <typename T>
struct tm_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

/// Six times the signed volume of the tetrahedron `a, b, c, d`, in double.
template <typename P>
auto tm_volume6(const P &a, const P &b, const P &c, const P &d) -> double {
  const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1],
                       double(b[2]) - a[2]};
  const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1],
                       double(c[2]) - a[2]};
  const double w[3] = {double(d[0]) - a[0], double(d[1]) - a[1],
                       double(d[2]) - a[2]};
  return u[0] * (v[1] * w[2] - v[2] * w[1]) -
         u[1] * (v[0] * w[2] - v[2] * w[0]) +
         u[2] * (v[0] * w[1] - v[1] * w[0]);
}

/// The volume a closed triangle surface winds, by the divergence theorem.
template <typename Mesh> auto tm_enclosed(const Mesh &surface) -> double {
  const auto points = surface.points();
  const decltype(points[0]) origin = points[0];
  double sum = 0;
  for (const auto &face : surface.faces())
    sum +=
        tm_volume6(origin, points[std::size_t(face[0])],
                   points[std::size_t(face[1])], points[std::size_t(face[2])]);
  return sum / 6;
}

/// The cross product of `b - a` and `c - a`, in double.
template <typename P>
auto tm_cross(const P &a, const P &b, const P &c) -> std::array<double, 3> {
  return {(double(b[1]) - a[1]) * (double(c[2]) - a[2]) -
              (double(b[2]) - a[2]) * (double(c[1]) - a[1]),
          (double(b[2]) - a[2]) * (double(c[0]) - a[0]) -
              (double(b[0]) - a[0]) * (double(c[2]) - a[2]),
          (double(b[0]) - a[0]) * (double(c[1]) - a[1]) -
              (double(b[1]) - a[1]) * (double(c[0]) - a[0])};
}

auto tm_dot(const std::array<double, 3> &u, const std::array<double, 3> &v)
    -> double {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/// The mesher's contract on a closed surface: meshed with nothing refused,
/// their volume `expected`; every input face in at least one facet; every
/// facet covered by triangles wound as its faces, of their area within
/// `area_tolerance` where no crossing cut a face of it, and all facets
/// together of the surface's area; and, emitted in double, every
/// tetrahedron positive and none flat or inverted by the report.
auto tm_meshes(const tm_mesh &surface, double expected, double tolerance,
               double area_tolerance) -> bool {
  {
    const auto [tets, report, map] = tf::test::build_tet_mesh_double(surface);
    for (const auto &tet : tets.tetrahedra())
      if (!(tm_volume6(tet[0], tet[1], tet[2], tet[3]) > 0))
        return false;
    if (report.inverted || report.flat)
      return false;
  }
  const auto [tets, report, map] = tf::test::build_tet_mesh(surface);
  if (report.status != tf::tet_mesh_status::meshed || !tets.size() ||
      map.face_facets.size() != surface.faces().size() ||
      !map.facet_triangles.size() || report.refused_segments ||
      report.refused_faces || report.input_open_edges ||
      report.recovered_open_edges || report.failed_triangles)
    return false;
  double volume = 0;
  for (const auto &tet : tets.tetrahedra()) {
    const double v = tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
    volume += v;
  }
  if (std::abs(volume - expected) > tolerance * std::abs(expected))
    return false;
  const auto points = tets.points();
  const auto input = surface.points();
  const auto normal_of = [&surface, &input](std::size_t f) {
    const auto &face = surface.faces()[f];
    return tm_cross(input[std::size_t(face[0])], input[std::size_t(face[1])],
                    input[std::size_t(face[2])]);
  };
  std::vector<std::vector<std::size_t>> faces_of(map.facet_triangles.size());
  std::vector<char> cut(map.facet_triangles.size(), 0);
  for (std::size_t f = 0; f < map.face_facets.size(); ++f) {
    if (!map.face_facets[f].size())
      return false;
    for (auto facet : map.face_facets[f]) {
      faces_of[std::size_t(facet)].push_back(f);
      cut[std::size_t(facet)] |= char(map.face_facets[f].size() > 1);
    }
  }
  double surface_area = 0, covered_area = 0;
  for (std::size_t f = 0; f < map.face_facets.size(); ++f)
    surface_area += std::sqrt(tm_dot(normal_of(f), normal_of(f)));
  for (std::size_t facet = 0; facet < map.facet_triangles.size(); ++facet) {
    if (faces_of[facet].empty())
      return false;
    const auto normal = normal_of(faces_of[facet][0]);
    const double length = std::sqrt(tm_dot(normal, normal));
    double area = 0, expected_area = 0;
    for (const auto &child : map.facet_triangles[facet])
      area += tm_dot(tm_cross(points[std::size_t(child[0])],
                              points[std::size_t(child[1])],
                              points[std::size_t(child[2])]),
                     normal);
    for (auto f : faces_of[facet])
      expected_area += tm_dot(normal_of(f), normal);
    covered_area += area / length;
    if (!cut[facet] &&
        std::abs(area - expected_area) > area_tolerance * expected_area)
      return false;
  }
  return std::abs(covered_area - surface_area) <= area_tolerance * surface_area;
}

/// The contract on a closed surface whose volume is the one its winding
/// states by the divergence theorem.
auto tm_holds(const tm_mesh &surface, double tolerance) -> bool {
  return tm_meshes(surface, tm_enclosed(surface), tolerance, tolerance);
}

/// `surface` moved by `(x, y, z)`.
auto tm_moved(tm_mesh surface, float x, float y, float z) -> tm_mesh {
  for (auto point : surface.points()) {
    point[0] += x;
    point[1] += y;
    point[2] += z;
  }
  return surface;
}

/// `surface` with every face's winding reversed.
auto tm_reversed(tm_mesh surface) -> tm_mesh {
  for (auto face : surface.faces())
    std::swap(face[1], face[2]);
  return surface;
}

/// The four faces of the tetrahedron on `corners`, each wound outward,
/// appended to `out` over four points of its own.
auto tm_add_tet(tm_mesh &out, const std::array<std::array<int, 3>, 4> &corners)
    -> void {
  const int base = int(out.points().size());
  for (const auto &corner : corners)
    out.points_buffer().push_back(
        {float(corner[0]), float(corner[1]), float(corner[2])});
  const bool positive =
      tm_volume6(corners[0], corners[1], corners[2], corners[3]) > 0;
  const int faces[4][3] = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  for (const auto &face : faces)
    out.faces_buffer().push_back(
        positive
            ? std::array<int, 3>{base + face[0], base + face[1], base + face[2]}
            : std::array<int, 3>{base + face[0], base + face[2],
                                 base + face[1]});
}

} // namespace

TEST_CASE("make_tet_mesh: the rungs are as wide as their predicates admit",
          "[tetmesh]") {
  using tf::tetmesh::tet_entrance_budget;
  using tf::topology::cdt::dt3::tet_deep_lattice;
  STATIC_REQUIRE(tet_entrance_budget<tf::exact::int32, std::int64_t>::width ==
                 40);
  STATIC_REQUIRE(tet_entrance_budget<tf::exact::int64, std::int64_t>::width ==
                 63);
  STATIC_REQUIRE(
      tet_entrance_budget<tf::exact::int32, std::int64_t>::lattice_bits == 30);
  STATIC_REQUIRE(
      tet_entrance_budget<tf::exact::int64, std::int64_t>::lattice_bits == 53);
  // a site below the frame reaches the deepest frame the ladder reached:
  // float 30 + 10 + 23 on int64, double 53 + 10 + 20 on int64 in int128
  STATIC_REQUIRE(tet_deep_lattice<tf::exact::int32, std::int64_t>::cap == 23);
  STATIC_REQUIRE(tet_deep_lattice<tf::exact::int64, std::int64_t>::cap == 20);
}

TEST_CASE("make_tet_mesh: one instantiation runs per input type", "[tetmesh]") {
  // The two home rungs and no deeper one: depth is a site's, not a rung's.
  STATIC_REQUIRE(tm_complete<tf::tetmesh::tet_entrance_rung_types<0>>::value);
  STATIC_REQUIRE(tm_complete<tf::tetmesh::tet_entrance_rung_types<1>>::value);
  STATIC_REQUIRE(!tm_complete<tf::tetmesh::tet_entrance_rung_types<2>>::value);
}

TEST_CASE("make_tet_mesh: closed surfaces mesh to their own volume",
          "[tetmesh]") {
  CHECK(tm_holds(tf::make_box_mesh<int>(1.f, 2.f, 3.f), 1e-6));
  CHECK(tm_holds(tf::make_box_mesh<int>(1.f, 2.f, 3.f, 3, 4, 5), 1e-6));
  CHECK(tm_holds(tf::make_sphere_mesh<int>(1.f, 20, 30), 1e-5));
  CHECK(tm_holds(tf::make_cylinder_mesh<int>(1.f, 2.f, 40), 1e-5));
}

TEST_CASE("make_tet_mesh: a shell inside a shell is labeled by winding",
          "[tetmesh]") {
  const auto outer = tf::make_box_mesh<int>(4.f, 4.f, 4.f);
  const auto inner = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  // Wound inward, the inner box is a cavity the mesh leaves out.
  CHECK(tm_holds(
      tf::concatenated(outer.polygons(), tm_reversed(inner).polygons()), 1e-6));
  // Wound outward, it is wound twice inside and kept once.
  const auto [tets, report, map] = tf::test::build_tet_mesh(
      tf::concatenated(outer.polygons(), inner.polygons()));
  CHECK(report.status == tf::tet_mesh_status::meshed);
  double volume = 0;
  for (const auto &tet : tets.tetrahedra())
    volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
  CHECK(std::abs(volume - 64) < 1e-4);
}

TEST_CASE("make_tet_mesh: an open surface is refused as open on its "
          "recovered surface",
          "[tetmesh]") {
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh open;
  open.points_buffer() = box.points_buffer();
  for (std::size_t f = 0; f + 1 < box.faces().size(); ++f)
    open.faces_buffer().push_back(
        {box.faces()[f][0], box.faces()[f][1], box.faces()[f][2]});
  CHECK(tf::has_holes(open.polygons()));
  const auto [tets, report, map] = tf::test::build_tet_mesh(open);
  CHECK(report.status == tf::tet_mesh_status::open);
  CHECK(report.input_open_edges == 3);
  CHECK(report.recovered_open_edges == 3);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: a facet lying on itself is a stack of its layers",
          "[tetmesh][stack]") {
  // A hexagonal pyramid whose flat top is one fan winding one and a half
  // times about its centre, wound alike, closed by its cone and two walls:
  // half the top lies under two layers of the one facet, which its cover
  // counts from the rim and the region winding reads. And a box pierced by a
  // square tunnel, its top and bottom facets each a ring of one layer about
  // an inner rim. Both mesh their own volume as given and resolved, with no
  // drain.
  tm_mesh strip;
  strip.points_buffer().push_back({0.f, 0.f, 1.f});
  strip.points_buffer().push_back({0.f, 0.f, -1.f});
  for (int i = 0; i < 10; ++i) {
    const double angle = 2 * tf::pi<double> * (i % 6) / 6;
    strip.points_buffer().push_back(
        {float(std::cos(angle)), float(std::sin(angle)), 1.f});
  }
  for (int i = 0; i < 9; ++i) {
    strip.faces_buffer().push_back({0, 2 + i, 3 + i});
    strip.faces_buffer().push_back({1, 3 + i, 2 + i});
  }
  strip.faces_buffer().push_back({1, 2, 0});
  strip.faces_buffer().push_back({1, 0, 11});
  tm_mesh tunnel;
  const float corners[8][2] = {{-1, -1},     {1, -1},        {1, 1},
                               {-1, 1},      {-.25f, -.25f}, {.25f, -.25f},
                               {.25f, .25f}, {-.25f, .25f}};
  for (int z = 0; z < 2; ++z)
    for (const auto &q : corners)
      tunnel.points_buffer().push_back({q[0], q[1], z ? 1.f : -1.f});
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) % 4;
    tunnel.faces_buffer().push_back({8 + i, 8 + j, 12 + j});
    tunnel.faces_buffer().push_back({8 + i, 12 + j, 12 + i});
    tunnel.faces_buffer().push_back({i, 4 + j, j});
    tunnel.faces_buffer().push_back({i, 4 + i, 4 + j});
    tunnel.faces_buffer().push_back({i, j, 8 + j});
    tunnel.faces_buffer().push_back({i, 8 + j, 8 + i});
    tunnel.faces_buffer().push_back({4 + i, 12 + j, 4 + j});
    tunnel.faces_buffer().push_back({4 + i, 12 + i, 12 + j});
  }
  const double hexagon = 3 * std::sqrt(3.0) / 2;
  for (const auto &[surface, expected] :
       std::vector<std::pair<const tm_mesh *, double>>{
           {&strip, hexagon * 2 / 3}, {&tunnel, 8 - 0.5}}) {
    const auto given =
        tf::test::build_tet_mesh_double(*surface, tf::tet_mesh_config{false});
    const auto resolved = tf::test::build_tet_mesh(*surface);
    CHECK(std::get<1>(given).status == tf::tet_mesh_status::meshed);
    CHECK(std::get<1>(resolved).status == tf::tet_mesh_status::meshed);
    CHECK(std::get<1>(resolved).drains == 0);
    CHECK(std::get<1>(given).negative_cells == 0);
    double volume = 0;
    for (const auto &tet : std::get<0>(given).tetrahedra())
      volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
    CHECK(std::abs(volume - expected) < 1e-5);
  }
}

TEST_CASE("make_tet_mesh: a seam the build welds or a segment splits meshes "
          "whole",
          "[tetmesh]") {
  // A box as a triangle soup, every corner its own point, and a box whose
  // top is split at the middle of an edge its front side holds whole: the
  // entrance's triangles leave every seam open, the build welds the
  // coincident corners and the front's segment splits at the site standing
  // on it, so both mesh the unit volume as given and resolved.
  const auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh soup;
  for (const auto &face : box.faces()) {
    const int base = int(soup.points().size());
    for (std::size_t k = 0; k < 3; ++k) {
      const auto p = box.points()[std::size_t(face[k])];
      soup.points_buffer().push_back({p[0], p[1], p[2]});
    }
    soup.faces_buffer().push_back({base, base + 1, base + 2});
  }
  tm_mesh split;
  for (const auto &p : std::vector<std::array<float, 3>>{{0, 0, 0},
                                                         {1, 0, 0},
                                                         {1, 1, 0},
                                                         {0, 1, 0},
                                                         {0, 0, 1},
                                                         {1, 0, 1},
                                                         {1, 1, 1},
                                                         {0, 1, 1},
                                                         {0.5f, 0, 1}})
    split.points_buffer().push_back({p[0], p[1], p[2]});
  for (const auto &f : std::vector<std::array<int, 3>>{{0, 2, 1},
                                                       {0, 3, 2},
                                                       {4, 8, 7},
                                                       {8, 5, 6},
                                                       {8, 6, 7},
                                                       {0, 1, 5},
                                                       {0, 5, 4},
                                                       {3, 7, 6},
                                                       {3, 6, 2},
                                                       {0, 4, 7},
                                                       {0, 7, 3},
                                                       {1, 2, 6},
                                                       {1, 6, 5}})
    split.faces_buffer().push_back({f[0], f[1], f[2]});
  for (const auto *surface : {&soup, &split}) {
    CHECK(tf::has_holes(surface->polygons()));
    const auto given =
        tf::test::build_tet_mesh_double(*surface, tf::tet_mesh_config{false});
    const auto resolved = tf::test::build_tet_mesh(*surface);
    for (const auto *report : {&std::get<1>(given), &std::get<1>(resolved)}) {
      CHECK(report->status == tf::tet_mesh_status::meshed);
      CHECK(report->recovered_open_edges == 0);
    }
    double volume = 0;
    for (const auto &tet : std::get<0>(given).tetrahedra())
      volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
    CHECK(std::abs(volume - 1) < 1e-4);
  }
}

TEST_CASE("make_tet_mesh: a surface wound inward encloses no cell positively",
          "[tetmesh]") {
  // The box's faces reversed wind its inside negatively: the region labeling is
  // consistent, the run is meshed, and the mesh is the empty set of cells
  // wound positively, which the report states by the cells wound negatively.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f);
  tm_mesh inward;
  inward.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    inward.faces_buffer().push_back({face[0], face[2], face[1]});
  const auto [tets, report, map] = tf::test::build_tet_mesh(inward);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(tets.size() == 0);
  CHECK(report.negative_cells > 0);
  const auto [outward_tets, outward, outward_map] =
      tf::test::build_tet_mesh(box);
  CHECK(outward.status == tf::tet_mesh_status::meshed);
  CHECK(outward_tets.size() == report.negative_cells);
  CHECK(outward.negative_cells == 0);
}

TEST_CASE("make_tet_mesh: a sheet hanging off a solid is unwound on its "
          "recovered surface",
          "[tetmesh]") {
  // A closed box with one extra triangle standing inside it on one of its
  // edges: two of its edges are held by it alone, but the rim they form
  // hangs off the edge the box and the triangle share, so the surface has no
  // hole — it is a solid plus a sheet.
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh sheet;
  sheet.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    sheet.faces_buffer().push_back({face[0], face[1], face[2]});
  const auto first = box.faces()[0];
  sheet.points_buffer().push_back({0.25f, 0.25f, 0.25f});
  sheet.faces_buffer().push_back(
      {first[0], first[1], int(sheet.points().size()) - 1});
  CHECK_FALSE(tf::has_holes(sheet.polygons()));
  CHECK(tf::has_flaps(sheet.polygons()));
  const auto [tets, report, map] = tf::test::build_tet_mesh(sheet);
  CHECK(report.status == tf::tet_mesh_status::unwound);
  CHECK(report.input_open_edges > 0);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: a face doubled on a solid is unwound on its "
          "recovered surface",
          "[tetmesh]") {
  // A closed box with one face stated twice the same way: every edge is
  // held by two or three triangles and none by one, yet the doubled face's
  // edges run twice one way and once the other, so the surface is no cycle.
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh doubled;
  doubled.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    doubled.faces_buffer().push_back({face[0], face[1], face[2]});
  const auto first = box.faces()[0];
  doubled.faces_buffer().push_back({first[0], first[1], first[2]});
  const auto [tets, report, map] = tf::test::build_tet_mesh(doubled);
  CHECK(report.status == tf::tet_mesh_status::unwound);
  CHECK(report.input_open_edges > 0);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: faces stacked three deep on one facet mesh",
          "[tetmesh]") {
  // The box's bottom face twice more, one wound against: the arrangement
  // stacks them on the box's own facets, and the region labeling sums the
  // stack.
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh stacked;
  stacked.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    stacked.faces_buffer().push_back({face[0], face[1], face[2]});
  const auto f0 = box.faces()[0], f1 = box.faces()[1];
  stacked.faces_buffer().push_back({f0[0], f0[1], f0[2]});
  stacked.faces_buffer().push_back({f0[0], f0[2], f0[1]});
  stacked.faces_buffer().push_back({f1[0], f1[1], f1[2]});
  stacked.faces_buffer().push_back({f1[0], f1[2], f1[1]});
  const auto [tets, report, map] = tf::test::build_tet_mesh(stacked);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(tets.size() > 0);
  CHECK(report.refused_faces == 0);
}

TEST_CASE("make_tet_mesh: boxes on coplanar grids mesh whole",
          "[tetmesh][fixtures]") {
  // A box whose faces are fine coplanar grids, and two boxes sharing a
  // wall their grids cut differently: the wall's two windings cancel, so
  // the pair meshes as one solid of both volumes, on the cheapest rung with
  // every site on the frame. A grid's corners off the lattice stand at their
  // lattice images, which a face's area reads.
  const auto grid = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 7, 9, 11);
  CHECK(tm_meshes(grid, 6, 1e-6, 1e-4));
  const auto left =
      tm_moved(tf::make_box_mesh<int>(1.f, 1.f, 1.f, 3, 3, 3), -0.5f, 0.f, 0.f);
  const auto right =
      tm_moved(tf::make_box_mesh<int>(1.f, 1.f, 1.f, 4, 5, 4), 0.5f, 0.f, 0.f);
  const auto wall = tf::concatenated(left.polygons(), right.polygons());
  CHECK(tm_meshes(wall, 2, 1e-6, 1e-4));
  const auto report = std::get<1>(tf::test::build_tet_mesh(wall));
  CHECK(report.rung == 0);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 0);
}

TEST_CASE("make_tet_mesh: nested shells are labeled by their winding",
          "[tetmesh][fixtures]") {
  // A sphere holding a cavity holding an island: the solid between the
  // outer two shells and the island inside the innermost.
  const auto outer = tf::make_sphere_mesh<int>(3.f, 16, 24);
  const auto cavity = tm_reversed(tf::make_sphere_mesh<int>(2.f, 14, 20));
  const auto island = tf::make_sphere_mesh<int>(1.f, 12, 18);
  CHECK(tm_holds(
      tf::concatenated(outer.polygons(), cavity.polygons(), island.polygons()),
      1e-5));
}

TEST_CASE("make_tet_mesh: a self-intersecting surface meshes its union",
          "[tetmesh][fixtures]") {
  // Two boxes overlapping in one surface: the recovery names the crossings,
  // a drain cuts the faces along their curves on the standing complex, each
  // face is covered by its pieces, and the cells wound once or twice make the
  // union.
  const auto first = tf::make_box_mesh<int>(2.f, 2.f, 2.f);
  const auto second =
      tm_moved(tf::make_box_mesh<int>(2.f, 2.f, 2.f), 0.5f, 0.75f, 1.25f);
  CHECK(tm_meshes(tf::concatenated(first.polygons(), second.polygons()),
                  16 - 1.5 * 1.25 * 0.75, 1e-6, 1e-6));
}

TEST_CASE("make_tet_mesh: crossings within the quantum of each other give the "
          "lawful product of that quantum",
          "[tetmesh][cut]") {
  // Two tetrahedra meeting inside a box of half extent 2^28. The box sets the
  // entrance's converter scale to one, so the lattice IS the integers written
  // here and the whole meeting stands inside one quantum. The quantum is a
  // grid, so whatever it makes of the meeting is the result in that quantum:
  // a valid solid, or a refusal the report names — never an open surface.
  const auto box =
      tf::make_box_mesh<int>(float(1 << 29), float(1 << 29), float(1 << 29));
  tm_mesh scene;
  scene.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    scene.faces_buffer().push_back({face[0], face[1], face[2]});
  tm_add_tet(scene, {{{3, -5, -8}, {-5, -4, 7}, {-7, 6, 0}, {6, 0, 4}}});
  tm_add_tet(scene, {{{-6, 4, -2}, {-4, 6, 6}, {2, -4, -4}, {7, -2, 4}}});
  const auto lawful = [](const auto &tets, const auto &report) {
    if (report.status == tf::tet_mesh_status::meshed)
      return tets.size() > 0 && report.flat == 0 && report.inverted == 0 &&
             report.recovered_open_edges == 0;
    return report.status == tf::tet_mesh_status::refused &&
           report.refused_segments + report.refused_faces +
                   report.failed_triangles + report.pairs_unresolved >
               0;
  };
  const auto [tets, report, map] = tf::test::build_tet_mesh(scene);
  CHECK(lawful(tets, report));
  const auto [deep_tets, deep, deep_map] =
      tf::test::build_tet_mesh_on_int64(scene);
  CHECK(lawful(deep_tets, deep));
}

TEST_CASE("make_tet_mesh: the dragon meshes whole through the entrance",
          "[tetmesh][fixtures]") {
  // A scanned surface that is non-manifold and meets itself. It meshes on
  // the cheapest rung with every site on the frame. Its sites come back at
  // their lattice images, which a tiny cut face's covering area reads.
  const auto dragon = tf::read_stl(TF_TETMESH_DRAGON);
  REQUIRE(dragon.faces().size() == 58298);
  CHECK(tm_meshes(dragon, tm_enclosed(dragon), 1e-6, 1e-2));
  const auto report = std::get<1>(tf::test::build_tet_mesh(dragon));
  CHECK(report.rung == 0);
  CHECK(report.deep_sites == 0);
}

TEST_CASE("make_tet_mesh: a float export counts the slivers it inverts",
          "[tetmesh][fixtures]") {
  // A fine sphere's slivers stand below float's resolution: emitted in
  // float some read negative and the report counts them; emitted in double
  // every cell is positive.
  const auto sphere = tf::make_sphere_mesh<int>(1.f, 40, 60);
  CHECK(tm_meshes(sphere, tm_enclosed(sphere), 1e-5, 1e-4));
  const auto report = std::get<1>(tf::test::build_tet_mesh(sphere));
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.rung == 0);
  CHECK(report.flat == 0);
  CHECK(report.inverted > 0);
}

TEST_CASE("make_tet_mesh: a face its frame names no new point in meshes below "
          "the frame",
          "[tetmesh][fixtures]") {
  // A double slab whose top is a sheet of sites a hair off one plane, and
  // on the sheet's edge A B the tetrahedron A B C X, a sliver: C stands a
  // fifty-thousandth of the face's size off A X B, over its interior. The
  // Delaunay edge from C down to a sheet site pierces A X B a tenth of a
  // line off the nearest line of the face's lattice on the frame; the
  // point the face places nearest it leaves the edge standing and is named
  // again, a depth witness, so the face places its point below the frame,
  // and the surface meshes on its home rung.
  const std::vector<std::array<std::int64_t, 3>> points{
      {-499367466649, -322220134971761, 24270539854},
      {526773931900, 319932786400928, 177618100},
      {-159266897623707, 310254244008190, 190060368061018},
      {-869418400552853, 858949120173077, 1035568154571680},
      {205542652468575, -53849461912185, 24908610467},
      {-235478292374299, 224715918321893, 53504504994},
      {104351422999997, -285462786192683, 8826732823},
      {-155121440647737, 26879544623126, 61484759839},
      {111656344154124, 39464029368420, 39649347312},
      {-904602772641685, -924919817839637, 0},
      {904602772641685, -924919817839637, 0},
      {904602772641685, 924919817839637, 0},
      {-904602772641685, 924919817839637, 0},
      {-904602772641685, -924919817839637, -294669116243968},
      {904602772641685, -924919817839637, -294669116243968},
      {904602772641685, 924919817839637, -294669116243968},
      {-904602772641685, 924919817839637, -294669116243968}};
  const std::vector<std::array<int, 3>> triangles{
      {9, 5, 12},   {5, 9, 7},    {7, 1, 5},    {7, 9, 0},    {0, 1, 7},
      {9, 10, 0},   {5, 1, 12},   {6, 4, 8},    {8, 0, 6},    {8, 4, 1},
      {1, 0, 8},    {4, 6, 10},   {10, 11, 4},  {11, 1, 4},   {6, 0, 10},
      {11, 12, 1},  {13, 15, 14}, {13, 16, 15}, {9, 13, 14},  {9, 14, 10},
      {10, 14, 15}, {10, 15, 11}, {11, 15, 16}, {11, 16, 12}, {12, 16, 13},
      {12, 13, 9},  {0, 1, 2},    {0, 3, 1},    {1, 3, 2},    {2, 3, 0}};
  tf::polygons_buffer<int, double, 3, 3> surface;
  for (const auto &point : points)
    surface.points_buffer().push_back(
        {double(point[0]), double(point[1]), double(point[2])});
  for (const auto &triangle : triangles)
    surface.faces_buffer().push_back({triangle[0], triangle[1], triangle[2]});
  const auto [tets, report, map] = tf::test::build_tet_mesh(surface);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.rung == 1);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 1);
  CHECK(report.deepest == 1);
  CHECK(report.refused_faces == 0);
  double volume = 0;
  for (const auto &tet : tets.tetrahedra())
    volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
  const double enclosed = tm_enclosed(surface);
  CHECK(std::abs(volume - enclosed) <= 1e-9 * enclosed);
}

TEST_CASE("make_tet_mesh: a face reads its plane after its points move the "
          "sites",
          "[tetmesh][fixtures]") {
  // A double slab whose top is a sheet of sites a hair off one plane, and
  // the tetrahedron 2 5 4 12 standing on the sheet's edge 2 5. The first
  // point a face of it places moves the sites its plane is read from; the
  // face still reads the plane its corners state, and the surface meshes on
  // rung 1 with every site on the frame.
  const std::vector<std::array<std::int64_t, 3>> points{
      {-77272687312290, 338240804641866, 384899128691},
      {-74224341051158, -668119725429422, 23051637844},
      {2134151474042, 827460340568449, 197609979725},
      {1167285142524594, -841078814218669, -853221023154176},
      {-279791638243687, -141847854940041, 184129564768971},
      {-776204374559, -827884674685357, 96124714028},
      {-1167285142524594, -841078814218669, -853221023154176},
      {1167285142524594, -841078814218669, 0},
      {1167285142524594, 841078814218669, -853221023154176},
      {1167285142524594, 841078814218669, 0},
      {80042221395288, -330426176555190, 176334010083},
      {-1167285142524594, 841078814218669, 0},
      {1101314444858034, 458829173898116, 986505356984935},
      {202807320533305, -366000691462384, 228459347649},
      {-1167285142524594, -841078814218669, 0},
      {-1167285142524594, 841078814218669, -853221023154176}};
  const std::vector<std::array<int, 3>> triangles{
      {2, 12, 5},  {7, 8, 9},   {11, 2, 9},  {8, 3, 6},  {15, 6, 11},
      {1, 0, 14},  {2, 5, 10},  {2, 11, 0},  {5, 7, 13}, {2, 4, 12},
      {11, 14, 0}, {5, 2, 0},   {9, 2, 13},  {8, 6, 15}, {5, 14, 7},
      {2, 5, 4},   {8, 15, 9},  {11, 6, 14}, {14, 3, 7}, {9, 13, 7},
      {1, 14, 5},  {11, 9, 15}, {5, 12, 4},  {3, 8, 7},  {13, 10, 5},
      {2, 10, 13}, {5, 0, 1},   {3, 14, 6}};
  tf::polygons_buffer<int, double, 3, 3> surface;
  for (const auto &point : points)
    surface.points_buffer().push_back(
        {double(point[0]), double(point[1]), double(point[2])});
  for (const auto &triangle : triangles)
    surface.faces_buffer().push_back({triangle[0], triangle[1], triangle[2]});
  const auto [tets, report, map] = tf::test::build_tet_mesh(surface);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.rung == 1);
  CHECK(report.deep_sites == 0);
  CHECK(report.census.faces.points > 0);
  double volume = 0;
  for (const auto &tet : tets.tetrahedra())
    volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
  const double enclosed = tm_enclosed(surface);
  CHECK(std::abs(volume - enclosed) <= 1e-9 * enclosed);
}

TEST_CASE("make_tet_mesh: the Stanford bunny meshes whole",
          "[tetmesh][fixtures][local][.]") {
  // Local data. A scan with corners down to 0.0008 degrees: a piece is kept
  // once it is an edge, so the cheapest rung's frames hold every corner with
  // every site on the frame.
  std::ifstream probe(TF_TEST_DATA_DIR "/Stanford_Bunny.stl");
  if (!probe)
    SKIP("Stanford_Bunny.stl is not in TF_TEST_DATA_DIR");
  const auto bunny = tf::read_stl(TF_TEST_DATA_DIR "/Stanford_Bunny.stl");
  CHECK(tm_meshes(bunny, tm_enclosed(bunny), 1e-6, 1e-1));
  const auto report = std::get<1>(tf::test::build_tet_mesh(bunny));
  CHECK(report.rung == 0);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 0);
}

TEST_CASE("make_tet_mesh: a pit layer in double meshes on 64-bit sites",
          "[tetmesh][fixtures][local][.]") {
  // Local data. A piece is kept once it is an edge, so the 10-bit frames a
  // double leaves on 64-bit sites hold its corners with every site on them.
  std::ifstream probe(TF_TEST_DATA_DIR "/pit_layer1/layer1.obj");
  if (!probe)
    SKIP("pit_layer1/layer1.obj is not in TF_TEST_DATA_DIR");
  const auto layer =
      tf::read_obj<3, int, double>(TF_TEST_DATA_DIR "/pit_layer1/layer1.obj");
  const auto [tets, report, map] = tf::test::build_tet_mesh(layer);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.rung == 1);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 0);
}

TEST_CASE("make_tet_mesh: a needle's long edges are a facet's interior",
          "[tetmesh][fixtures][local][.]") {
  // Local data, Thingi10K 100344. Its long edges pass gaps far narrower
  // than themselves, which recovering them as segments split down to at any
  // width, until the site budget refused; they lie inside coplanar runs of
  // triangles, so they are facet interiors and are never recovered, and the
  // surface meshes.
  std::ifstream probe(TF_TEST_DATA_DIR "/thingi_100344.obj");
  if (!probe)
    SKIP("thingi_100344.obj is not in TF_TEST_DATA_DIR");
  const auto needle = tf::read_obj<3>(TF_TEST_DATA_DIR "/thingi_100344.obj");
  const auto report = std::get<1>(tf::test::build_tet_mesh(needle));
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.refused_budget == 0);
}

TEST_CASE("make_tet_mesh: an edge past a narrow gap meshes in bounded time",
          "[tetmesh][fixtures][local][.]") {
  // Local data, Thingi10K 225958. Edges between facets pass gaps far
  // narrower than themselves; the face points refining the faces beside the
  // gap break pieces the segments split again, and the surface meshes on
  // the cheapest rung's frame, no site below it.
  std::ifstream probe(TF_TEST_DATA_DIR "/thingi_225958.obj");
  if (!probe)
    SKIP("thingi_225958.obj is not in TF_TEST_DATA_DIR");
  const auto gap = tf::read_obj<3>(TF_TEST_DATA_DIR "/thingi_225958.obj");
  const auto start = std::chrono::steady_clock::now();
  const auto report = std::get<1>(tf::test::build_tet_mesh(gap));
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.rung == 0);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 0);
  CHECK(seconds < 120);
}

TEST_CASE("make_tet_mesh: a face split on one edge alone stays closed",
          "[tetmesh][entrance]") {
  // A tetrahedron whose face B D C is split at M and N on B C, closed by
  // the lines B M N and B N C. Merging throws the lines out and splits
  // A B C at M and N; the ear at A, its first corner, would leave B M N C
  // on one line with no ear left to cut, so the cut must start at B or C.
  const std::vector<std::array<std::int64_t, 3>> points{
      {0, 8, 0}, {0, 0, 0}, {9, 0, 0}, {0, 0, 8}, {3, 0, 0}, {6, 0, 0}};
  enum : int { A, B, C, D, M, N };
  const std::vector<std::array<int, 3>> triangles{
      {A, B, C}, {A, C, D}, {A, D, B}, {B, D, M},
      {M, D, N}, {N, D, C}, {B, M, N}, {B, N, C}};
  tf::buffer<std::array<int, 3>> kept;
  tf::buffer<int> origin, facet_of;
  std::array<std::size_t, 3> thrown{};
  tf::tetmesh::tet_open_edges open_edges{1, 1};
  tf::face_membership<int> membership;
  membership.build(tf::make_faces(triangles), points.size());
  tf::manifold_edge_link<int, 3> link;
  link.build(tf::make_faces(triangles), membership);
  tf::tetmesh::tet_facet_workspace<int, tf::exact::int64> work;
  const auto peers = [&link](std::size_t t) { return link[t]; };
  tf::tetmesh::merge_tet_facets<tf::exact::int64, int>(
      points, triangles, membership, peers, work, kept, origin, facet_of,
      thrown);
  open_edges = tf::tetmesh::count_tet_merged_open_edges(
      triangles, membership, peers, kept, thrown, work.steps);
  CHECK(open_edges.unbalanced == 0);
  CHECK(open_edges.boundary == 0);
  CHECK(thrown[1] == 2);
  CHECK(thrown[2] == 2);
  CHECK(kept.size() == 8);
  std::map<std::pair<int, int>, int> balance;
  for (const auto &face : kept)
    for (std::size_t k = 0; k < 3; ++k) {
      const int a = face[k], b = face[(k + 1) % 3];
      balance[{std::min(a, b), std::max(a, b)}] += a < b ? 1 : -1;
    }
  for (const auto &edge : balance)
    CHECK(edge.second == 0);
}

TEST_CASE("make_tet_mesh: a clean solid leaves its restoration to the region "
          "labeling",
          "[tetmesh][restore]") {
  // Phase S recovers a box's edges without rewriting the complex, so its
  // damage stands far below tet_execution_tuning::phase_restore_density and
  // the one restoration after the region labeling reads the cells the region
  // labeling keeps.
  const auto box = tf::make_box_mesh<int>(1.0f, 1.0f, 1.0f);
  const auto result = tf::test::build_tet_mesh(box);
  const auto &report = std::get<1>(result);
  REQUIRE(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.census.restored_before_faces == false);
  CHECK(std::get<0>(result).tetras().size() > 0);
}

TEST_CASE("make_tet_mesh: the placement a solid takes is one at every worker "
          "count",
          "[tetmesh][restore]") {
  // The density the placement is elected by is an integer count over a fixed
  // set, so it reads the same number at any width and the product it elects
  // is the same bytes.
  const auto sphere = tf::make_sphere_mesh<int>(1.0f, 12, 16);
  const std::size_t workers = tf::test::machine_workers();
  bool early = false;
  std::vector<std::array<int, 4>> first;
  for (const std::size_t width : tf::test::worker_ladder({1, 4, workers})) {
    tbb::task_arena arena(static_cast<int>(width));
    const auto result =
        arena.execute([&] { return tf::test::build_tet_mesh(sphere); });
    const auto &report = std::get<1>(result);
    REQUIRE(report.status == tf::tet_mesh_status::meshed);
    std::vector<std::array<int, 4>> cells;
    for (const auto &tet : std::get<0>(result).tetras())
      cells.push_back({tet[0], tet[1], tet[2], tet[3]});
    if (first.empty()) {
      first = cells;
      early = report.census.restored_before_faces;
      continue;
    }
    CHECK(report.census.restored_before_faces == early);
    CHECK(cells == first);
  }
}

TEST_CASE("make_tet_mesh: the cells a solid publishes stand in their rank",
          "[tetmesh][restore]") {
  // The restoration's flips write past the rows the region labeling read, and
  // the complex is published canonically after it, so the cells follow their
  // sorted corners and each is read from its smallest — the order the
  // tetrahedralizer states, whatever rows the flips stood in.
  const auto sphere = tf::make_sphere_mesh<int>(1.0f, 10, 14);
  const auto result = tf::test::build_tet_mesh(sphere);
  REQUIRE(std::get<1>(result).status == tf::tet_mesh_status::meshed);
  std::array<int, 4> previous{-1, -1, -1, -1};
  std::size_t read = 0;
  for (const auto &tet : std::get<0>(result).tetras()) {
    std::array<int, 4> key{tet[0], tet[1], tet[2], tet[3]};
    CHECK(key[0] == *std::min_element(key.begin(), key.end()));
    std::sort(key.begin(), key.end());
    if (read)
      CHECK_FALSE(key < previous);
    previous = key;
    ++read;
  }
  CHECK(read > 0);
}

TEST_CASE("make_tet_mesh: a densely rewritten solid restores before its faces",
          "[tetmesh][fixtures][local][.]") {
  // Local data, Thingi10K 1001930. Its segment recovery rewrites the complex
  // 2.29 times over, above tet_execution_tuning::phase_restore_density, so
  // phase S restores its own rows before phase F reads them; one restoration
  // after the region labeling costs it 35% more. The smallest scene that
  // reaches the bound is a real mesh: a solid a phase recovers cleanly stands
  // near zero, and no synthetic scene of five to eight sites rewrites the
  // complex twice over.
  std::ifstream probe(TF_TEST_DATA_DIR "/thingi_1001930.obj");
  if (!probe)
    SKIP("thingi_1001930.obj is not in TF_TEST_DATA_DIR");
  const auto dense = tf::read_obj<3>(TF_TEST_DATA_DIR "/thingi_1001930.obj");
  const auto report = std::get<1>(tf::test::build_tet_mesh(dense));
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.census.restored_before_faces == true);
}
