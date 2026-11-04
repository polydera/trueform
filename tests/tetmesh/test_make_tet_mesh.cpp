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
#include <catch2/catch_test_macros.hpp>
#include <tetmesh_builders.hpp>
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
#include <trueform/tetmesh/entrance/merge_tet_facets.hpp>
#include <trueform/tetmesh/entrance/tet_entrance_budget.hpp>
#include <trueform/tetmesh/entrance/tet_entrance_rung_types.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>
#include <trueform/topology/cdt/dt3/tet_deep_lattice.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>

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
    const auto [tets, children, face_facets, report] =
        tf::test::build_tet_mesh_double(surface);
    for (const auto &tet : tets.tetrahedra())
      if (!(tm_volume6(tet[0], tet[1], tet[2], tet[3]) > 0))
        return false;
    if (report.inverted || report.flat)
      return false;
  }
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(surface);
  if (report.status != tf::tet_mesh_status::meshed || !tets.size() ||
      face_facets.size() != surface.faces().size() || !children.size() ||
      report.census.segments_refused || report.census.faces.refused ||
      report.input_open_edges || report.entrance_open_edges ||
      report.entrance_failed_planes)
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
  std::vector<std::vector<std::size_t>> faces_of(children.size());
  std::vector<char> cut(children.size(), 0);
  for (std::size_t f = 0; f < face_facets.size(); ++f) {
    if (!face_facets[f].size())
      return false;
    for (auto facet : face_facets[f]) {
      faces_of[std::size_t(facet)].push_back(f);
      cut[std::size_t(facet)] |= char(face_facets[f].size() > 1);
    }
  }
  double surface_area = 0, covered_area = 0;
  for (std::size_t f = 0; f < face_facets.size(); ++f)
    surface_area += std::sqrt(tm_dot(normal_of(f), normal_of(f)));
  for (std::size_t facet = 0; facet < children.size(); ++facet) {
    if (faces_of[facet].empty())
      return false;
    const auto normal = normal_of(faces_of[facet][0]);
    const double length = std::sqrt(tm_dot(normal, normal));
    double area = 0, expected_area = 0;
    for (const auto &child : children[facet])
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

TEST_CASE("make_tet_mesh: a shell inside a shell carves by winding",
          "[tetmesh]") {
  const auto outer = tf::make_box_mesh<int>(4.f, 4.f, 4.f);
  const auto inner = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  // Wound inward, the inner box is a cavity the mesh leaves out.
  CHECK(tm_holds(
      tf::concatenated(outer.polygons(), tm_reversed(inner).polygons()), 1e-6));
  // Wound outward, it is wound twice inside and kept once.
  const auto [tets, children, face_facets, report] = tf::test::build_tet_mesh(
      tf::concatenated(outer.polygons(), inner.polygons()));
  CHECK(report.status == tf::tet_mesh_status::meshed);
  double volume = 0;
  for (const auto &tet : tets.tetrahedra())
    volume += tm_volume6(tet[0], tet[1], tet[2], tet[3]) / 6;
  CHECK(std::abs(volume - 64) < 1e-4);
}

TEST_CASE("make_tet_mesh: an open surface is refused as open before any "
          "volume is built",
          "[tetmesh]") {
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh open;
  open.points_buffer() = box.points_buffer();
  for (std::size_t f = 0; f + 1 < box.faces().size(); ++f)
    open.faces_buffer().push_back(
        {box.faces()[f][0], box.faces()[f][1], box.faces()[f][2]});
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(open);
  CHECK(report.status == tf::tet_mesh_status::open);
  CHECK(report.input_open_edges == 3);
  CHECK(report.entrance_open_edges == 3);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: a surface wound inward encloses no cell positively",
          "[tetmesh]") {
  // The box's faces reversed wind its inside negatively: the carve is
  // consistent, the run is meshed, and the mesh is the empty set of cells
  // wound positively, which the report states by the cells wound negatively.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f);
  tm_mesh inward;
  inward.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    inward.faces_buffer().push_back({face[0], face[2], face[1]});
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(inward);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(tets.size() == 0);
  CHECK(report.negative_cells > 0);
  const auto [outward_tets, outward_children, outward_facets, outward] =
      tf::test::build_tet_mesh(box);
  CHECK(outward.status == tf::tet_mesh_status::meshed);
  CHECK(outward_tets.size() == report.negative_cells);
  CHECK(outward.negative_cells == 0);
}

TEST_CASE("make_tet_mesh: a sheet hanging off a solid is open at the entrance",
          "[tetmesh]") {
  // A closed box with one extra triangle standing inside it on one of its
  // edges: two of its edges are held by it alone, so the surface is open,
  // and the verdict is read before any volume is built.
  auto box = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  tm_mesh sheet;
  sheet.points_buffer() = box.points_buffer();
  for (const auto &face : box.faces())
    sheet.faces_buffer().push_back({face[0], face[1], face[2]});
  const auto first = box.faces()[0];
  sheet.points_buffer().push_back({0.25f, 0.25f, 0.25f});
  sheet.faces_buffer().push_back(
      {first[0], first[1], int(sheet.points().size()) - 1});
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(sheet);
  CHECK(report.status == tf::tet_mesh_status::open);
  CHECK(report.input_open_edges > 0);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: a face doubled on a solid is unwound at the entrance",
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
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(doubled);
  CHECK(report.status == tf::tet_mesh_status::unwound);
  CHECK(report.input_open_edges > 0);
  CHECK(tets.size() == 0);
}

TEST_CASE("make_tet_mesh: faces stacked three deep on one facet mesh",
          "[tetmesh]") {
  // The box's bottom face twice more, one wound against: the arrangement
  // stacks them on the box's own facets, and the carve sums the stack.
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
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(stacked);
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
  const auto report = std::get<3>(tf::test::build_tet_mesh(wall));
  CHECK(report.rung == 0);
  CHECK(report.frame_bits == 10);
  CHECK(report.deep_sites == 0);
}

TEST_CASE("make_tet_mesh: nested shells carve by their winding",
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
  // Two boxes overlapping in one surface: the entrance cuts every face at
  // the crossings, each face is covered by its pieces, and the cells wound
  // once or twice make the union.
  const auto first = tf::make_box_mesh<int>(2.f, 2.f, 2.f);
  const auto second =
      tm_moved(tf::make_box_mesh<int>(2.f, 2.f, 2.f), 0.5f, 0.75f, 1.25f);
  CHECK(tm_meshes(tf::concatenated(first.polygons(), second.polygons()),
                  16 - 1.5 * 1.25 * 0.75, 1e-6, 1e-6));
}

TEST_CASE("make_tet_mesh: the dragon meshes whole through the entrance",
          "[tetmesh][fixtures]") {
  // A scanned surface that is non-manifold and meets itself. It meshes on
  // the cheapest rung with every site on the frame. Its sites come back at
  // their lattice images, which a tiny cut face's covering area reads.
  const auto dragon = tf::read_stl(TF_TETMESH_DRAGON);
  REQUIRE(dragon.faces().size() == 58298);
  CHECK(tm_meshes(dragon, tm_enclosed(dragon), 1e-6, 1e-2));
  const auto report = std::get<3>(tf::test::build_tet_mesh(dragon));
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
  const auto report = std::get<3>(tf::test::build_tet_mesh(sphere));
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
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(surface);
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
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(surface);
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
  const auto report = std::get<3>(tf::test::build_tet_mesh(bunny));
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
  const auto [tets, children, face_facets, report] =
      tf::test::build_tet_mesh(layer);
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
  const auto report = std::get<3>(tf::test::build_tet_mesh(needle));
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
  const auto report = std::get<3>(tf::test::build_tet_mesh(gap));
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
  tf::tetmesh::merge_tet_facets<tf::exact::int64, int>(
      points, triangles, membership, link, kept, origin, facet_of, thrown,
      open_edges);
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
