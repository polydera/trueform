/**
 * @file test_plane_support.cpp
 * @brief The points a carrier stands on, and the readers that consume them.
 *
 * One scan answers "which three offered points decide this plane": the first,
 * the first distinct from it, and the first off their line. The scenes here
 * offer it degenerate leading points from both shapes that read it — the
 * index-shaped face reader and the offer-shaped frame — and then check the
 * consequence that made the scan one producer: a face whose first two corners
 * are ONE lattice point still names its plane, so it still pools with the
 * same-tag face it shares an edge with.
 *
 * The naming half is the identity: a carrier's plane is the NAME it states,
 * so the scenes state which faces share one and which do not — whether or
 * not an edge joins them, since a name knows nothing of connectivity — and
 * that a face with no plane names none.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/exact/canonical_plane.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/make_supported_plane_frame.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/name_plane_carriers.hpp>
#include <trueform/intersect/graph/plane_face_support.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using support_index_t = int;
using support_int_t = tf::exact::int32;
using pt3_t = tf::exact::pt3<support_int_t>;
using support_mesh_t =
    tf::polygons_buffer<support_index_t, int, 3, tf::dynamic_size>;

/// One tag's points and faces, as the plane graph offers them.
struct scene_t {
  std::vector<pt3_t> points;
  std::vector<std::vector<support_index_t>> faces;
};

auto face_support(const scene_t &scene, std::size_t object, pt3_t &a, pt3_t &b,
                  pt3_t &c) -> bool {
  const auto &face = scene.faces[object];
  return tf::intersect::graph::plane_face_support<support_int_t>(
      tf::make_range(face),
      [&](std::size_t corner) {
        return scene.points[std::size_t(face[corner])];
      },
      a, b, c);
}

auto make_support_mesh(const scene_t &scene) -> support_mesh_t {
  support_mesh_t mesh;
  for (const auto &p : scene.points)
    mesh.points_buffer().emplace_back(p[0], p[1], p[2]);
  tf::buffer<support_index_t> face;
  for (const auto &corners : scene.faces) {
    face.clear();
    for (const auto corner : corners)
      face.push_back(corner);
    mesh.faces_buffer().push_back(tf::make_range(face));
  }
  return mesh;
}

/// The scene as the walk reads it: the source mesh's own connectivity plus
/// the lattice points the graph would hand it.
struct support_fixture {
  support_mesh_t mesh;
  decltype(tf::make_face_membership(
      std::declval<support_mesh_t &>().polygons())) fm;
  decltype(tf::make_manifold_edge_link(
      std::declval<support_mesh_t &>().polygons())) mel;

  explicit support_fixture(const scene_t &scene)
      : mesh(make_support_mesh(scene)),
        fm(tf::make_face_membership(mesh.polygons())),
        mel(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() { return mesh.polygons() | tf::tag(fm) | tf::tag(mel); }
};

/// The plane id of every face of the scene, and how many planes the
/// naming found.
struct naming_t {
  std::vector<support_index_t> plane_of;
  support_index_t n_planes = 0;
};

auto named(const scene_t &scene) -> naming_t {
  support_fixture f(scene);
  tf::buffer<tf::intersect::graph::face_descriptor<support_index_t>>
      descriptors;
  descriptors.allocate(scene.faces.size());
  for (std::size_t g = 0; g < scene.faces.size(); ++g)
    descriptors[g] = {support_index_t(0), support_index_t(g)};

  const auto form = f.form();
  const auto apply_to_form = [&form](support_index_t, const auto &apply) {
    apply(form);
  };
  const auto get_point = [&scene](std::int16_t, support_index_t id) {
    return scene.points[std::size_t(id)];
  };

  tf::buffer<support_index_t> plane_of;
  support_index_t n_planes = 0;
  tf::intersect::graph::name_plane_carriers<support_index_t, support_int_t>(
      descriptors, apply_to_form, get_point, plane_of, n_planes);

  naming_t out;
  out.n_planes = n_planes;
  for (const auto id : plane_of)
    out.plane_of.push_back(id);
  return out;
}

auto shares_plane(const scene_t &scene, std::size_t a, std::size_t b) -> bool {
  const auto n = named(scene);
  return n.plane_of[a] == n.plane_of[b];
}

/// The plane z = 0, two triangles across the edge (1, 2).
auto connected_scene() -> scene_t {
  return {{pt3_t{0, 0, 0}, pt3_t{4, 0, 0}, pt3_t{0, 4, 0}, pt3_t{4, 4, 0}},
          {{0, 1, 2}, {1, 3, 2}}};
}

/// The plane z = 0, with a second face floating inside the first — one
/// plane, no edge between them.
auto disjoint_scene() -> scene_t {
  return {{pt3_t{0, 0, 0}, pt3_t{9, 0, 0}, pt3_t{0, 9, 0}, pt3_t{1, 1, 0},
           pt3_t{5, 1, 0}, pt3_t{1, 5, 0}},
          {{0, 1, 2}, {3, 4, 5}}};
}

/// Two triangles across an edge, the second lifted `rise` units off the
/// first's plane at its far corner.
auto tilted_scene(int rise) -> scene_t {
  return {{pt3_t{0, 0, 0}, pt3_t{1000, 0, 0}, pt3_t{0, 1000, 0},
           pt3_t{1000, 1000, rise}},
          {{0, 1, 2}, {1, 3, 2}}};
}

} // namespace

TEST_CASE("plane support: a repeated leading corner does not hide the plane",
          "[intersect][graph]") {
  auto scene = connected_scene();
  // the second face names its first corner twice
  scene.faces[1] = {1, 1, 3, 2};

  pt3_t a{}, b{}, c{};
  REQUIRE(face_support(scene, 1, a, b, c));
  CHECK(a == scene.points[1]);
  CHECK(b == scene.points[3]);
  CHECK(c == scene.points[2]);
}

TEST_CASE("plane support: a face with no plane states the line it collapsed to",
          "[intersect][graph]") {
  const scene_t scene{
      {pt3_t{0, 0, 0}, pt3_t{0, 0, 0}, pt3_t{6, 0, 0}, pt3_t{9, 0, 0}},
      {{0, 1, 2, 3}}};

  pt3_t a{}, b{}, c{};
  CHECK(!face_support(scene, 0, a, b, c));
  CHECK(a == scene.points[0]);
  CHECK(b == scene.points[2]);
  CHECK(c == b);
}

TEST_CASE("plane support: a line carrier's frame projects along the line",
          "[intersect][exact]") {
  const std::array<pt3_t, 3> line{pt3_t{0, 0, 0}, pt3_t{0, 0, 7},
                                  pt3_t{0, 0, 3}};
  const auto frame = tf::exact::make_supported_plane_frame<support_int_t>(
      [&](const auto &consider) {
        for (const auto &point : line)
          consider(point);
      });
  using T2 = tf::exact::meta<support_int_t>::T2;
  CHECK(frame.plane_n[0] == T2(0));
  CHECK(frame.plane_n[1] == T2(0));
  CHECK(frame.plane_n[2] == T2(0));
  // z dominates the direction, so the projection keeps it
  CHECK(frame.ax0 == 0);
  CHECK(frame.ax1 == 2);
}

TEST_CASE("plane naming: coplanar same-tag faces name one plane",
          "[intersect][graph]") {
  SECTION("plainly wound") {
    const auto n = named(connected_scene());
    CHECK(n.n_planes == 1);
    CHECK(n.plane_of[0] == n.plane_of[1]);
  }

  SECTION("one of them repeats its first corner") {
    auto scene = connected_scene();
    scene.faces[1] = {1, 1, 3, 2};
    CHECK(shares_plane(scene, 0, 1));
  }

  SECTION("opposite windings still name one plane") {
    auto scene = connected_scene();
    // the peer reversed: the name is orientation-free, so it is the same
    scene.faces[1] = {2, 3, 1};
    CHECK(shares_plane(scene, 0, 1));
  }

  SECTION("a face across the edge on another plane names another") {
    auto scene = connected_scene();
    scene.points[3][2] = 1;
    CHECK(!shares_plane(scene, 0, 1));
  }
}

TEST_CASE("plane naming: connectivity is not the identity",
          "[intersect][graph]") {
  // one plane, no edge between the two faces: a NAME knows nothing of
  // connectivity, so both carriers are the same plane
  const auto n = named(disjoint_scene());
  CHECK(n.n_planes == 1);
  CHECK(n.plane_of[0] == n.plane_of[1]);
}

TEST_CASE("plane naming: a face off the plane is another carrier",
          "[intersect][graph]") {
  // the peer's far corner stands off the first's plane, so the two faces
  // name two planes however small the rise — the identity is exact
  CHECK(!shares_plane(tilted_scene(3), 0, 1));
  CHECK(!shares_plane(tilted_scene(1), 0, 1));
}

TEST_CASE("plane naming: a face with no plane names no plane",
          "[intersect][graph]") {
  auto scene = connected_scene();
  // the peer collapses onto the line through 1 and 2
  scene.points[3] = pt3_t{2, 2, 0};
  const auto n = named(scene);
  CHECK(n.n_planes == 2);
  CHECK(n.plane_of[0] != n.plane_of[1]);
}
