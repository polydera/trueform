/**
 * @file test_pinched_loop_walls.cpp
 * @brief A face's boundary loop walking one wall twice, and the cover it owes.
 *
 * A base loop normally names each of its walls once. Two states make it walk
 * one twice, and it walks it the OTHER way when it does:
 *
 *  - a SPUR — a face whose boundary runs out to a point and straight back
 *    over the same segment, which is what a self-touching polygon becomes
 *    once its coincident corners are one identity. This needs no tolerance;
 *  - a CORNER THE BAND WELDS ACROSS — a sharp corner cut on both of its
 *    sides, whose two created identities fall inside the tolerance of each
 *    other while both stay far outside it from the corner. The weld is
 *    legitimate, and it pinches the ring exactly as a spur does.
 *
 * AB and BA are oppositely wound: two instances of one canonical group, and
 * groups fuse where instances never do. Reduce the pair to one row and the
 * carrier holds a boundary chain with a dead end — whose region walk then
 * covers less than the face, in silence.
 *
 * So the assertion is the COVER: a cut face's emitted triangles carry exactly
 * the face's own projected area, on the lattice, exactly. It is a ZERO-BAND
 * statement — above zero a weld MOVES the boundary and a welded ring no
 * longer owes the source area — so the welded scene is read for the fact a
 * weld may never destroy instead: a closed surface stays closed.
 *
 * Coordinates are integers, so the converter is the identity and the
 * tolerance is stated in lattice units.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/make_arrangement_graph.hpp>
#include <trueform/arrangement/make_arrangement_mesh.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/intersect/graph/vertex.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using pinched_walls_index_t = int;
using Coord = int;
using pinched_walls_int_t = tf::exact::resolve_int_type<tf::none_t, Coord>;
using T1 = typename tf::exact::meta<pinched_walls_int_t>::T1;
using T2 = typename tf::exact::meta<pinched_walls_int_t>::T2;
using pinched_walls_mesh_t =
    tf::polygons_buffer<pinched_walls_index_t, Coord, 3, 3>;
using ngon_t =
    tf::polygons_buffer<pinched_walls_index_t, Coord, 3, tf::dynamic_size>;
using vertex_t = tf::intersect::graph::vertex<pinched_walls_index_t>;
using vsource = tf::intersect::graph::vertex_source;

/// The welded scene's band, and the two lengths it must separate: the cut's
/// two created identities sit `weld_gap` apart on the wedge's two sides, and
/// each sits `corner_reach` from the corner between them.
constexpr double weld_tolerance = 32.0;
constexpr int weld_gap = 20;
constexpr int corner_reach = 500;

auto make_pinched_walls_mesh(
    const std::vector<std::array<Coord, 3>> &pts,
    const std::vector<std::array<pinched_walls_index_t, 3>> &faces)
    -> pinched_walls_mesh_t {
  pinched_walls_mesh_t mesh;
  mesh.points_buffer().allocate(pts.size());
  mesh.faces_buffer().allocate(faces.size());
  for (std::size_t i = 0; i < pts.size(); ++i)
    for (int d = 0; d < 3; ++d)
      mesh.points()[i][d] = pts[i][d];
  for (std::size_t i = 0; i < faces.size(); ++i)
    for (int d = 0; d < 3; ++d)
      mesh.faces()[i][d] = faces[i][d];
  return mesh;
}

auto make_ngon(const std::vector<std::array<Coord, 3>> &pts,
               const std::vector<std::vector<pinched_walls_index_t>> &faces)
    -> ngon_t {
  ngon_t mesh;
  for (const auto &p : pts)
    mesh.points_buffer().emplace_back(p[0], p[1], p[2]);
  for (const auto &face : faces) {
    tf::buffer<pinched_walls_index_t> ids;
    for (const auto v : face)
      ids.push_back(v);
    mesh.faces_buffer().push_back(tf::make_range(ids));
  }
  return mesh;
}

/// A stream vertex's lattice point: a created id indexes the unified table
/// directly, an original is a FLAT id over the per-tag vertex offsets.
template <typename Graph, typename MeshPoint>
auto stream_point(const Graph &graph, const MeshPoint &mesh_point,
                  const vertex_t &v) -> tf::point<pinched_walls_int_t, 3> {
  if (v.source == vsource::created)
    return graph.created_points()[std::size_t(v.id)];
  const pinched_walls_index_t tag = graph.tag_of_flat(v.id);
  return mesh_point(tag, v.id - graph.vertex_offsets()[std::size_t(tag)]);
}

/// How many cut faces do NOT carry their own projected area. Exact integers
/// in the face's own projection, so the reading has no scale: short is a
/// hole and long is an overlap.
template <typename Graph, typename FaceIds, typename MeshPoint>
auto uncovered_cut_faces(const Graph &graph, const FaceIds &face_ids,
                         const MeshPoint &mesh_point) -> int {
  const auto &global = graph.global();
  const auto descriptors = global.exposed_descriptors();
  const auto triangles = global.exposed_tris();
  int uncovered = 0;
  for (pinched_walls_index_t slot = 0;
       slot < pinched_walls_index_t(descriptors.size()); ++slot) {
    const auto &descriptor = descriptors[std::size_t(slot)];
    if (descriptor.plane == pinched_walls_index_t(-1))
      continue;
    const pinched_walls_index_t tag = pinched_walls_index_t(descriptor.tag);
    const auto face = face_ids(tag, descriptor.object);
    const auto axes = tf::exact::projection_axes(mesh_point(tag, face[0]),
                                                 mesh_point(tag, face[1]),
                                                 mesh_point(tag, face[2]));
    auto area2 = [&axes](const tf::point<pinched_walls_int_t, 3> &a,
                         const tf::point<pinched_walls_int_t, 3> &b,
                         const tf::point<pinched_walls_int_t, 3> &c) -> T2 {
      const T1 abx = T1(b[axes.first]) - T1(a[axes.first]);
      const T1 aby = T1(b[axes.second]) - T1(a[axes.second]);
      const T1 acx = T1(c[axes.first]) - T1(a[axes.first]);
      const T1 acy = T1(c[axes.second]) - T1(a[axes.second]);
      return T2(abx) * acy - T2(aby) * acx;
    };
    T2 owed(0);
    for (std::size_t c = 1; c + 1 < face.size(); ++c)
      owed += area2(mesh_point(tag, face[0]), mesh_point(tag, face[c]),
                    mesh_point(tag, face[c + 1]));
    const auto span = graph.slot_range(slot);
    T2 carried(0);
    for (pinched_walls_index_t e = span[0]; e < span[1]; ++e)
      carried += area2(
          stream_point(graph, mesh_point, triangles[std::size_t(e)][0]),
          stream_point(graph, mesh_point, triangles[std::size_t(e)][1]),
          stream_point(graph, mesh_point, triangles[std::size_t(e)][2]));
    uncovered += (owed < T2(0) ? -owed : owed) !=
                 (carried < T2(0) ? -carried : carried);
  }
  return uncovered;
}

struct reading {
  int uncovered = 0;
  bool closed = false;
  bool manifold = false;
  std::size_t failed = 0;
  std::vector<std::int64_t> mesh;
};

template <typename Mesh0, typename Mesh1>
auto read(const Mesh0 &m0, const Mesh1 &m1, double tolerance) -> reading {
  const auto config = tf::arrangement_config{tf::intersect_config{
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours,
      tolerance}};
  auto graph = tf::make_arrangement_graph(m0.polygons(), m1.polygons(), config);
  auto mesh_point =
      [&](pinched_walls_index_t tag,
          pinched_walls_index_t id) -> tf::point<pinched_walls_int_t, 3> {
    return tag == 0 ? graph.converter().convert(m0.points()[id])
                    : graph.converter().convert(m1.points()[id]);
  };
  auto face_ids =
      [&](pinched_walls_index_t tag,
          pinched_walls_index_t object) -> std::vector<pinched_walls_index_t> {
    std::vector<pinched_walls_index_t> ids;
    if (tag == 0)
      for (const auto v : m0.faces()[object])
        ids.push_back(v);
    else
      for (const auto v : m1.faces()[object])
        ids.push_back(v);
    return ids;
  };
  reading out;
  out.uncovered = uncovered_cut_faces(graph, face_ids, mesh_point);
  out.failed = graph.arrangement().failed().size();
  const auto arrangement =
      tf::make_arrangement_mesh<pinched_walls_int_t>(graph);
  out.closed = tf::is_closed(arrangement.polygons());
  out.manifold = tf::is_manifold(arrangement.polygons());
  for (const auto face : arrangement.faces())
    for (const auto v : face)
      out.mesh.push_back(std::int64_t(v));
  for (const auto point : arrangement.points())
    for (int d = 0; d < 3; ++d)
      out.mesh.push_back(std::int64_t(point[d]));
  return out;
}

/// A lobe whose boundary runs out to an INTERIOR point and straight back:
/// corner 1 is named twice, so the sides (1, 2) and (2, 1) are one wall
/// walked both ways. The spur sits inside the corner a blade clips off, so
/// the clipped piece is the one whose ring is pinched.
auto spur_lobe(int tip_x, int tip_y) -> ngon_t {
  return make_ngon(
      {{{0, 0, 0}}, {{1000, 0, 0}}, {{tip_x, tip_y, 0}}, {{0, 1000, 0}}},
      {{0, 1, 2, 1, 3}});
}

/// A blade clipping that lobe's corner 1 off, crossing both of its sides and
/// missing the spur.
auto clip_blade(int at) -> pinched_walls_mesh_t {
  return make_pinched_walls_mesh({{{at, -100, -500}},
                                  {{at, 1100, -500}},
                                  {{at, 1100, 500}},
                                  {{at, -100, 500}}},
                                 {{{0, 1, 2}}, {{0, 2, 3}}});
}

/// A closed wedge whose caps carry a corner one degree wide: a cut across it
/// at `corner_reach` from the corner lands the two created identities
/// `weld_gap` apart, so a band between those two lengths welds them and
/// nothing else.
auto sharp_wedge() -> pinched_walls_mesh_t {
  const Coord tip = 10000;
  const Coord width = (weld_gap * tip) / corner_reach;
  return make_pinched_walls_mesh({{{0, 0, 0}},
                                  {{tip, 0, 0}},
                                  {{0, width, 0}},
                                  {{0, 0, 400}},
                                  {{tip, 0, 400}},
                                  {{0, width, 400}}},
                                 {{{0, 2, 1}},
                                  {{3, 4, 5}},
                                  {{0, 1, 4}},
                                  {{0, 4, 3}},
                                  {{1, 2, 5}},
                                  {{1, 5, 4}},
                                  {{2, 0, 3}},
                                  {{2, 3, 5}}});
}

/// A box whose one wall crosses that wedge, `corner_reach` short of the tip.
auto corner_box() -> pinched_walls_mesh_t {
  const Coord x0 = 10000 - corner_reach, x1 = 20000;
  const Coord y0 = -1000, y1 = 1000, z0 = -1000, z1 = 1000;
  return make_pinched_walls_mesh({{{x0, y0, z0}},
                                  {{x1, y0, z0}},
                                  {{x1, y1, z0}},
                                  {{x0, y1, z0}},
                                  {{x0, y0, z1}},
                                  {{x1, y0, z1}},
                                  {{x1, y1, z1}},
                                  {{x0, y1, z1}}},
                                 {{{0, 2, 1}},
                                  {{0, 3, 2}},
                                  {{4, 5, 6}},
                                  {{4, 6, 7}},
                                  {{0, 1, 5}},
                                  {{0, 5, 4}},
                                  {{1, 2, 6}},
                                  {{1, 6, 5}},
                                  {{2, 3, 7}},
                                  {{2, 7, 6}},
                                  {{3, 0, 4}},
                                  {{3, 4, 7}}});
}

} // namespace

TEST_CASE("pinched loop walls: a spur is stated both ways",
          "[intersect][arrangement]") {
  // no tolerance is involved — the input's own boundary walks the wall twice
  const auto blade = clip_blade(400);
  for (const auto tip : {std::array<int, 2>{500, 300},
                         std::array<int, 2>{400, 300},
                         std::array<int, 2>{500, 400}}) {
    const auto lobe = spur_lobe(tip[0], tip[1]);
    const auto exact = read(lobe, blade, 0.0);
    CHECK(exact.failed == 0);
    CHECK(exact.uncovered == 0);
    const auto again = read(lobe, blade, 0.0);
    CHECK(again.mesh == exact.mesh);
  }
}

TEST_CASE("pinched loop walls: a corner the band welds across",
          "[intersect][arrangement][tolerance]") {
  const auto wedge = sharp_wedge();
  const auto box = corner_box();

  const auto exact = read(wedge, box, 0.0);
  CHECK(exact.failed == 0);
  CHECK(exact.uncovered == 0);
  CHECK(exact.closed);
  const auto again = read(wedge, box, 0.0);
  CHECK(again.mesh == exact.mesh);

  // The band spans the two created identities and reaches neither corner, so
  // the weld it makes is the corner pinch and nothing else. The cover is not
  // asked of it — a weld moves the boundary — but the surface may not open.
  const auto welded = read(wedge, box, weld_tolerance);
  CHECK(welded.failed == 0);
  CHECK(welded.closed);
}
