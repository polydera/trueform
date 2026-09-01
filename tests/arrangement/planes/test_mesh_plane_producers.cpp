/**
 * @file test_mesh_plane_producers.cpp
 * @brief The producers a mesh world stands on, one at a time
 *
 * `build_mesh_plane_defs` states the definitions at whole-mesh grain; the
 * frame and the winding are the world's own statement per carrier, read off
 * the face a plane IS. What each owes:
 *
 *   defs         — endpoints in key order, the reversed flag recovering loop
 *                  order, one row per corner in the face's own slice, and
 *                  `id` the emission row the canonicalization rewrites;
 *   canon        — a canonical group IS a mesh edge, and the plane CSR of a
 *                  one-face carrier is that face's own rows, ascending;
 *   frames       — non-degenerate wherever three corners are not collinear,
 *                  including when the LEADING three are;
 *   orientations — the face's own signed-area sign in that frame.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_mesh_fixture.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/mesh/build_mesh_plane_defs.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/plane_frame.hpp>
#include <trueform/exact/signed_area.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/exact/vertex_converter.hpp>
#include <trueform/intersect/graph/build_plane_edges.hpp>
#include <trueform/intersect/graph/canonicalize_plane_edge_defs.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

namespace {

using producers_index_t = tf::test::plane_index_t;
using producers_def_t = tf::intersect::graph::plane_edge_def<producers_index_t>;

template <typename Int> struct producers_real_of;
template <> struct producers_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct producers_real_of<tf::exact::int64> {
  using type = double;
};

/// The lattice a mesh's identities name.
template <typename Int, typename Policy>
auto lattice_of(const tf::polygons<Policy> &polygons)
    -> tf::buffer<tf::point<Int, 3>> {
  const auto converter = tf::exact::make_vertex_converter<Int>(polygons);
  tf::buffer<tf::point<Int, 3>> points;
  points.allocate(polygons.points().size());
  for (std::size_t point = 0; point < points.size(); ++point)
    points[point] = converter.convert(polygons.points()[point]);
  return points;
}

/// Every unordered corner pair the mesh's sides name, counted once.
template <typename Policy>
auto mesh_edges(const tf::polygons<Policy> &polygons)
    -> std::set<std::array<producers_index_t, 2>> {
  std::set<std::array<producers_index_t, 2>> edges;
  for (const auto face : polygons.faces()) {
    const auto n = face.size();
    for (std::size_t side = 0; side < n; ++side) {
      auto a = producers_index_t(face[side]);
      auto b = producers_index_t(face[(side + 1) % n]);
      if (b < a)
        std::swap(a, b);
      edges.insert({a, b});
    }
  }
  return edges;
}

struct defs_report_t {
  std::size_t rows = 0;
  std::size_t key_order_defects = 0;
  std::size_t loop_order_defects = 0;
  std::size_t span_defects = 0;
  std::size_t row_id_defects = 0;
  std::size_t face_defects = 0;
  std::size_t whole_side_defects = 0;
};

template <typename Policy>
auto read_defs(const tf::polygons<Policy> &polygons,
               const tf::buffer<producers_def_t> &defs,
               const tf::buffer<producers_index_t> &face_def_offsets)
    -> defs_report_t {
  defs_report_t report;
  report.rows = defs.size();
  for (producers_index_t face = 0;
       face < producers_index_t(polygons.faces().size()); ++face) {
    const auto corners = polygons.faces()[std::size_t(face)];
    const auto begin = std::size_t(face_def_offsets[std::size_t(face)]);
    const auto end = std::size_t(face_def_offsets[std::size_t(face) + 1]);
    if (end - begin != corners.size())
      ++report.span_defects;
    for (auto row = begin; row < end; ++row) {
      const auto &def = defs[row];
      const auto side = std::size_t(def.side);
      if (!(std::array<producers_index_t, 2>{producers_index_t(def.point_tag_0),
                                             def.point_0} <
            std::array<producers_index_t, 2>{producers_index_t(def.point_tag_1),
                                             def.point_1}))
        ++report.key_order_defects;
      if (producers_index_t(def.id) != producers_index_t(row))
        ++report.row_id_defects;
      if (def.face != face || def.object_other != face || def.tag_other != 0)
        ++report.face_defects;
      if (!(def.flags &
            tf::intersect::graph::plane_edge_whole_side_flag) ||
          def.ordinal != def.side)
        ++report.whole_side_defects;
      const auto start = tf::intersect::graph::plane_edge_loop_start(def);
      const auto finish = tf::intersect::graph::plane_edge_loop_end(def);
      if (start[1] != producers_index_t(corners[side]) ||
          finish[1] != producers_index_t(corners[(side + 1) % corners.size()]))
        ++report.loop_order_defects;
    }
  }
  return report;
}

template <typename Policy>
auto build_defs(const tf::polygons<Policy> &polygons,
                tf::buffer<producers_def_t> &defs,
                tf::buffer<producers_index_t> &face_def_offsets) -> void {
  tf::arrangement::build_mesh_plane_defs<producers_index_t>(
      polygons.faces(), defs, face_def_offsets);
}

template <typename Mesh> auto check_defs(Mesh &mesh) -> void {
  tf::buffer<producers_def_t> defs;
  tf::buffer<producers_index_t> face_def_offsets;
  build_defs(mesh.polygons(), defs, face_def_offsets);
  const auto report = read_defs(mesh.polygons(), defs, face_def_offsets);
  CHECK(report.rows == mesh.faces_buffer().data_buffer().size());
  CHECK(report.span_defects == 0);
  CHECK(report.key_order_defects == 0);
  CHECK(report.loop_order_defects == 0);
  CHECK(report.row_id_defects == 0);
  CHECK(report.face_defects == 0);
  CHECK(report.whole_side_defects == 0);
}

template <typename Mesh> auto check_canon(Mesh &mesh) -> void {
  const auto polygons = mesh.polygons();
  tf::buffer<producers_def_t> defs;
  tf::buffer<producers_index_t> face_def_offsets;
  build_defs(polygons, defs, face_def_offsets);

  tf::buffer<producers_index_t> def_offsets;
  tf::buffer<producers_index_t> emission_to_canon;
  producers_index_t n_canon = 0;
  tf::intersect::graph::canonicalize_plane_edge_defs<producers_index_t>(
      defs, def_offsets, n_canon, emission_to_canon);
  const auto edges = mesh_edges(polygons);
  CHECK(std::size_t(n_canon) == edges.size());

  std::size_t key_defects = 0;
  std::size_t membership_defects = 0;
  for (producers_index_t group = 0; group < n_canon; ++group) {
    const auto begin = std::size_t(def_offsets[std::size_t(group)]);
    const auto end = std::size_t(def_offsets[std::size_t(group) + 1]);
    const std::array<producers_index_t, 2> key{defs[begin].point_0,
                                               defs[begin].point_1};
    if (edges.count(key) == 0)
      ++membership_defects;
    for (auto row = begin; row < end; ++row)
      if (defs[row].point_0 != key[0] || defs[row].point_1 != key[1] ||
          defs[row].id != group)
        ++key_defects;
  }
  CHECK(key_defects == 0);
  CHECK(membership_defects == 0);

  const auto n_faces = producers_index_t(polygons.faces().size());
  tf::offset_block_buffer<producers_index_t, producers_index_t> plane_edges;
  tf::intersect::graph::build_plane_edges<producers_index_t>(
      n_faces, tf::make_blocked_range<1>(tf::make_sequence_range(n_faces)),
      face_def_offsets, emission_to_canon, plane_edges);
  std::size_t csr_defects = 0;
  for (producers_index_t face = 0; face < n_faces; ++face) {
    const auto block = tf::make_range(plane_edges)[std::size_t(face)];
    if (block.size() != polygons.faces()[std::size_t(face)].size())
      ++csr_defects;
    for (const auto row : block)
      if (defs[std::size_t(row)].face != face)
        ++csr_defects;
    if (!std::is_sorted(block.begin(), block.end()))
      ++csr_defects;
  }
  CHECK(csr_defects == 0);
}

template <typename Int, typename Mesh> auto check_orientations(Mesh &mesh)
    -> void {
  const auto polygons = mesh.polygons();
  const auto points = lattice_of<Int>(polygons);
  const auto get_point = [&](producers_index_t id) {
    return points[std::size_t(id)];
  };
  const auto fixture = tf::test::make_plane_mesh_fixture<Int>(polygons);
  const auto &world = fixture.input;
  REQUIRE(world.n_faces() == producers_index_t(polygons.faces().size()));

  std::size_t defects = 0;
  for (producers_index_t face = 0;
       face < producers_index_t(polygons.faces().size()); ++face) {
    const auto frame = world.frame(face);
    const auto area =
        tf::exact::signed_area_2x(polygons.faces()[std::size_t(face)],
                                  [&](auto corner) -> tf::exact::pt2<Int> {
                                    const auto q =
                                        get_point(producers_index_t(corner));
                                    return {q[frame.ax0], q[frame.ax1]};
                                  });
    const auto expected = area >= 0 ? std::int8_t(1) : std::int8_t(-1);
    if (world.face_orientation(face) != expected)
      ++defects;
  }
  CHECK(defects == 0);
}

} // namespace

TEMPLATE_TEST_CASE("mesh plane defs: a side states itself in key order",
                   "[arrangement][planes][mesh_producers]", tf::exact::int32,
                   tf::exact::int64) {
  using Real = typename producers_real_of<TestType>::type;
  auto reflex = tf::test::make_mesh_non_convex_l<Real>();
  auto collinear = tf::test::make_mesh_collinear_run<Real>();
  auto shared = tf::test::make_mesh_shared_edge<Real>();
  auto box = tf::test::make_mesh_box<Real>();
  check_defs(reflex);
  check_defs(collinear);
  check_defs(shared);
  check_defs(box);
}

TEMPLATE_TEST_CASE("mesh plane defs: a canonical group is a mesh edge",
                   "[arrangement][planes][mesh_producers]", tf::exact::int32,
                   tf::exact::int64) {
  using Real = typename producers_real_of<TestType>::type;
  auto shared = tf::test::make_mesh_shared_edge<Real>();
  auto box = tf::test::make_mesh_box<Real>();
  check_canon(shared);
  check_canon(box);
}

TEMPLATE_TEST_CASE("mesh plane orientations: the winding is the area's sign",
                   "[arrangement][planes][mesh_producers]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename producers_real_of<Int>::type;
  auto quad = tf::test::make_mesh_convex_quad<Real>();
  auto tilted = tf::test::make_mesh_tilted_quad<Real>();
  auto reflex = tf::test::make_mesh_non_convex_l<Real>();
  auto collinear = tf::test::make_mesh_collinear_run<Real>();
  auto box = tf::test::make_mesh_box<Real>();
  check_orientations<Int>(quad);
  check_orientations<Int>(tilted);
  check_orientations<Int>(reflex);
  check_orientations<Int>(collinear);
  check_orientations<Int>(box);
}

TEMPLATE_TEST_CASE("mesh plane frames: a collinear leading run is scanned past",
                   "[arrangement][planes][mesh_producers]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename producers_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_collinear_run<Real>();
  const auto polygons = mesh.polygons();
  const auto fixture = tf::test::make_plane_mesh_fixture<Int>(polygons);
  using T2 = typename tf::exact::meta<Int>::T2;
  REQUIRE(fixture.input.n_faces() == producers_index_t(1));
  const auto frame = fixture.input.frame(producers_index_t(0));
  CHECK(frame.ax0 != frame.ax1);
  CHECK((frame.plane_n[0] != T2(0) || frame.plane_n[1] != T2(0) ||
         frame.plane_n[2] != T2(0)));
}
