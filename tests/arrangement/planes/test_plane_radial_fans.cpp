/**
 * @file test_plane_radial_fans.cpp
 * @brief Tests for the radial fan tier of the plane arrangement
 *
 * Is it a fan? Collect all the planes around it. The admission is the
 * fence's verdict and nothing else: every fan piece has a fan, size two
 * included, and its pages — one carrier plane on one side, carrying the
 * occurrences that sit on it — come out in radial order around the
 * edge, so forms alternate where geometry says they must. Orientation
 * comes from the resolved piece endpoints.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"
#include "input_lattice_for.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/graph/make_plane_radial_fans.hpp>
#include <trueform/csg/graph/make_plane_triangle_faces.hpp>
#include <trueform/arrangement/planes/make_plane_arrangement_cells.hpp>
#include <trueform/arrangement/planes/make_plane_piece_fences.hpp>
#include <trueform/arrangement/planes/make_plane_piece_incidence.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

using radial_fans_index_t = tf::test::plane_index_t;

template <typename Int> struct radial_fans_real_of;
template <> struct radial_fans_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct radial_fans_real_of<tf::exact::int64> {
  using type = double;
};

template <typename Real>
using radial_fans_mesh_t = tf::polygons_buffer<radial_fans_index_t, Real, 3, 3>;

/// One fan, decoded for assertions: the form tags of its occurrences in
/// page order, with the direction bits in lockstep.
struct fan_t {
  std::vector<radial_fans_index_t> tags;
  std::vector<char> dirs;
  /// Per page, the wedge the tier orders — the emitting face's normal turned
  /// by the page's walk direction — and the piece's key-order direction the
  /// ring turns around. Both are read here from the scene's own lattice
  /// points, so they answer the ring independently of the tier.
  std::vector<std::array<double, 3>> wedges;
  std::array<double, 3> axis{};
};

struct fans_t {
  radial_fans_index_t n_fan_pieces = 0;
  /// Fans whose definitions named no single carrier line.
  radial_fans_index_t n_refused = 0;
  std::vector<fan_t> fans;
  /// Per fan: the pages its ring walks — one carrier plane on one side.
  std::vector<radial_fans_index_t> pages;
};

template <typename Int, typename Real>
auto radial_fans_measure(std::vector<radial_fans_mesh_t<Real>> meshes,
                         bool within) -> fans_t {
  std::vector<
      std::unique_ptr<tf::test::tagged_operand<radial_fans_index_t, Real>>>
      scenes;
  for (auto &mesh : meshes)
    scenes.push_back(
        std::make_unique<tf::test::tagged_operand<radial_fans_index_t, Real>>(
            std::move(mesh)));
  std::vector<decltype(scenes[0]->form())> forms;
  for (auto &scene : scenes)
    forms.push_back(scene->form());
  const auto form_range =
      tf::make_range(forms.data(), forms.data() + forms.size());
  const auto apply_to_face = [&](int tag, radial_fans_index_t object,
                                 const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[object]);
  };
  const auto apply_to_form = [&](radial_fans_index_t tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<radial_fans_index_t> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + radial_fans_index_t(forms[tag].faces().size());

  const auto mode = within ? tf::intersect_mode::primitives |
                                 tf::intersect_mode::resolve_crossing_contours |
                                 tf::intersect_mode::within
                           : tf::intersect_mode::primitives |
                                 tf::intersect_mode::resolve_crossing_contours;
  tf::polygon_intersections<radial_fans_index_t, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(form_range, 0.0);
  intersections.build(form_range, intersections_lattice, tf::intersect_config{mode, 0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_mesh_point = [&](int tag,
                                  radial_fans_index_t id) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[id]);
  };

  tf::arrangement::plane_arrangement<radial_fans_index_t, Int> arrangement;
  arrangement.record_triangle_cells();
  arrangement.record_triangle_arrangement();
  tf::intersect::graph::local_arrangement<radial_fans_index_t, Real, Int> world;
  world.build(std::move(intersections), get_mesh_point, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), false, false);
  arrangement.build(world, get_mesh_point);

  REQUIRE(arrangement.failed().size() == 0);
  const auto cells = tf::arrangement::make_plane_arrangement_cells(arrangement);
  const auto incidence =
      tf::arrangement::make_plane_piece_incidence(arrangement, cells);
  const auto fences = tf::arrangement::make_plane_piece_fences(
      arrangement, world, incidence, cells);
  const auto fans = tf::csg::graph::make_plane_radial_fans(
      arrangement, world, incidence, fences, get_mesh_point, apply_to_face);

  fans_t out;
  for (std::size_t piece = 0; piece < fences.fan.size(); ++piece)
    out.n_fan_pieces += radial_fans_index_t(fences.fan[piece] != char(0));
  REQUIRE(radial_fans_index_t(fans.pieces.size()) == out.n_fan_pieces);
  out.n_refused = fans.n_refused;

  const auto face_of = tf::csg::graph::make_plane_triangle_faces(arrangement);
  // the piece endpoints, as the scene names them: an original vertex or an
  // identity the local arrangement minted
  const auto endpoint = [&](std::int16_t tag, radial_fans_index_t id) {
    REQUIRE((tag >= std::int16_t(0) || id < world.n_created_points()));
    return tag >= std::int16_t(0) ? get_mesh_point(int(tag), id)
                                  : world.point_of(id, get_mesh_point);
  };
  const auto face_normal = [&](radial_fans_index_t face) {
    const auto &descriptor = world.descriptor(face);
    std::array<double, 3> normal{};
    apply_to_face(int(descriptor.tag), descriptor.object,
                  [&](const auto &corners) {
                    const auto p0 = get_mesh_point(int(descriptor.tag),
                                                   corners[0]);
                    const auto p1 = get_mesh_point(int(descriptor.tag),
                                                   corners[1]);
                    const auto p2 = get_mesh_point(int(descriptor.tag),
                                                   corners[2]);
                    const std::array<double, 3> e0{double(p1[0]) - double(p0[0]),
                                                   double(p1[1]) - double(p0[1]),
                                                   double(p1[2]) - double(p0[2])};
                    const std::array<double, 3> e1{double(p2[0]) - double(p0[0]),
                                                   double(p2[1]) - double(p0[1]),
                                                   double(p2[2]) - double(p0[2])};
                    normal = {e0[1] * e1[2] - e0[2] * e1[1],
                              e0[2] * e1[0] - e0[0] * e1[2],
                              e0[0] * e1[1] - e0[1] * e1[0]};
                  });
    return normal;
  };

  const auto &offsets = fans.rows.offsets_buffer();
  for (std::size_t k = 0; k < fans.pieces.size(); ++k) {
    fan_t fan;
    const auto &reference =
        arrangement.piece_definitions(world, fans.pieces[k])[0];
    const auto from = endpoint(reference.point_tag_0, reference.point_0);
    const auto to = endpoint(reference.point_tag_1, reference.point_1);
    fan.axis = {double(to[0]) - double(from[0]), double(to[1]) - double(from[1]),
                double(to[2]) - double(from[2])};
    for (auto page = std::size_t(fans.page_offsets[k]);
         page < std::size_t(fans.page_offsets[k + 1]); ++page) {
      // one page is one carrier plane on one side: its occurrences all
      // sit on it, so the ring reads the page and the tags read through
      REQUIRE(offsets[page + 1] > offsets[page]);
      auto wedge = face_normal(face_of[std::size_t(
          fans.rows.data_buffer()[std::size_t(offsets[page])] /
          radial_fans_index_t(3))]);
      if (!fans.dirs[std::size_t(offsets[page])])
        for (auto &component : wedge)
          component = -component;
      fan.wedges.push_back(wedge);
      for (auto at = std::size_t(offsets[page]);
           at < std::size_t(offsets[page + 1]); ++at) {
        const auto face = face_of[std::size_t(fans.rows.data_buffer()[at] /
                                              radial_fans_index_t(3))];
        fan.tags.push_back(radial_fans_index_t(world.descriptor(face).tag));
        fan.dirs.push_back(fans.dirs[at]);
      }
    }
    out.pages.push_back(
        radial_fans_index_t(fans.page_offsets[k + 1] - fans.page_offsets[k]));
    out.fans.push_back(std::move(fan));
  }
  return out;
}

/// A ring is a SWEEP: read as angles about the piece's key-order direction and
/// measured from its first page, the pages come out non-decreasing. An axis on
/// the wrong component scrambles them; an axis of the wrong sign reverses them.
auto sweeps(const fan_t &fan) -> bool {
  const auto dot = [](const std::array<double, 3> &a,
                      const std::array<double, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const auto scaled = [](const std::array<double, 3> &a, double by) {
    return std::array<double, 3>{a[0] * by, a[1] * by, a[2] * by};
  };
  const auto perpendicular = [&](const std::array<double, 3> &a,
                                 const std::array<double, 3> &d) {
    const auto along = dot(a, d);
    return std::array<double, 3>{a[0] - along * d[0], a[1] - along * d[1],
                                 a[2] - along * d[2]};
  };
  if (fan.wedges.size() < 3)
    return true;
  const auto axis_length = std::sqrt(dot(fan.axis, fan.axis));
  if (axis_length == 0.0)
    return false;
  const auto d = scaled(fan.axis, 1.0 / axis_length);
  auto u = perpendicular(fan.wedges[0], d);
  const auto u_length = std::sqrt(dot(u, u));
  if (u_length == 0.0)
    return false;
  u = scaled(u, 1.0 / u_length);
  const std::array<double, 3> v{d[1] * u[2] - d[2] * u[1],
                                d[2] * u[0] - d[0] * u[2],
                                d[0] * u[1] - d[1] * u[0]};
  double previous = 0.0;
  for (std::size_t page = 1; page < fan.wedges.size(); ++page) {
    const auto in_plane = perpendicular(fan.wedges[page], d);
    auto angle = std::atan2(dot(in_plane, v), dot(in_plane, u));
    if (angle < 0.0)
      angle += 2.0 * 3.14159265358979323846;
    if (angle + 1e-9 < previous)
      return false;
    previous = angle;
  }
  return true;
}

/// Around a full radial ring, neighbouring occurrences must alternate
/// between the two classes; an open pair trivially does.
auto alternates(const std::vector<radial_fans_index_t> &tags) -> bool {
  const auto n = tags.size();
  for (std::size_t i = 0; i < n; ++i)
    if (tags[i] == tags[(i + 1) % n])
      return false;
  return true;
}

template <typename Real>
auto radial_fans_bare_touch() -> std::vector<radial_fans_mesh_t<Real>> {
  radial_fans_mesh_t<Real> a;
  a.points_buffer().push_back({Real(0), Real(0), Real(0)});
  a.points_buffer().push_back({Real(8), Real(0), Real(0)});
  a.points_buffer().push_back({Real(0), Real(8), Real(0)});
  a.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  radial_fans_mesh_t<Real> b;
  b.points_buffer().push_back({Real(0), Real(0), Real(0)});
  b.points_buffer().push_back({Real(8), Real(0), Real(0)});
  b.points_buffer().push_back({Real(0), Real(0), Real(8)});
  b.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(2), radial_fans_index_t(1)});
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(std::move(a));
  meshes.push_back(std::move(b));
  return meshes;
}

/// Form 0 a wing pair in y = 0, form 1 a wing pair in z = 0, all four on
/// the shared original edge (0,0,0)-(8,0,0).
template <typename Real>
auto radial_fans_exact_edge_merge() -> std::vector<radial_fans_mesh_t<Real>> {
  radial_fans_mesh_t<Real> a;
  a.points_buffer().push_back({Real(0), Real(0), Real(0)});
  a.points_buffer().push_back({Real(8), Real(0), Real(0)});
  a.points_buffer().push_back({Real(4), Real(0), Real(5)});
  a.points_buffer().push_back({Real(4), Real(0), Real(-5)});
  a.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  a.faces_buffer().push_back(
      {radial_fans_index_t(1), radial_fans_index_t(0), radial_fans_index_t(3)});
  radial_fans_mesh_t<Real> b;
  b.points_buffer().push_back({Real(0), Real(0), Real(0)});
  b.points_buffer().push_back({Real(8), Real(0), Real(0)});
  b.points_buffer().push_back({Real(4), Real(5), Real(0)});
  b.points_buffer().push_back({Real(4), Real(-5), Real(0)});
  b.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  b.faces_buffer().push_back(
      {radial_fans_index_t(1), radial_fans_index_t(0), radial_fans_index_t(3)});
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(std::move(a));
  meshes.push_back(std::move(b));
  return meshes;
}

template <typename Real>
auto triangle(std::array<std::array<Real, 3>, 3> corners)
    -> radial_fans_mesh_t<Real> {
  radial_fans_mesh_t<Real> mesh;
  for (const auto &corner : corners)
    mesh.points_buffer().push_back({corner[0], corner[1], corner[2]});
  mesh.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  return mesh;
}

/// Three forms standing on ONE shared original edge (0,0,0)-(8,0,0), each in
/// its own plane: three statements of one line, and no cut anywhere.
template <typename Real>
auto shared_edge_three_tags() -> std::vector<radial_fans_mesh_t<Real>> {
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(triangle<Real>(
      {{{Real(0), Real(0), Real(0)}, {Real(8), Real(0), Real(0)},
        {Real(4), Real(6), Real(0)}}}));
  meshes.push_back(triangle<Real>(
      {{{Real(0), Real(0), Real(0)}, {Real(8), Real(0), Real(0)},
        {Real(4), Real(0), Real(6)}}}));
  meshes.push_back(triangle<Real>(
      {{{Real(0), Real(0), Real(0)}, {Real(8), Real(0), Real(0)},
        {Real(4), Real(-3), Real(-3)}}}));
  return meshes;
}

/// An intersection edge that is ALSO an original edge: form 0's wing pair
/// shares the segment (-4,0,0)-(4,0,0), and form 1's plate is cut by exactly
/// that segment in its interior.
template <typename Real>
auto shared_edge_and_cut() -> std::vector<radial_fans_mesh_t<Real>> {
  radial_fans_mesh_t<Real> wings;
  wings.points_buffer().push_back({Real(-4), Real(0), Real(0)});
  wings.points_buffer().push_back({Real(4), Real(0), Real(0)});
  wings.points_buffer().push_back({Real(0), Real(0), Real(6)});
  wings.points_buffer().push_back({Real(0), Real(0), Real(-6)});
  wings.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  wings.faces_buffer().push_back(
      {radial_fans_index_t(1), radial_fans_index_t(0), radial_fans_index_t(3)});
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(std::move(wings));
  meshes.push_back(triangle<Real>(
      {{{Real(-9), Real(-3), Real(0)}, {Real(9), Real(-3), Real(0)},
        {Real(0), Real(6), Real(0)}}}));
  return meshes;
}

/// Three planes meeting on ONE line, each cut in its interior: every pair
/// states the same intersection edge, so one piece carries six statements
/// from three different pairs and none of them is an original edge.
template <typename Real>
auto three_planes_on_one_line() -> std::vector<radial_fans_mesh_t<Real>> {
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(triangle<Real>(
      {{{Real(-6), Real(-3), Real(0)}, {Real(6), Real(-3), Real(0)},
        {Real(0), Real(6), Real(0)}}}));
  meshes.push_back(triangle<Real>(
      {{{Real(-6), Real(0), Real(-3)}, {Real(6), Real(0), Real(-3)},
        {Real(0), Real(0), Real(6)}}}));
  meshes.push_back(triangle<Real>(
      {{{Real(-6), Real(-3), Real(-3)}, {Real(6), Real(-3), Real(-3)},
        {Real(0), Real(6), Real(6)}}}));
  return meshes;
}

/// One form whose three wings meet on one NON-MANIFOLD input edge.
template <typename Real> auto non_manifold_wings() -> radial_fans_mesh_t<Real> {
  radial_fans_mesh_t<Real> mesh;
  mesh.points_buffer().push_back({Real(0), Real(0), Real(0)});
  mesh.points_buffer().push_back({Real(8), Real(0), Real(0)});
  mesh.points_buffer().push_back({Real(4), Real(6), Real(0)});
  mesh.points_buffer().push_back({Real(4), Real(0), Real(6)});
  mesh.points_buffer().push_back({Real(4), Real(-3), Real(-3)});
  mesh.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(2)});
  mesh.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(3)});
  mesh.faces_buffer().push_back(
      {radial_fans_index_t(0), radial_fans_index_t(1), radial_fans_index_t(4)});
  return mesh;
}

/// That non-manifold edge reached by a second form standing on it: three of
/// the four statements come from ONE form's own edge, and the piece is never
/// split.
template <typename Real>
auto non_manifold_shared_edge() -> std::vector<radial_fans_mesh_t<Real>> {
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(non_manifold_wings<Real>());
  meshes.push_back(triangle<Real>(
      {{{Real(0), Real(0), Real(0)}, {Real(8), Real(0), Real(0)},
        {Real(4), Real(3), Real(-3)}}}));
  return meshes;
}

/// The same non-manifold edge, split in its interior by a second form: every
/// piece of it inherits the original edge's line with its endpoints moved.
template <typename Real>
auto non_manifold_wings_split() -> std::vector<radial_fans_mesh_t<Real>> {
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(non_manifold_wings<Real>());
  meshes.push_back(triangle<Real>(
      {{{Real(4), Real(-8), Real(-8)}, {Real(4), Real(8), Real(-8)},
        {Real(4), Real(0), Real(8)}}}));
  return meshes;
}

/// THE SAME CROSSING STATED ON THE LATTICE ITSELF, at its extremes: the
/// coordinates are already the integers the arrangement runs on, so no
/// converter rescales them and every corner stands at the span's edge. A
/// carrier's exact normal is then a degree-two value past 2^63, which is the
/// width where a DOT of two of them leaves the rung its ladder covers — the
/// ring's classes are read off determinants and component senses, so the
/// pages must come out in the same radial order they do in the small.
auto full_span_crossing() -> std::vector<radial_fans_mesh_t<std::int32_t>> {
  const std::int32_t s = 2147483646;
  std::vector<radial_fans_mesh_t<std::int32_t>> meshes;
  // wound so its carrier's normal points DOWN: the ring's first page is then
  // the positive wedge, and it is the positive operand of that width a
  // formed dot cannot carry
  meshes.push_back(triangle<std::int32_t>(
      {{{std::int32_t(-s), std::int32_t(-s), std::int32_t(0)},
        {std::int32_t(0), std::int32_t(s), std::int32_t(0)},
        {std::int32_t(s), std::int32_t(-s), std::int32_t(0)}}}));
  meshes.push_back(triangle<std::int32_t>(
      {{{std::int32_t(-s), std::int32_t(0), std::int32_t(-s)},
        {std::int32_t(s), std::int32_t(0), std::int32_t(-s)},
        {std::int32_t(0), std::int32_t(0), std::int32_t(s)}}}));
  return meshes;
}

/// A plate of form 0 in z = 0 crossed by a standing plate of form 1.
template <typename Real>
auto crossing() -> std::vector<radial_fans_mesh_t<Real>> {
  std::vector<radial_fans_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<radial_fans_index_t, Real>(6, 6, 1, 1));
  auto standing = tf::make_plane_mesh<radial_fans_index_t, Real>(6, 6, 1, 1);
  for (std::size_t point = 0; point < standing.points().size(); ++point)
    std::swap(standing.points()[point][1], standing.points()[point][2]);
  meshes.push_back(std::move(standing));
  return meshes;
}

} // namespace

TEMPLATE_TEST_CASE("plane radial fans: a bare touch keeps its size-two fan",
                   "[cut][planes][radial-fans]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // Two forms on one shared edge: one fan piece, one fan of two, kept —
  // its curve closes and its division holds like any other. The two
  // occurrences walk the edge in opposite directions.
  const auto touch =
      radial_fans_measure<Int, Real>(radial_fans_bare_touch<Real>(), false);
  CHECK(touch.n_fan_pieces == 1);
  REQUIRE(touch.fans.size() == 1);
  CHECK(touch.pages[0] == 2);
  CHECK(touch.fans[0].tags.size() == 2);
  CHECK(alternates(touch.fans[0].tags));
  CHECK(touch.fans[0].dirs[0] != touch.fans[0].dirs[1]);
}

TEMPLATE_TEST_CASE("plane radial fans: four wings on one line alternate "
                   "forms radially",
                   "[cut][planes][radial-fans]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // Two wing pairs of two forms merged on one exact edge: one fan of
  // four, and around the ring the forms must alternate — the wings of
  // one form face each other across the other's plane.
  const auto merge = radial_fans_measure<Int, Real>(
      radial_fans_exact_edge_merge<Real>(), false);
  REQUIRE(merge.fans.size() == 1);
  CHECK(merge.pages[0] == 4);
  CHECK(merge.fans[0].tags.size() == 4);
  CHECK(alternates(merge.fans[0].tags));
}

TEMPLATE_TEST_CASE("plane radial fans: a crossing fans four around the cut, "
                   "alternating",
                   "[cut][planes][radial-fans]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // Two plates crossing: every piece of the cut line carries a fan of
  // four — each plate's two halves — and the plates alternate radially.
  const auto crossed = radial_fans_measure<Int, Real>(crossing<Real>(), false);
  CHECK(crossed.fans.size() > 0);
  for (std::size_t k = 0; k < crossed.fans.size(); ++k) {
    // two plates, four half-planes: the pages are the sides, never the
    // triangles the plates were cut into
    CHECK(crossed.pages[k] == 4);
    CHECK(crossed.fans[k].tags.size() == 4);
    CHECK(alternates(crossed.fans[k].tags));
  }
}

TEST_CASE("plane radial fans: a crossing at the lattice's full span rings",
          "[cut][planes][radial-fans][radial-authority]") {
  // the wedges of the two plates are exact normals of full-span faces, so a
  // page pair on one plane is antipodal at a magnitude no dot may be formed
  // at: the classes must still be 0 and 2, and the ring must still sweep
  const auto crossed = radial_fans_measure<tf::exact::int32, std::int32_t>(
      full_span_crossing(), false);
  REQUIRE(crossed.fans.size() > 0);
  CHECK(crossed.n_refused == 0);
  for (std::size_t k = 0; k < crossed.fans.size(); ++k) {
    CHECK(crossed.pages[k] == 4);
    CHECK(crossed.fans[k].tags.size() == 4);
    CHECK(alternates(crossed.fans[k].tags));
    CHECK(sweeps(crossed.fans[k]));
  }
}

TEMPLATE_TEST_CASE("plane radial fans: three forms on one shared original "
                   "edge state one line",
                   "[cut][planes][radial-fans][radial-authority]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // three original edges welded into one canonical piece, all naming the same
  // lattice segment: the authority stands and the ring sweeps
  const auto shared =
      radial_fans_measure<Int, Real>(shared_edge_three_tags<Real>(), false);
  REQUIRE(shared.fans.size() == 1);
  CHECK(shared.n_refused == 0);
  CHECK(shared.pages[0] == 3);
  CHECK(shared.fans[0].tags.size() == 3);
  CHECK(sweeps(shared.fans[0]));
}

TEMPLATE_TEST_CASE("plane radial fans: an original edge that is also an "
                   "intersection edge",
                   "[cut][planes][radial-fans][radial-authority]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // form 0's own edge cuts form 1's plate: the piece carries both an original
  // edge and a producing pair, and they must agree
  const auto mixed =
      radial_fans_measure<Int, Real>(shared_edge_and_cut<Real>(), false);
  REQUIRE(mixed.fans.size() > 0);
  CHECK(mixed.n_refused == 0);
  for (std::size_t k = 0; k < mixed.fans.size(); ++k) {
    CHECK(mixed.pages[k] == 4);
    CHECK(alternates(mixed.fans[k].tags));
    CHECK(sweeps(mixed.fans[k]));
  }
}

TEMPLATE_TEST_CASE("plane radial fans: three pairs stating one intersection "
                   "edge reconcile",
                   "[cut][planes][radial-fans][radial-authority]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // no original edge anywhere: three planes meet on one line, so a piece of it
  // carries a statement from every pair and every one of them is checked
  const auto pencil =
      radial_fans_measure<Int, Real>(three_planes_on_one_line<Real>(), false);
  REQUIRE(pencil.fans.size() > 0);
  CHECK(pencil.n_refused == 0);
  for (std::size_t k = 0; k < pencil.fans.size(); ++k) {
    CHECK(pencil.pages[k] == 6);
    CHECK(sweeps(pencil.fans[k]));
  }
}

TEMPLATE_TEST_CASE("plane radial fans: a non-manifold input edge fans on its "
                   "own line",
                   "[cut][planes][radial-fans][radial-authority]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // three faces of form 0 on one input edge, reached by form 1 standing on the
  // same edge: four statements, three of them the same form's own edge
  const auto wings =
      radial_fans_measure<Int, Real>(non_manifold_shared_edge<Real>(), false);
  REQUIRE(wings.fans.size() == 1);
  CHECK(wings.n_refused == 0);
  CHECK(wings.pages[0] == 4);
  CHECK(sweeps(wings.fans[0]));
}

TEMPLATE_TEST_CASE("plane radial fans: a split piece keeps its original "
                   "edge's line",
                   "[cut][planes][radial-fans][radial-authority]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename radial_fans_real_of<Int>::type;

  // the non-manifold edge cut in its interior: both halves carry the same
  // three original-edge statements with their endpoints moved
  const auto split =
      radial_fans_measure<Int, Real>(non_manifold_wings_split<Real>(), false);
  REQUIRE(split.fans.size() >= 2);
  CHECK(split.n_refused == 0);
  radial_fans_index_t three_page = 0;
  for (std::size_t k = 0; k < split.fans.size(); ++k) {
    three_page += radial_fans_index_t(split.pages[k] == 3);
    CHECK(sweeps(split.fans[k]));
  }
  CHECK(three_page >= radial_fans_index_t(2));
}
