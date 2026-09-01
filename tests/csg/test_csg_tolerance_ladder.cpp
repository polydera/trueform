/**
 * @file test_csg_tolerance_ladder.cpp
 * @brief What a band may and may not change, read through the full entries.
 *
 * A band is a statement about the INPUT: every original vertex moves at most
 * the band onto a lattice point of the planes its own faces state, and the
 * arrangement of that moved mesh is then exact. Two consequences are the
 * whole contract, and this suite walks a ladder of bands over two operands
 * that genuinely cut each other to pin both.
 *
 * FIRST, structure is invariant. A band that moves nothing across a feature
 * cannot change what the operands enclose: the same domains, all closed, no
 * carrier refusing, and every boolean expression closed. A band that changed
 * the domain count would be attracting one form's surface onto another's,
 * which the placement never does.
 *
 * SECOND, geometry moves by less than the band and by no more. Each domain's
 * volume is compared against the exact answer with a margin the band's own
 * displacement of the surface allows, so a placement that dragged a wall
 * further than it promised fails here rather than downstream.
 *
 * THIRD, one lattice point is one identity. A created point whose
 * position rounds onto an original vertex, or a recovery crossing that
 * lands a lattice unit from an endpoint of the constraint it splits, is
 * that vertex: the cut faces and the uncut faces of one fan then name one
 * output point, and every surface stays closed with its domains intact.
 * The deep pair below is the scene that states both. Two originals a band
 * places on one point are that point on the same terms, whether a cut face
 * names them or nothing does.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/clean/polygons.hpp>
#include <trueform/core/cross.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg/expression.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/topology/is_closed.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "input_lattice_for.hpp"
#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace {

using index_t = int;
using real_t = float;
using ladder_mesh_t = tf::polygons_buffer<index_t, real_t, 3, 3>;

auto shifted(ladder_mesh_t mesh, real_t dx) -> ladder_mesh_t {
  for (std::size_t i = 0; i < mesh.points_buffer().size(); ++i)
    mesh.points_buffer()[i][0] += dx;
  return mesh;
}

auto joined(ladder_mesh_t mesh, const ladder_mesh_t &other) -> ladder_mesh_t {
  const auto base = index_t(mesh.points_buffer().size());
  for (std::size_t i = 0; i < other.points_buffer().size(); ++i) {
    const auto point = other.points_buffer()[i];
    mesh.points_buffer().emplace_back(point[0], point[1], point[2]);
  }
  for (index_t i = 0; i < index_t(other.faces_buffer().size()); ++i) {
    const auto face = other.faces_buffer()[std::size_t(i)];
    mesh.faces_buffer().emplace_back(base + face[0], base + face[1],
                                     base + face[2]);
  }
  return mesh;
}

/// Two spheres of radius 1 whose centres are `shift` apart — one radius
/// by default: every boolean is non-trivial and the seam runs through the
/// middle of both.
struct pair_t {
  tf::test::tagged_operand<index_t, real_t> left, right;
  explicit pair_t(int stacks, int segments, real_t shift = real_t(1))
      : left(tf::make_sphere_mesh<index_t>(real_t(1), stacks, segments)),
        right(shifted(tf::make_sphere_mesh<index_t>(real_t(1), stacks,
                                                    segments),
                      shift)) {}
  auto views() {
    return std::array<decltype(left.mesh.polygons()), 2>{left.mesh.polygons(),
                                                         right.mesh.polygons()};
  }
  auto forms() {
    return std::vector<tf::test::form_t<index_t, real_t, 3>>{left.form(),
                                                             right.form()};
  }
};

struct reading_t {
  std::size_t n_domains = 0;
  std::size_t n_failed = 0;
  std::size_t n_arrangement_faces = 0;
  std::size_t n_arrangement_points = 0;
  bool all_closed = false;
  bool arrangement_closed = false;
  bool arrangement_degenerate = false;
  std::vector<double> volumes;
};

template <typename Forms>
auto read_at(Forms &forms, double tolerance) -> reading_t {
  auto graph = tf::test::build_range_csg_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()),
      tf::test::no_sheets(),
      tf::arrangement_config{tf::intersect_config{
          tf::intersect_mode::primitives |
              tf::intersect_mode::resolve_crossing_contours,
          tolerance}});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  static_cast<void>(ids);

  reading_t reading;
  reading.n_domains = cells.size();
  reading.n_failed = graph.arrangement().failed().size();
  for (auto &cell : cells)
    reading.volumes.push_back(double(tf::signed_volume(cell.polygons())));
  std::sort(reading.volumes.begin(), reading.volumes.end());

  const auto merged =
      tf::test::csg_mesh_of(graph, tf::csg::op(0) | tf::csg::op(1));
  const auto shared =
      tf::test::csg_mesh_of(graph, tf::csg::op(0) & tf::csg::op(1));
  const auto cut =
      tf::test::csg_mesh_of(graph, tf::csg::op(0) - tf::csg::op(1));
  reading.all_closed = tf::is_closed(merged.polygons()) &&
                       tf::is_closed(shared.polygons()) &&
                       tf::is_closed(cut.polygons());
  const auto arrangement = tf::test::csg_mesh_of(graph);
  reading.n_arrangement_faces = arrangement.polygons().size();
  reading.n_arrangement_points = arrangement.points_buffer().size();
  reading.arrangement_closed = tf::is_closed(arrangement.polygons());
  for (const auto face : arrangement.polygons())
    if (tf::cross(face[1] - face[0], face[2] - face[0]).length2() == real_t(0))
      reading.arrangement_degenerate = true;
  return reading;
}

/// The operands AS THE BAND PLACED THEM: every original vertex read off the
/// lattice's own table and cleaned to shared-vertex identity. The band's
/// contract is that the arrangement it answers is the exact arrangement of
/// this mesh, so this is the oracle, not a second implementation of one.
template <typename Views>
auto moved_operands(Views &views, double tolerance)
    -> std::vector<ladder_mesh_t> {
  const auto lattice = tf::test::input_lattice_for(
      tf::make_range(views.data(), views.data() + views.size()), tolerance);
  const auto placed = lattice.placed_points();
  std::vector<ladder_mesh_t> moved;
  for (index_t tag = 0; tag < index_t(views.size()); ++tag) {
    ladder_mesh_t mesh;
    const auto points = views[std::size_t(tag)].points();
    for (index_t id = 0; id < index_t(points.size()); ++id) {
      const auto at = lattice.converter().deconvert(
          placed[std::size_t(lattice.flat_vertex(int(tag), id))]);
      mesh.points_buffer().emplace_back(at[0], at[1], at[2]);
    }
    for (const auto face : views[std::size_t(tag)].faces())
      mesh.faces_buffer().emplace_back(face[0], face[1], face[2]);
    moved.push_back(tf::cleaned<index_t>(mesh.polygons()));
  }
  return moved;
}

/// The bands the ladder walks, from far below one lattice unit of the
/// operands' own converter to well above it.
constexpr double ladder[] = {1e-8, 3e-8, 5e-8, 8e-8, 9e-8,
                             1e-7, 2e-7, 3e-7, 5e-7, 1e-6};

} // namespace

TEST_CASE("csg tolerance ladder: a band changes no structure",
          "[csg][tolerance]") {
  for (int segments : {6, 12}) {
    pair_t pair(2, segments);
    auto forms = pair.forms();
    const auto exact = read_at(forms, 0.0);
    CAPTURE(segments);
    REQUIRE(exact.n_domains == 3u);
    REQUIRE(exact.n_failed == 0u);
    REQUIRE(exact.all_closed);

    for (double tolerance : ladder) {
      auto forms = pair.forms();
      const auto banded = read_at(forms, tolerance);
      CAPTURE(tolerance);
      REQUIRE(banded.n_failed == 0u);
      REQUIRE(banded.n_domains == exact.n_domains);
      REQUIRE(banded.all_closed);
      // A surface displaced by at most the band changes a volume it
      // bounds by at most the band times that surface's area; a unit
      // sphere pair has area well under 30, and the ladder's widest band
      // is 1e-6.
      for (std::size_t d = 0; d < banded.volumes.size(); ++d)
        REQUIRE(banded.volumes[d] ==
                Catch::Approx(exact.volumes[d]).margin(30.0 * tolerance));
    }
  }
}

TEST_CASE("csg tolerance ladder: a band of zero is the identity",
          "[csg][tolerance]") {
  pair_t pair(2, 12);
  auto forms = pair.forms();
  const auto exact = read_at(forms, 0.0);
  const auto tiny = read_at(forms, 1e-30);
  REQUIRE(tiny.n_domains == exact.n_domains);
  REQUIRE(tiny.volumes == exact.volumes);
}

TEST_CASE("csg tolerance ladder: a point the rounding puts on a vertex is "
          "that vertex",
          "[csg][tolerance]") {
  pair_t pair(40, 60, real_t(0.5));
  auto forms = pair.forms();
  const auto exact = read_at(forms, 0.0);
  REQUIRE(exact.n_domains == 3u);
  REQUIRE(exact.n_failed == 0u);
  REQUIRE(exact.all_closed);
  REQUIRE(exact.arrangement_closed);

  for (double tolerance : ladder) {
    auto forms = pair.forms();
    const auto banded = read_at(forms, tolerance);
    CAPTURE(tolerance);
    REQUIRE(banded.n_failed == 0u);
    REQUIRE(banded.n_domains == exact.n_domains);
    REQUIRE(banded.all_closed);
    REQUIRE(banded.arrangement_closed);
    for (std::size_t d = 0; d < banded.volumes.size(); ++d)
      REQUIRE(banded.volumes[d] ==
              Catch::Approx(exact.volumes[d]).margin(30.0 * tolerance));
  }
}

TEST_CASE("csg tolerance ladder: a band that makes two originals one point "
          "closes the faces they leave",
          "[csg][tolerance]") {
  const real_t thickness = real_t(2e-5);
  tf::test::tagged_operand<index_t, real_t> cube(
      tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
  tf::test::tagged_operand<index_t, real_t> plate(
      tf::make_box_mesh<index_t>(real_t(2), real_t(2), thickness));
  std::array<decltype(cube.form()), 2> views{cube.form(), plate.form()};

  // The plate pokes through the cube on every side and is thinner than the
  // upper bands, so each of its four vertical edges collapses to one lattice
  // point: two originals become one identity, and the eight side triangles
  // that carried them bound nothing. The surface they were part of must
  // still close.
  for (double tolerance : {2e-5, 3e-5, 5e-5, 1e-4}) {
    const auto banded = read_at(views, tolerance);
    CAPTURE(tolerance);
    REQUIRE(banded.n_failed == 0u);
    REQUIRE(banded.arrangement_closed);
    if (tolerance < 3e-5)
      continue;
    // At and above this band the collapse has happened, and the answer is
    // the exact arrangement of the mesh the placement made.
    auto moved = moved_operands(views, tolerance);
    std::vector<tf::test::tagged_operand<index_t, real_t>> moved_operands_;
    moved_operands_.reserve(moved.size());
    for (auto &mesh : moved)
      moved_operands_.push_back(tf::test::make_tagged_operand(mesh));
    auto moved_forms = tf::test::tagged_forms(moved_operands_);
    const auto placed = read_at(moved_forms, 0.0);
    REQUIRE(banded.n_arrangement_faces == placed.n_arrangement_faces);
    REQUIRE(banded.n_arrangement_points == placed.n_arrangement_points);
    REQUIRE(banded.n_domains == placed.n_domains);
  }
}

TEST_CASE("csg tolerance ladder: a collapse no cut face names is still one "
          "point",
          "[csg][tolerance]") {
  const real_t thickness = real_t(2e-5);
  // One operand carries the cut and a thin slab that stands well clear of
  // it, so the slab's four vertical edges collapse in the placement while
  // no definition, no split and no promoted ring ever names one of their
  // ends. Only the identity gate can answer them.
  tf::test::tagged_operand<index_t, real_t> cut_and_slab(joined(
      tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)),
      shifted(tf::make_box_mesh<index_t>(real_t(2), real_t(2), thickness),
              real_t(2.5))));
  tf::test::tagged_operand<index_t, real_t> cutter(shifted(
      tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)),
      real_t(0.5)));
  std::array<decltype(cut_and_slab.form()), 2> views{cut_and_slab.form(),
                                                     cutter.form()};

  for (double tolerance : {2e-5, 3e-5, 5e-5, 1e-4}) {
    const auto banded = read_at(views, tolerance);
    CAPTURE(tolerance);
    REQUIRE(banded.n_failed == 0u);
    REQUIRE(banded.arrangement_closed);
    if (tolerance < 3e-5)
      continue;
    auto moved = moved_operands(views, tolerance);
    // the oracle's own premise: the placement flattened the slab, so the
    // operand's points are the cube's eight and the slab's four
    REQUIRE(moved[0].points_buffer().size() == 12u);
    std::vector<tf::test::tagged_operand<index_t, real_t>> moved_ops;
    moved_ops.reserve(moved.size());
    for (auto &mesh : moved)
      moved_ops.push_back(tf::test::make_tagged_operand(mesh));
    auto moved_forms = tf::test::tagged_forms(moved_ops);
    const auto placed = read_at(moved_forms, 0.0);
    REQUIRE_FALSE(banded.arrangement_degenerate);
    REQUIRE(banded.n_arrangement_points == placed.n_arrangement_points);
    REQUIRE(banded.n_arrangement_faces == placed.n_arrangement_faces);
    REQUIRE(banded.n_domains == placed.n_domains);
  }
}
