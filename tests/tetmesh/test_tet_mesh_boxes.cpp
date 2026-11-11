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
#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"
#include "uniform_draw.hpp"
#include "worker_ladder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <tbb/task_arena.h>
#include <tetmesh_builders.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/expression/expr.hpp>
#include <trueform/csg/expression/operators.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/reindex/concatenated.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using tb_mesh = tf::polygons_buffer<int, float, 3, 3>;
using tb_rotation = std::array<double, 9>;

constexpr tb_rotation tb_identity{1, 0, 0, 0, 1, 0, 0, 0, 1};

/// A box of `size` rotated by the row-major `rotation` about its centre and
/// moved to `center`, wound outward, each face a grid of `ticks` by `ticks`
/// squares.
struct tb_box {
  std::array<double, 3> size;
  std::array<double, 3> center;
  tb_rotation rotation = tb_identity;
  int ticks = 1;
};

auto tb_mesh_of(const tb_box &box) -> tb_mesh {
  auto mesh = box.ticks > 1 ? tf::make_box_mesh<int>(1.f, 1.f, 1.f, box.ticks,
                                                     box.ticks, box.ticks)
                            : tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  for (auto point : mesh.points()) {
    const double local[3] = {double(point[0]) * box.size[0],
                             double(point[1]) * box.size[1],
                             double(point[2]) * box.size[2]};
    for (std::size_t k = 0; k < 3; ++k)
      point[k] = float(box.rotation[3 * k] * local[0] +
                       box.rotation[3 * k + 1] * local[1] +
                       box.rotation[3 * k + 2] * local[2] + box.center[k]);
  }
  return mesh;
}

/// `mesh` with every face's winding reversed.
auto tb_reversed(tb_mesh mesh) -> tb_mesh {
  for (auto face : mesh.faces())
    std::swap(face[1], face[2]);
  return mesh;
}

/// The boxes as one surface, those `inward` names wound inward.
auto tb_surface_of(const std::vector<tb_box> &boxes,
                   const std::vector<int> &inward) -> tb_mesh {
  std::vector<tb_mesh> meshes;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    auto mesh = tb_mesh_of(boxes[i]);
    meshes.push_back(std::find(inward.begin(), inward.end(), int(i)) !=
                             inward.end()
                         ? tb_reversed(std::move(mesh))
                         : std::move(mesh));
  }
  using view_t = decltype(meshes[0].polygons());
  std::vector<view_t> views;
  for (auto &mesh : meshes)
    views.push_back(mesh.polygons());
  return tf::concatenated(
      tf::make_range(views.data(), views.data() + views.size()));
}

/// The volume a closed triangle surface winds, by the divergence theorem.
template <typename Mesh> auto tb_enclosed(const Mesh &surface) -> double {
  const auto points = surface.points();
  double sum = 0;
  for (const auto &face : surface.faces()) {
    const auto a = points[std::size_t(face[0])],
               b = points[std::size_t(face[1])],
               c = points[std::size_t(face[2])];
    sum += double(a[0]) * (double(b[1]) * c[2] - double(b[2]) * c[1]) -
           double(a[1]) * (double(b[0]) * c[2] - double(b[2]) * c[0]) +
           double(a[2]) * (double(b[0]) * c[1] - double(b[1]) * c[0]);
  }
  return sum / 6;
}

/// `mesh` in double, so the csg tier's volume of its boxes is read exactly.
auto tb_double(const tb_mesh &mesh) -> tf::polygons_buffer<int, double, 3, 3> {
  tf::polygons_buffer<int, double, 3, 3> out;
  out.faces_buffer().allocate(mesh.faces().size());
  for (std::size_t f = 0; f < mesh.faces().size(); ++f)
    for (std::size_t k = 0; k < 3; ++k)
      out.faces_buffer()[f][k] = mesh.faces()[f][k];
  out.points_buffer().allocate(mesh.points().size());
  for (std::size_t i = 0; i < mesh.points().size(); ++i)
    for (std::size_t k = 0; k < 3; ++k)
      out.points_buffer()[i][k] = double(mesh.points()[i][k]);
  return out;
}

/// The volume of the solid the boxes bound as separate operands of the
/// csg tier, read in double: the union of those `inward` does not name,
/// less the union of those it does.
auto tb_csg_volume(const std::vector<tb_box> &boxes,
                   const std::vector<int> &inward) -> double {
  std::vector<tf::test::tagged_operand<int, double, 3>> operands;
  operands.reserve(boxes.size());
  for (const auto &box : boxes)
    operands.emplace_back(tb_double(tb_mesh_of(box)));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::sheets_of({}), {});
  std::vector<int> kept, cut;
  for (int i = 0; i < int(boxes.size()); ++i)
    (std::find(inward.begin(), inward.end(), i) != inward.end() ? cut : kept)
        .push_back(i);
  const auto union_of = [](const std::vector<int> &ids) {
    auto e = tf::csg::op(ids[0]);
    for (std::size_t k = 1; k < ids.size(); ++k)
      e = e | tf::csg::op(ids[k]);
    return e;
  };
  const auto e = cut.empty() ? union_of(kept) : union_of(kept) - union_of(cut);
  return tb_enclosed(tf::test::csg_mesh_of(graph, e));
}

/// The exact volume of a union of boxes on the axes, read at their float
/// corners: the slabs between consecutive distinct x, each the area its
/// boxes cover on the y z grid of their own sides.
auto tb_axis_union_volume(const std::vector<tb_box> &boxes) -> double {
  struct span {
    std::array<double, 3> lo, hi;
  };
  std::vector<span> spans;
  for (const auto &box : boxes) {
    span b{{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
    const auto mesh = tb_mesh_of(box);
    for (const auto &point : mesh.points())
      for (std::size_t k = 0; k < 3; ++k) {
        b.lo[k] = std::min(b.lo[k], double(point[k]));
        b.hi[k] = std::max(b.hi[k], double(point[k]));
      }
    spans.push_back(b);
  }
  const auto cuts = [](std::size_t k, const std::vector<const span *> &in) {
    std::vector<double> at;
    for (const auto *b : in) {
      at.push_back(b->lo[k]);
      at.push_back(b->hi[k]);
    }
    std::sort(at.begin(), at.end());
    at.erase(std::unique(at.begin(), at.end()), at.end());
    return at;
  };
  std::vector<const span *> all;
  for (const auto &b : spans)
    all.push_back(&b);
  const auto xs = cuts(0, all);
  double total = 0;
  for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
    const double mid = 0.5 * (xs[i] + xs[i + 1]);
    std::vector<const span *> in;
    for (const auto *b : all)
      if (b->lo[0] < mid && mid < b->hi[0])
        in.push_back(b);
    const auto ys = cuts(1, in), zs = cuts(2, in);
    double area = 0;
    for (std::size_t y = 0; y + 1 < ys.size(); ++y)
      for (std::size_t z = 0; z + 1 < zs.size(); ++z) {
        const double my = 0.5 * (ys[y] + ys[y + 1]),
                     mz = 0.5 * (zs[z] + zs[z + 1]);
        for (const auto *b : in)
          if (b->lo[1] < my && my < b->hi[1] && b->lo[2] < mz &&
              mz < b->hi[2]) {
            area += (ys[y + 1] - ys[y]) * (zs[z + 1] - zs[z]);
            break;
          }
      }
    total += area * (xs[i + 1] - xs[i]);
  }
  return total;
}

/// Whether the kept cells' boundary is closed and consistently wound: every
/// triangle the cells leave unpaired stands once, and every directed edge of
/// those triangles is met by its reverse.
template <typename Tets> auto tb_boundary_closed(const Tets &tets) -> bool {
  std::map<std::array<int, 3>, int> faces;
  for (const auto &tet : tets.tetras()) {
    const int a = tet[0], b = tet[1], c = tet[2], d = tet[3];
    const int sides[4][3] = {{b, c, d}, {a, d, c}, {a, b, d}, {a, c, b}};
    for (const auto &side : sides) {
      std::array<int, 3> key{side[0], side[1], side[2]};
      int parity = 0;
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = i + 1; j < 3; ++j)
          parity += key[i] > key[j];
      std::sort(key.begin(), key.end());
      faces[key] += parity % 2 ? -1 : 1;
    }
  }
  std::map<std::pair<int, int>, int> edges;
  std::size_t boundary = 0;
  for (const auto &[key, count] : faces) {
    if (!count)
      continue;
    if (count != 1 && count != -1)
      return false;
    ++boundary;
    std::array<int, 3> face = key;
    if (count < 0)
      std::swap(face[1], face[2]);
    for (std::size_t k = 0; k < 3; ++k) {
      const int u = face[k], v = face[(k + 1) % 3];
      if (u < v)
        ++edges[{u, v}];
      else
        --edges[{v, u}];
    }
  }
  for (const auto &edge : edges)
    if (edge.second)
      return false;
  return boundary > 0;
}

/// The mesher's contract on boxes crossing in one surface, at one worker
/// and at the machine's: meshed with nothing refused or unresolved, the kept
/// cells' boundary closed and wound consistently, every cell positive on the
/// lattice, their volume within 1e-7 of `expected`, and the same cells at
/// both counts. The clause it breaks, empty where it holds.
auto tb_meshes_volume(const tb_mesh &surface, double expected) -> std::string {
  std::vector<std::size_t> cells;
  for (const std::size_t width :
       tf::test::worker_ladder({1, tf::test::machine_workers()})) {
    tbb::task_arena arena(static_cast<int>(width));
    const auto [tets, report, map] = arena.execute(
        [&surface] { return tf::test::build_tet_mesh_double(surface); });
    const std::string at = " at " + std::to_string(width) + " workers";
    if (report.status != tf::tet_mesh_status::meshed || !tets.size())
      return "not meshed" + at;
    if (report.refused_segments || report.refused_faces ||
        report.pairs_unresolved || report.failed_triangles)
      return "refused or unresolved" + at;
    if (report.flat || report.negative_cells)
      return std::to_string(report.flat) + " flat and " +
             std::to_string(report.negative_cells) + " negative cells" + at;
    if (!tb_boundary_closed(tets))
      return "boundary open" + at;
    double volume = 0;
    for (const auto &tet : tets.tetrahedra()) {
      double u[3], v[3], w[3];
      for (std::size_t k = 0; k < 3; ++k) {
        u[k] = double(tet[1][k]) - double(tet[0][k]);
        v[k] = double(tet[2][k]) - double(tet[0][k]);
        w[k] = double(tet[3][k]) - double(tet[0][k]);
      }
      const double cell = (u[0] * (v[1] * w[2] - v[2] * w[1]) -
                           u[1] * (v[0] * w[2] - v[2] * w[0]) +
                           u[2] * (v[0] * w[1] - v[1] * w[0])) /
                          6;
      volume += cell;
    }
    // the lattice floor: the input is read once on the lattice, a float
    // corner moving by up to half a unit (5.4e-8 on the turned cubes)
    if (std::abs(volume - expected) > 1e-7 * std::abs(expected)) {
      char text[96];
      std::snprintf(text, sizeof text, "volume %.17g against %.17g, %.3e",
                    volume, expected,
                    std::abs(volume - expected) / std::abs(expected));
      return text + at;
    }
    cells.push_back(tets.size());
  }
  for (auto n : cells)
    if (n != cells[0])
      return "cell counts differ across worker counts";
  return {};
}

/// The contract against the csg tier's volume of the same boxes, which
/// agrees with `closed_form` where one is stated.
auto tb_meshes_boxes(const std::vector<tb_box> &boxes,
                     const std::vector<int> &inward = {},
                     double closed_form = -1) -> std::string {
  const double reference = tb_csg_volume(boxes, inward);
  // the closed form is the nominal boxes', the csg volume the float
  // corners': they agree to the input's own rounding
  if (closed_form >= 0 &&
      std::abs(reference - closed_form) > 1e-6 * closed_form)
    return "the csg volume " + std::to_string(reference) +
           " against the closed form";
  return tb_meshes_volume(tb_surface_of(boxes, inward), reference);
}

/// A rotation drawn uniformly through a normalized quaternion.
auto tb_rotation_of(std::mt19937_64 &rng) -> tb_rotation {
  double q[4];
  double norm = 0;
  for (auto &x : q) {
    x = tf::test::normal_real(rng, 0.0, 1.0);
    norm += x * x;
  }
  norm = std::sqrt(norm);
  for (auto &x : q)
    x /= norm;
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
          2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
          2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
}

/// `n` boxes of sides in [0.1, 0.3) about centres in [0.2, 0.8)^3, each
/// rotated generically about its centre where `rotated` says so, on the
/// axes otherwise.
auto tb_random_boxes(std::size_t n, std::uint64_t seed, bool rotated)
    -> std::vector<tb_box> {
  std::mt19937_64 rng(seed);
  std::vector<tb_box> boxes;
  for (std::size_t i = 0; i < n; ++i) {
    tb_box box;
    for (auto &s : box.size)
      s = tf::test::uniform_real(rng, 0.1, 0.3);
    for (auto &c : box.center)
      c = tf::test::uniform_real(rng, 0.2, 0.8);
    if (rotated)
      box.rotation = tb_rotation_of(rng);
    boxes.push_back(box);
  }
  return boxes;
}

auto tb_rotated_boxes(std::size_t n, std::uint64_t seed)
    -> std::vector<tb_box> {
  return tb_random_boxes(n, seed, true);
}

/// `boxes` on grids of their own, three, five, seven and eleven squares a
/// side in turn, so no two share a grid plane.
auto tb_generic_grids(std::vector<tb_box> boxes) -> std::vector<tb_box> {
  const int ticks[4] = {3, 5, 7, 11};
  for (std::size_t i = 0; i < boxes.size(); ++i)
    boxes[i].ticks = ticks[i % 4];
  return boxes;
}

/// The three unit cubes of the triple-point scenes, each face `ticks` by
/// `ticks`, all turned by one generic rotation about the origin.
auto tb_turned_cubes(int ticks) -> std::vector<tb_box> {
  std::mt19937_64 rng(7);
  const auto rotation = tb_rotation_of(rng);
  const std::array<std::array<double, 3>, 3> centers{
      {{0, 0, 0}, {0.5, 0.4, 0.3}, {0.2, 0.6, 0.55}}};
  std::vector<tb_box> boxes;
  for (const auto &c : centers) {
    tb_box box{{1, 1, 1}, {0, 0, 0}, rotation, ticks};
    for (std::size_t k = 0; k < 3; ++k)
      box.center[k] = rotation[3 * k] * c[0] + rotation[3 * k + 1] * c[1] +
                      rotation[3 * k + 2] * c[2];
    boxes.push_back(box);
  }
  return boxes;
}

} // namespace

TEST_CASE("make_tet_mesh: two boxes overlapping on a corner mesh their union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes({{{2, 2, 2}, {0, 0, 0}}, {{2, 2, 2}, {1, 1, 1}}}, {},
                        15) == "");
}

TEST_CASE("make_tet_mesh: two bars crossing as a plus mesh their union",
          "[tetmesh][cut]") {
  // Flush, the bars share their top and bottom planes; through, the second
  // bar is thinner and passes through the first's sides alone.
  CHECK(tb_meshes_boxes({{{3, 1, 1}, {0, 0, 0}}, {{1, 3, 1}, {0, 0, 0}}}, {},
                        5) == "");
  CHECK(tb_meshes_boxes({{{3, 1, 1}, {0, 0, 0}}, {{1, 3, 0.6}, {0, 0, 0}}}, {},
                        4.2) == "");
}

TEST_CASE("make_tet_mesh: boxes overlapping on a shared face plane mesh their "
          "union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes({{{1, 1, 1}, {0, 0, 0}}, {{1, 1, 1}, {0.5, 0.5, 0}}},
                        {}, 1.75) == "");
}

TEST_CASE("make_tet_mesh: a box inside a box is kept or carved by its winding",
          "[tetmesh][cut]") {
  // Wound outward the inner box winds its inside twice and the cells there
  // are kept; wound inward it winds them zero times, a cavity.
  const std::vector<tb_box> nested{{{2, 2, 2}, {0, 0, 0}},
                                   {{1, 1, 1}, {0, 0, 0}}};
  CHECK(tb_meshes_boxes(nested, {}, 8) == "");
  CHECK(tb_meshes_boxes(nested, {1}, 7) == "");
}

TEST_CASE("make_tet_mesh: three cubes on the axes meeting at triple points "
          "mesh their union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes({{{1, 1, 1}, {0, 0, 0}},
                         {{1, 1, 1}, {0.5, 0.4, 0.3}},
                         {{1, 1, 1}, {0.2, 0.6, 0.55}}}) == "");
}

TEST_CASE("make_tet_mesh: three boxes meeting at generic triple points mesh "
          "their union",
          "[tetmesh][cut]") {
  // Three unit cubes turned by one generic rotation: a face of each meets a
  // face of both others at two points no face of the input names. Turned
  // back onto the axes the same scene meshes.
  CHECK(tb_meshes_boxes(tb_turned_cubes(1)) == "");
}

TEST_CASE("make_tet_mesh: sixteen rotated boxes mesh their union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes(tb_rotated_boxes(16, 20261108u + 16u)) == "");
}

TEST_CASE("make_tet_mesh: two subdivided boxes overlapping on a corner mesh "
          "their union",
          "[tetmesh][cut]") {
  // Faces of four by four squares, the offset off their grid, so the
  // crossings land inside the grid's triangles and on its edges.
  CHECK(tb_meshes_boxes({{{2, 2, 2}, {0, 0, 0}, tb_identity, 4},
                         {{2, 2, 2}, {0.5, 0.75, 1.25}, tb_identity, 4}},
                        {}, 16 - 1.5 * 1.25 * 0.75) == "");
}

TEST_CASE("make_tet_mesh: three subdivided cubes on the axes meeting at triple "
          "points mesh their union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes({{{1, 1, 1}, {0, 0, 0}, tb_identity, 4},
                         {{1, 1, 1}, {0.5, 0.4, 0.3}, tb_identity, 4},
                         {{1, 1, 1}, {0.2, 0.6, 0.55}, tb_identity, 4}}) == "");
}

TEST_CASE("make_tet_mesh: three subdivided cubes with unequal grids mesh their "
          "union",
          "[tetmesh][cut]") {
  // The turned cubes, each of its own size and grid, so no two share a grid
  // plane and every crossing is generic.
  auto boxes = tb_turned_cubes(1);
  const int ticks[3] = {3, 5, 7};
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    boxes[i].ticks = ticks[i];
    for (auto &x : boxes[i].size)
      x *= 1.0 + 0.0371 * double(i + 1);
  }
  CHECK(tb_meshes_boxes(boxes) == "");
}

TEST_CASE("make_tet_mesh: three subdivided cubes meeting at generic triple "
          "points mesh their union",
          "[tetmesh][cut][.]") {
  // Equal grids on equal cubes put their subdivision lines on one set of
  // planes, which the rotation's rounding leaves a few lattice units apart;
  // it does not mesh at the tip.
  CHECK(tb_meshes_boxes(tb_turned_cubes(4)) == "");
}

TEST_CASE("make_tet_mesh: sixteen boxes on the axes mesh their union",
          "[tetmesh][cut]") {
  const auto boxes = tb_random_boxes(16, 20261108u + 16u, false);
  const double exact = tb_axis_union_volume(boxes);
  CHECK(std::abs(tb_csg_volume(boxes, {}) - exact) <= 1e-12 * exact);
  CHECK(tb_meshes_boxes(boxes) == "");
}

TEST_CASE("make_tet_mesh: sixteen boxes on the axes on one grid mesh their "
          "union",
          "[tetmesh][cut]") {
  // Every face four by four squares: boxes sharing a plane share its grid
  // lines there.
  auto boxes = tb_random_boxes(16, 20261108u + 16u, false);
  for (auto &box : boxes)
    box.ticks = 4;
  CHECK(tb_meshes_boxes(boxes) == "");
}

TEST_CASE("make_tet_mesh: sixteen rotated boxes on grids of their own mesh "
          "their union",
          "[tetmesh][cut]") {
  CHECK(tb_meshes_boxes(
            tb_generic_grids(tb_rotated_boxes(16, 20261108u + 16u))) == "");
}

TEST_CASE("make_tet_mesh: sixty-four rotated boxes on grids of their own are "
          "refused",
          "[tetmesh][cut][.]") {
  // Refused on depth, not hidden: segments and faces whose frames run out
  // at the deep lattice's cap (tet_mesh_report::refused_depth); it waits for
  // the deeper frame. Seconds, so it runs on request, not in the suite.
  const auto boxes = tb_generic_grids(tb_rotated_boxes(64, 20261108u + 64u));
  const auto [tets, report, map] =
      tf::test::build_tet_mesh_double(tb_surface_of(boxes, {}));
  CHECK(report.status == tf::tet_mesh_status::refused);
  CHECK(report.refused_depth > 0);
}
