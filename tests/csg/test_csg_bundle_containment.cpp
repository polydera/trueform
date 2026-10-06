/**
 * @file test_csg_bundle_containment.cpp
 * @brief Where a disconnected bundle sits, read off the cast that finds it.
 *
 * A bundle the arrangement never joined to anything -- a solid floating
 * inside another region, touching nothing -- is placed by one segment cast
 * against the other operands. Two facts decide it. The census must be
 * complete: a face the arrangement cut carries no component of its own, so
 * its transition is the piece the segment actually crossed. And ray parity
 * states a DIFFERENCE, not a membership: crossing the boundary of a domain
 * an odd number of times says the segment's two ends disagree about it, so
 * the region holding the far end reads odd from everywhere and is never the
 * container.
 *
 * The lens scene is built so every wall the floating ball's ray can leave
 * through is cut by the other operand; the controls are the same ball where
 * its exit wall is whole, and behind a sheet, where the region that holds
 * it is unbounded and the right answer anyway.
 *
 * The census sees every wall the segment crosses, whatever form carries it:
 * a sheet severing the region the bundle floats in, and a form whose box
 * misses the bundle's. A flap -- an open sheet component -- is no wall, and
 * the regions on its two sides are never a landing: they differ only by
 * sheet side, which the winding answers.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

using containment_index_t = int;
using containment_real_t = double;
using containment_mesh_t =
    tf::polygons_buffer<containment_index_t, containment_real_t, 3, 3>;
using containment_operand_t =
    tf::test::tagged_operand<containment_index_t, containment_real_t>;
using containment_form_t =
    tf::test::form_t<containment_index_t, containment_real_t, 3>;

auto containment_box(containment_real_t x0, containment_real_t y0,
                     containment_real_t z0, containment_real_t x1,
                     containment_real_t y1, containment_real_t z1)
    -> containment_mesh_t {
  containment_mesh_t mesh;
  const containment_real_t c[8][3] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0},
                                      {x0, y1, z0}, {x0, y0, z1}, {x1, y0, z1},
                                      {x1, y1, z1}, {x0, y1, z1}};
  for (const auto &p : c)
    mesh.points_buffer().emplace_back(p[0], p[1], p[2]);
  const containment_index_t f[12][3] = {
      {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {3, 0, 4}, {3, 4, 7}};
  for (const auto &t : f)
    mesh.faces_buffer().emplace_back(t[0], t[1], t[2]);
  return mesh;
}

/// A square in the plane z, half-extent e; `up` picks the normal, so one
/// geometry can be written with either side declared the sheet's back.
auto containment_sheet(containment_real_t z, containment_real_t e, bool up)
    -> containment_mesh_t {
  containment_mesh_t mesh;
  const containment_real_t p[4][3] = {
      {-e, -e, z}, {e, -e, z}, {e, e, z}, {-e, e, z}};
  for (const auto &q : p)
    mesh.points_buffer().emplace_back(q[0], q[1], q[2]);
  if (up) {
    mesh.faces_buffer().emplace_back(0, 1, 2);
    mesh.faces_buffer().emplace_back(0, 2, 3);
  } else {
    mesh.faces_buffer().emplace_back(0, 2, 1);
    mesh.faces_buffer().emplace_back(0, 3, 2);
  }
  return mesh;
}

/// The rectangle [x0, x1] x [y0, y1] on the plane z + slope * x, normal up.
auto containment_rect_sheet(containment_real_t x0, containment_real_t x1,
                            containment_real_t y0, containment_real_t y1,
                            containment_real_t z, containment_real_t slope)
    -> containment_mesh_t {
  containment_mesh_t mesh;
  const containment_real_t p[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
  for (const auto &q : p)
    mesh.points_buffer().emplace_back(q[0], q[1], z + slope * q[0]);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  return mesh;
}

/// The ball: 0.3 on a side, volume 0.027. It sits off the diagonal so the
/// segment to the far corner leaves through one wall rather than an edge.
auto containment_ball() -> containment_mesh_t {
  return containment_box(3.35, 1.25, 1.95, 3.65, 1.55, 2.25);
}

struct containment_scene {
  std::vector<containment_operand_t> operands;
  std::vector<containment_form_t> forms;
  std::vector<int> sheets;
  decltype(tf::test::build_range_csg_graph(
      tf::test::forms_range(std::declval<std::vector<containment_form_t> &>()),
      tf::test::no_sheets(), tf::arrangement_config{})) graph;

  containment_scene(std::vector<containment_mesh_t> meshes,
                    std::vector<int> declared)
      : operands(make_operands(std::move(meshes))),
        forms(tf::test::tagged_forms(operands)), sheets(std::move(declared)),
        graph(tf::test::build_range_csg_graph(
            tf::test::forms_range(forms), tf::test::sheets_of(sheets), {})) {}

  containment_scene(const containment_scene &) = delete;
  auto operator=(const containment_scene &) -> containment_scene & = delete;

private:
  static auto make_operands(std::vector<containment_mesh_t> meshes)
      -> std::vector<containment_operand_t> {
    std::vector<containment_operand_t> operands;
    for (auto &mesh : meshes)
      operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
    return operands;
  }
};

/// Signed and sorted: a cell wound inward -- the unbounded outside emitted
/// as a domain -- sorts first and is negative.
template <typename Cells> auto containment_volumes(const Cells &cells) {
  std::vector<double> volumes;
  for (const auto &cell : cells)
    volumes.push_back(
        cell.size() == 0 ? 0.0 : double(tf::signed_volume(cell.polygons())));
  std::sort(volumes.begin(), volumes.end());
  return volumes;
}

/// The cells of a domain decomposition are disjoint and cover what the
/// operands enclose, so they are closed, positive, and sum to the whole.
template <typename Cells>
void containment_check(const Cells &cells,
                       const std::vector<double> &expected) {
  REQUIRE(cells.size() == expected.size());
  auto volumes = containment_volumes(cells);
  double total = 0.0, expected_total = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(volumes[i], Catch::Matchers::WithinAbs(expected[i], 1e-9));
    total += volumes[i];
    expected_total += expected[i];
  }
  REQUIRE_THAT(total, Catch::Matchers::WithinAbs(expected_total, 1e-9));
  for (const auto &cell : cells) {
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(tf::is_manifold(cell.polygons()));
  }
}

/// One shell of a cell: its signed volume (a cavity is wound inward) and
/// its box.
struct containment_shell {
  double volume;
  tf::aabb<double, 3> box;
};

template <typename Cell>
auto containment_shells_of(const Cell &cell)
    -> std::vector<containment_shell> {
  auto polygons = cell.polygons();
  REQUIRE(tf::is_closed(polygons));
  REQUIRE(tf::is_manifold(polygons));
  auto [labels, n] =
      tf::make_manifold_edge_connected_component_labels(polygons);
  auto [pieces, ids] = tf::split_into_components(polygons, labels);
  REQUIRE(pieces.size() == std::size_t(n));
  std::vector<containment_shell> shells;
  for (const auto &piece : pieces) {
    REQUIRE(tf::euler_characteristic(piece.polygons()) == 2);
    shells.push_back({double(tf::signed_volume(piece.polygons())),
                      tf::aabb_from(piece.points())});
  }
  return shells;
}

auto containment_box_holds(const tf::aabb<double, 3> &box,
                           const std::array<double, 3> &p) -> bool {
  for (int k = 0; k < 3; ++k)
    if (p[k] < double(box.min[k]) || p[k] > double(box.max[k]))
      return false;
  return true;
}

/// An expected domain, named by a point inside it, and its shells'
/// volumes.
struct containment_domain {
  std::array<double, 3> inside;
  std::vector<double> shells;
};

/// The cells are matched to the expected domains by content, never by the
/// order they come in: a domain's cell is the tightest one whose positive
/// shell's box holds the domain's point and none of whose cavities do.
template <typename Cells>
void containment_domains_check(
    const Cells &cells, const std::vector<containment_domain> &expected) {
  REQUIRE(cells.size() == expected.size());
  std::vector<std::vector<containment_shell>> shells;
  for (const auto &cell : cells)
    shells.push_back(containment_shells_of(cell));
  std::vector<char> matched(cells.size(), 0);
  for (const auto &domain : expected) {
    std::size_t best = cells.size();
    double best_volume = 0.0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      bool held = false, in_cavity = false;
      double volume = 0.0;
      for (const auto &shell : shells[i]) {
        const bool holds = containment_box_holds(shell.box, domain.inside);
        if (shell.volume > 0.0 && holds) {
          held = true;
          volume = shell.volume;
        }
        in_cavity = in_cavity || (shell.volume < 0.0 && holds);
      }
      if (held && !in_cavity &&
          (best == cells.size() || volume < best_volume)) {
        best = i;
        best_volume = volume;
      }
    }
    INFO("the domain holding (" << domain.inside[0] << ", "
                                << domain.inside[1] << ", "
                                << domain.inside[2] << ")");
    REQUIRE(best < cells.size());
    REQUIRE_FALSE(matched[best]);
    matched[best] = 1;
    std::vector<double> volumes;
    for (const auto &shell : shells[best])
      volumes.push_back(shell.volume);
    std::sort(volumes.begin(), volumes.end());
    auto wanted = domain.shells;
    std::sort(wanted.begin(), wanted.end());
    REQUIRE(volumes.size() == wanted.size());
    for (std::size_t k = 0; k < wanted.size(); ++k)
      REQUIRE_THAT(volumes[k], Catch::Matchers::WithinAbs(wanted[k], 1e-9));
  }
}

/// The open cell whose points satisfy `holds`; the index past the end when
/// none or several do.
template <typename Cells, typename Holds>
auto containment_open_cell(const Cells &cells, const Holds &holds)
    -> std::size_t {
  std::size_t found = cells.size();
  std::size_t n_found = 0;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (tf::is_closed(cells[i].polygons()))
      continue;
    bool any = false;
    for (auto p : cells[i].points())
      any = any || holds(double(p[0]), double(p[1]), double(p[2]));
    if (any) {
      found = i;
      ++n_found;
    }
  }
  return n_found == 1 ? found : cells.size();
}

/// Whether some point of the cell satisfies `at`.
template <typename Cell, typename At>
auto containment_cell_reaches(const Cell &cell, const At &at) -> bool {
  for (auto p : cell.points())
    if (at(double(p[0]), double(p[1]), double(p[2])))
      return true;
  return false;
}

/// The box [-5, 5]^3 the client scenes float their boxes in, and the
/// floating box itself. The far end of a seed's segment is the corner just
/// past the union of the operands' boxes, so from anywhere on this box the
/// segment meets the plane z = 2.5 inside the outer one.
auto containment_outer() -> containment_mesh_t {
  return containment_box(-5, -5, -5, 5, 5, 5);
}
auto containment_inner(containment_real_t dz) -> containment_mesh_t {
  return containment_box(-4, -4, -2 + dz, 0, 0, 2 + dz);
}

} // namespace

// ============================================================================
// A ball floating in the lens of two boxes. The lens is bounded by A's far
// wall and by B's side walls, and each of those is cut by the other operand,
// so every wall the ball's ray can leave through carries no component of its
// own. The ball belongs to the lens.
//   A = [0,4]^3 (64), B = [2,6]x[1,3]^2 (16), lens = [2,4]x[1,3]^2 (8),
//   ball = 0.027  ->  A-only 56, lens-minus-ball 7.973, B-only 8, ball 0.027
// ============================================================================
TEST_CASE("csg containment: a ball in the lens of two boxes belongs to it",
          "[csg][domains]") {
  containment_scene scene({containment_box(0, 0, 0, 4, 4, 4),
                           containment_box(2, 1, 1, 6, 3, 3),
                           containment_ball()},
                          {});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_check(kept.first, {0.027, 7.973, 8.0, 56.0});
}

// ============================================================================
// The same ball where its exit wall is whole: the census was never
// incomplete here, and the answer does not move.
// ============================================================================
TEST_CASE("csg containment: a ball in one box belongs to it",
          "[csg][domains]") {
  containment_scene scene(
      {containment_box(0, 0, 0, 4, 4, 4), containment_ball()}, {});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_check(kept.first, {0.027, 63.973});
}

// ============================================================================
// A sheet severs the box and splits the outside into two unbounded regions.
// A ball outside the box, off to one side, sits in one of them and its cast
// crosses nothing at all -- no volume stands between it and the far corner.
// The region it belongs to is then told by the only thing that tells those
// two regions apart: which side of the sheet they are on, which the winding
// pass already asked of this bundle's seed.
//
// Read with every domain kept, the outside cell carrying the ball's wall is
// the one on the ball's own side: it holds the box's top face and not its
// bottom when the ball is above, and the other way when it is below. The
// sheet's own winding renames the sides but does not move them, so all four
// readings state the same geometry.
// ============================================================================
TEST_CASE("csg containment: a ball beside a severing sheet joins its own side",
          "[csg][sheets][domains]") {
  struct placement_t {
    const char *name;
    containment_real_t z0;
    bool above;
  };
  const placement_t places[2] = {{"ball above the sheet", 3.35, true},
                                 {"ball below the sheet", 0.35, false}};
  for (const auto &place : places)
    for (int up = 1; up >= 0; --up) {
      DYNAMIC_SECTION(place.name << ", sheet normal " << (up ? "+z" : "-z")) {
        containment_scene scene(
            {containment_box(0, 0, 0, 4, 4, 4),
             containment_sheet(2.0, 8.0, up != 0),
             containment_box(5.85, 0.35, place.z0, 6.15, 0.65, place.z0 + 0.3)},
            {1});
        REQUIRE(scene.graph.failed().size() == 0);

        // The bounded reading is the box's two halves and the ball.
        auto kept = tf::test::csg_domains_of(scene.graph);
        containment_check(kept.first, {0.027, 32.0, 32.0});

        // The open reading says which outside the ball joined.
        auto all =
            tf::test::csg_domains_of(scene.graph, tf::domain_config::none);
        REQUIRE(all.first.size() == 5);
        std::size_t carriers = 0;
        bool holds_top = false, holds_bottom = false;
        for (const auto &cell : all.first) {
          if (tf::is_closed(cell.polygons()))
            continue; // the two unbounded regions are the open cells
          bool ball = false, top = false, bottom = false;
          for (auto p : cell.points()) {
            // the ball alone lives here: the box ends at 4 and the
            // sheet's only points past it are its corners at 8
            ball = ball || (double(p[0]) > 5.0 && double(p[0]) < 7.0);
            top = top || std::abs(double(p[2]) - 4.0) < 1e-9;
            bottom = bottom || std::abs(double(p[2])) < 1e-9;
          }
          if (!ball)
            continue;
          ++carriers;
          holds_top = top;
          holds_bottom = bottom;
        }
        REQUIRE(carriers == 1);
        REQUIRE(holds_top == place.above);
        REQUIRE(holds_bottom == !place.above);
      }
    }
}

// ============================================================================
// A sheet severs a box, and a box floats below it touching nothing. The
// segment from the floating box crosses the sheet inside the box before it
// leaves through the box's top, so the sheet is the first wall: the box
// floats in the lower half, which carries it as a cavity.
//   below 750 with the cavity -64, above 250, the floating box 64
// ============================================================================
TEST_CASE("csg containment: a box below a severing sheet is the lower "
          "half's cavity",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_outer(),
                           containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0),
                           containment_inner(0)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_domains_check(kept.first, {{{2, -2, -4}, {-64.0, 750.0}},
                                         {{0, 0, 4}, {250.0}},
                                         {{-2, -2, 0}, {64.0}}});
}

// ============================================================================
// The census rows: the same box under a wall that differs in kind, side, or
// pose. Each places the floating box where its own region is.
// ============================================================================
TEST_CASE("csg containment: the census rows place the floating box",
          "[csg][sheets][domains]") {
  SECTION("an undeclared plane is a wall whose box misses the floater's") {
    containment_scene scene(
        {containment_outer(),
         containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0),
         containment_inner(0)},
        {});
    REQUIRE(scene.graph.failed().size() == 0);
    auto kept = tf::test::csg_domains_of(scene.graph);
    containment_domains_check(kept.first,
                              {{{2, -2, -4}, {-64.0, 750.0}},
                               {{0, 0, 4}, {250.0}},
                               {{-2, -2, 0}, {64.0}}});
  }
  SECTION("a sheet below the floater is never on its segment") {
    containment_scene scene(
        {containment_outer(),
         containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, -2.5, 0),
         containment_inner(0)},
        {1});
    REQUIRE(scene.graph.failed().size() == 0);
    auto kept = tf::test::csg_domains_of(scene.graph);
    containment_domains_check(kept.first,
                              {{{2, -2, 0}, {-64.0, 750.0}},
                               {{0, 0, -4}, {250.0}},
                               {{-2, -2, 0}, {64.0}}});
  }
  SECTION("a tilted sheet whose box meets the floater's") {
    containment_scene scene(
        {containment_outer(),
         containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0.1),
         containment_inner(0)},
        {1});
    REQUIRE(scene.graph.failed().size() == 0);
    auto kept = tf::test::csg_domains_of(scene.graph);
    containment_domains_check(kept.first,
                              {{{2, -2, -4}, {-64.0, 750.0}},
                               {{0, 0, 4}, {250.0}},
                               {{-2, -2, 0}, {64.0}}});
  }
  SECTION("the floater a tenth below the sheet") {
    containment_scene scene(
        {containment_outer(),
         containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0),
         containment_inner(0.4)},
        {1});
    REQUIRE(scene.graph.failed().size() == 0);
    auto kept = tf::test::csg_domains_of(scene.graph);
    containment_domains_check(kept.first,
                              {{{2, -2, -4}, {-64.0, 750.0}},
                               {{0, 0, 4}, {250.0}},
                               {{-2, -2, 0}, {64.0}}});
  }
}

// ============================================================================
// A sheet that crosses one wall of the box and ends inside it divides
// nothing: the region wraps its free edge, so the box's interior is one
// domain holding both of the sheet's sides, and the floater is its cavity.
// ============================================================================
TEST_CASE("csg containment: a sheet ending inside the box divides nothing",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_outer(),
                           containment_rect_sheet(-3, 8, -3, 3, 2.5, 0),
                           containment_inner(0)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_domains_check(kept.first, {{{2, -2, -4}, {-64.0, 1000.0}},
                                         {{-2, -2, 0}, {64.0}}});
}

// ============================================================================
// The inner box stands on the outer box's floor. It touches, so it is no
// nesting: the lower half is one shell dented by it, and the same region
// the floating scene states as a shell and a cavity.
//   below 750 - 64 = 686, above 250, the standing box 64
// ============================================================================
TEST_CASE("csg containment: a box standing on the floor dents the lower half",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_outer(),
                           containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0),
                           containment_box(-4, -4, -5, 0, 0, -1)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_domains_check(kept.first, {{{2, -2, -4}, {686.0}},
                                         {{0, 0, 4}, {250.0}},
                                         {{-2, -2, -3}, {64.0}}});
}

// ============================================================================
// A closed box W straddles the sheet on the floater's segment, and its box
// misses the floater's. The segment enters W from the lower half, crosses
// the sheet inside W and leaves W into the upper half: W's walls are the
// first the floater meets, so the census needs them as much as the sheet's.
//   W = [2,4] x [1,3] x [2,3]: 2 on each side of the sheet
// ============================================================================
TEST_CASE("csg containment: a box straddling the sheet is a wall of the "
          "census",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_outer(),
                           containment_rect_sheet(-5.5, 5.5, -5.5, 5.5, 2.5, 0),
                           containment_inner(0),
                           containment_box(2, 1, 2, 4, 3, 3)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto kept = tf::test::csg_domains_of(scene.graph);
  containment_domains_check(kept.first, {{{2, -2, -4}, {-64.0, 748.0}},
                                         {{0, 0, 4}, {248.0}},
                                         {{3, 2, 2.25}, {2.0}},
                                         {{3, 2, 2.75}, {2.0}},
                                         {{-2, -2, 0}, {64.0}}});
}

// ============================================================================
// The sheet reaches far out of the box, and the floater hangs in the air
// below that flap. Its segment crosses the flap and then passes through a
// box Q that pierces the flap above it, entering and leaving Q on the
// upper side. The upper outside is named twice and holds the far end, so
// parity alone would elect it -- but a flap's sides are never a landing:
// the census elects nothing, and the winding puts the floater below.
// ============================================================================
TEST_CASE("csg containment: a floater under a flap stays on its own side",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_outer(),
                           containment_rect_sheet(-9, 9, -9, 9, 2.5, 0),
                           containment_box(6.5, -4.2, 2.4, 8, -2.5, 3.5),
                           containment_box(5.5, -7, 1, 6, -6.5, 1.5)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto all = tf::test::csg_domains_of(scene.graph, tf::domain_config::none);
  const auto carrier =
      containment_open_cell(all.first, [](double x, double y, double z) {
        return x > 5.4 && x < 6.1 && y > -7.1 && y < -6.4 && z > 0.9 &&
               z < 1.6;
      });
  REQUIRE(carrier < all.first.size());
  const auto &cell = all.first[carrier];
  REQUIRE(containment_cell_reaches(
      cell, [](double, double, double z) { return std::abs(z + 5) < 1e-9; }));
  REQUIRE_FALSE(containment_cell_reaches(
      cell, [](double, double, double z) { return std::abs(z - 5) < 1e-9; }));
}

// ============================================================================
// A sheet floating in the air divides nothing, and a floater behind it
// whose segment crosses it joins the outside the far box stands in.
// ============================================================================
TEST_CASE("csg containment: a free sheet is no wall for a floater behind it",
          "[csg][sheets][domains]") {
  containment_scene scene({containment_box(-20, -20, -20, -10, -10, -10),
                           containment_rect_sheet(-3, 3, -3, 3, 2, 0),
                           containment_box(-1.5, -1.5, 0, -1, -1, 0.5)},
                          {1});
  REQUIRE(scene.graph.failed().size() == 0);
  auto all = tf::test::csg_domains_of(scene.graph, tf::domain_config::none);
  const auto carrier =
      containment_open_cell(all.first, [](double x, double y, double z) {
        return x > -1.6 && x < -0.9 && y > -1.6 && y < -0.9 && z > -0.1 &&
               z < 0.6;
      });
  REQUIRE(carrier < all.first.size());
  REQUIRE(containment_cell_reaches(
      all.first[carrier],
      [](double x, double, double) { return std::abs(x + 20) < 1e-9; }));
}
