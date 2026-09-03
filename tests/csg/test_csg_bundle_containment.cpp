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
