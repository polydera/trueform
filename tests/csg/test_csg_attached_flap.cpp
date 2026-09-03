/**
 * @file test_csg_attached_flap.cpp
 * @brief A cutter that carries a flap across its own non-manifold edge.
 *
 * The cutter is a severing plane with a flap attached along an interior
 * edge of it, so that edge carries three faces. The fence refuses to fan a
 * non-manifold input edge, which leaves the flap its own bundle, and the
 * bundle's inclusion is then the seeding cast's alone. Undeclared, the
 * flap is a dangling fragment: it fuses away, the cube is the severing
 * plane's two halves, and the universe stays droppable.
 *
 * The sections are the same two flap triangles written six ways: four
 * corner ROTATIONS, which preserve the flap's winding, and two SWAPS,
 * which reverse it. The four rotations are one mesh under six names and
 * must answer alike; the two flips are a second mesh, and a dangling
 * fragment's own orientation cannot move the partition either.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/topology/non_manifold_edges.hpp>
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

using flap_index_t = int;
using flap_real_t = double;
using flap_mesh_t = tf::polygons_buffer<flap_index_t, flap_real_t, 3, 3>;
using flap_operand_t = tf::test::tagged_operand<flap_index_t, flap_real_t>;
using flap_form_t = tf::test::form_t<flap_index_t, flap_real_t, 3>;

auto flap_unit_cube() -> flap_mesh_t {
  flap_mesh_t mesh;
  const flap_real_t pts[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                 {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (const auto &p : pts)
    mesh.points_buffer().emplace_back(p[0], p[1], p[2]);
  const flap_index_t faces[12][3] = {
      {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {3, 0, 4}, {3, 4, 7}};
  for (const auto &f : faces)
    mesh.faces_buffer().emplace_back(f[0], f[1], f[2]);
  return mesh;
}

/// The flap's two triangles, one geometry written six ways. P=4 and Q=5
/// lie ON the severing plane; R=6 and S=7 do not.
struct flap_order_t {
  const char *name;
  flap_index_t triangle[2][3];
};

const flap_order_t flap_orders[6] = {
    {"rotation (P,Q,R),(P,R,S)", {{4, 5, 6}, {4, 6, 7}}},
    {"rotation (Q,R,P),(R,S,P)", {{5, 6, 4}, {6, 7, 4}}},
    {"rotation (R,P,Q),(S,P,R)", {{6, 4, 5}, {7, 4, 6}}},
    {"rotation (S,P,R),(R,P,Q)", {{7, 4, 6}, {6, 4, 5}}},
    {"flip (P,R,Q),(P,S,R)", {{4, 6, 5}, {4, 7, 6}}},
    {"flip (Q,P,R),(R,P,S)", {{5, 4, 6}, {6, 4, 7}}}};

/// The severing plane z=0.5, triangulated so that P-Q is one of its edges,
/// plus the flap P,Q,R,S standing on x=0.5 above it. P-Q then carries
/// three faces.
auto flap_cutter(const flap_order_t &order) -> flap_mesh_t {
  flap_mesh_t mesh;
  const flap_real_t pts[8][3] = {
      {-0.5, -0.5, 0.5}, {1.5, -0.5, 0.5}, {1.5, 1.5, 0.5}, {-0.5, 1.5, 0.5},
      {0.5, 0.3, 0.5},   {0.5, 0.7, 0.5},  {0.5, 0.7, 0.8}, {0.5, 0.3, 0.8}};
  for (const auto &p : pts)
    mesh.points_buffer().emplace_back(p[0], p[1], p[2]);
  const flap_index_t plane[6][3] = {{0, 1, 4}, {1, 5, 4}, {1, 2, 5},
                                    {2, 3, 5}, {3, 4, 5}, {3, 0, 4}};
  for (const auto &f : plane)
    mesh.faces_buffer().emplace_back(f[0], f[1], f[2]);
  for (const auto &f : order.triangle)
    mesh.faces_buffer().emplace_back(f[0], f[1], f[2]);
  return mesh;
}

auto flap_operands(const flap_order_t &order) -> std::vector<flap_operand_t> {
  std::vector<flap_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(flap_unit_cube()));
  operands.push_back(tf::test::make_tagged_operand(flap_cutter(order)));
  return operands;
}

struct flap_scene {
  std::vector<flap_operand_t> operands;
  std::vector<flap_form_t> forms;
  std::vector<int> sheets;
  decltype(tf::test::build_range_csg_graph(
      tf::test::forms_range(std::declval<std::vector<flap_form_t> &>()),
      tf::test::no_sheets(), tf::arrangement_config{})) graph;

  explicit flap_scene(const flap_order_t &order, std::vector<int> declared = {})
      : operands(flap_operands(order)), forms(tf::test::tagged_forms(operands)),
        sheets(std::move(declared)),
        graph(tf::test::build_range_csg_graph(
            tf::test::forms_range(forms), tf::test::sheets_of(sheets), {})) {}

  flap_scene(const flap_scene &) = delete;
  auto operator=(const flap_scene &) -> flap_scene & = delete;
};

/// Signed, so the unbounded outside emitted as a domain -- a cell wound
/// inward around everything -- cannot hide behind an absolute value.
template <typename Cells> auto flap_sorted_volumes(const Cells &cells) {
  std::vector<double> volumes;
  for (const auto &cell : cells)
    volumes.push_back(
        cell.size() == 0 ? 0.0 : double(tf::signed_volume(cell.polygons())));
  std::sort(volumes.begin(), volumes.end());
  return volumes;
}

template <typename Cells> void flap_check_cells(const Cells &cells) {
  for (const auto &cell : cells) {
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(tf::is_manifold(cell.polygons()));
  }
}

} // namespace

// ============================================================================
// The scene is the one the case claims: the cutter carries exactly one
// non-manifold edge, so the fence has something to refuse.
// ============================================================================
TEST_CASE("csg attached flap: the cutter's flap edge carries three faces",
          "[csg][domains]") {
  for (const auto &order : flap_orders) {
    DYNAMIC_SECTION(order.name) {
      auto cutter = flap_cutter(order);
      REQUIRE_FALSE(tf::is_manifold(cutter.polygons()));
      REQUIRE(tf::make_non_manifold_edges(cutter.polygons()).size() == 1);
    }
  }
}

// ============================================================================
// Undeclared: the flap is a dangling fragment, so the partition is the
// severing plane's alone and the universe is droppable — under every
// corner order of the flap's own triangles.
// ============================================================================
TEST_CASE("csg attached flap: the severed cube is two domains",
          "[csg][domains]") {
  for (const auto &order : flap_orders) {
    DYNAMIC_SECTION(order.name) {
      flap_scene scene(order);
      REQUIRE(scene.graph.failed().size() == 0);

      auto kept = tf::test::csg_domains_of(scene.graph);
      auto volumes = flap_sorted_volumes(kept.first);
      REQUIRE(kept.first.size() == 2);
      REQUIRE_THAT(volumes[0], Catch::Matchers::WithinAbs(0.5, 1e-9));
      REQUIRE_THAT(volumes[1], Catch::Matchers::WithinAbs(0.5, 1e-9));
      flap_check_cells(kept.first);

      // The universe is a domain like any other, and only
      // exclude_outer_shell drops it: with the flag off it comes back,
      // wound inward around the whole cube.
      auto all = tf::test::csg_domains_of(
          scene.graph, tf::domain_config::ignore_open_fragments);
      auto all_volumes = flap_sorted_volumes(all.first);
      REQUIRE(all.first.size() == 3);
      REQUIRE_THAT(all_volumes[0], Catch::Matchers::WithinAbs(-1.0, 1e-9));
    }
  }
}

// ============================================================================
// Declared a sheet, the same scene: the flap floats in the upper half, so
// that half is its container. The region beyond the cutter is unbounded and
// holds the far end of every cast, which is what once made it read as a
// container; the domains read is the severing plane's two halves and
// nothing else.
// ============================================================================
TEST_CASE("csg attached flap: the sheet reading keeps only the two halves",
          "[csg][sheets][domains]") {
  for (const auto &order : flap_orders) {
    DYNAMIC_SECTION(order.name) {
      flap_scene scene(order, {1});
      REQUIRE(scene.graph.failed().size() == 0);

      auto kept = tf::test::csg_domains_of(scene.graph);
      auto volumes = flap_sorted_volumes(kept.first);
      REQUIRE(kept.first.size() == 2);
      REQUIRE_THAT(volumes[0], Catch::Matchers::WithinAbs(0.5, 1e-9));
      REQUIRE_THAT(volumes[1], Catch::Matchers::WithinAbs(0.5, 1e-9));
      flap_check_cells(kept.first);
    }
  }
}
