/**
 * @file test_nonmanifold_edge_rings.cpp
 * @brief The sectors around a source non-manifold edge: uncut and mixed
 *        sheets enter the radial ring, read through domains, bundles and
 *        the outer shell.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/clean/polygons.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/geometry/ensure_positive_orientation.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/reindex/concatenated.hpp>
#include <trueform/reindex/split_into_domains.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/topology/make_domain_labels.hpp>
#include <trueform/topology/make_non_manifold_edge_fans.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ring_index_t = int;
using ring_real_t = double;
using ring_mesh_t = tf::polygons_buffer<ring_index_t, ring_real_t, 3, 3>;

auto ring_translated(ring_mesh_t mesh, ring_real_t dx, ring_real_t dy,
                     ring_real_t dz) -> ring_mesh_t {
  for (auto p : mesh.points_buffer().points()) {
    p[0] = p[0] + dx;
    p[1] = p[1] + dy;
    p[2] = p[2] + dz;
  }
  return mesh;
}

auto ring_box(ring_real_t sx, ring_real_t sy, ring_real_t sz, ring_real_t dx,
              ring_real_t dy, ring_real_t dz) -> ring_mesh_t {
  auto box = tf::make_box_mesh<ring_index_t>(sx, sy, sz);
  tf::ensure_positive_orientation(box.polygons());
  return ring_translated(std::move(box), dx, dy, dz);
}

/// Two unit boxes welded along one vertical edge: the 4-sheet pinch.
auto ring_welded_boxes() -> ring_mesh_t {
  auto a = ring_box(1, 1, 1, 0, 0, 0);
  auto b = ring_box(1, 1, 1, 1, 1, 0);
  auto soup = tf::concatenated(a.polygons(), b.polygons());
  return tf::cleaned(soup.polygons());
}

template <typename Cells>
auto ring_sorted_volumes(const Cells &cells) -> std::vector<ring_real_t> {
  std::vector<ring_real_t> v;
  v.reserve(cells.size());
  for (auto &c : cells)
    v.push_back(std::abs(tf::signed_volume(c.polygons())));
  std::sort(v.begin(), v.end());
  return v;
}

/// Domain parity against the mesh tier: same cell count, same volume
/// multiset, every cell closed and manifold.
template <typename CellsA, typename CellsB>
void ring_check_parity(const CellsA &cells, const CellsB &ocells) {
  auto v1 = ring_sorted_volumes(cells);
  auto v2 = ring_sorted_volumes(ocells);
  REQUIRE(cells.size() == ocells.size());
  for (std::size_t i = 0; i < v1.size(); ++i)
    REQUIRE_THAT(v1[i], Catch::Matchers::WithinRel(v2[i], ring_real_t(0.02)));
  for (auto &c : cells) {
    REQUIRE(tf::is_closed(c.polygons()));
    REQUIRE(tf::is_manifold(c.polygons()));
  }
}

} // namespace

TEST_CASE("the all-uncut pinch edge is sectored", "[csg][nm-ring]") {
  auto welded = ring_welded_boxes();
  auto operand = tf::test::make_tagged_operand(welded);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  // one ring at the pinch edge: both outsides one domain, each inside
  // its own, and the two surface components one body
  CHECK(graph.descriptor().n_domains == 3);
  CHECK(graph.descriptor().n_bundles == 1);

  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto dl = tf::make_domain_labels(
      welded.polygons(), tf::domain_config::exclude_outer_shell |
                             tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(welded.polygons(), dl);
  ring_check_parity(cells, ocells);
  REQUIRE(cells.size() == 2);
}

TEST_CASE("the outer shell keeps both welded boxes", "[csg][nm-ring]") {
  auto welded = ring_welded_boxes();
  auto operand = tf::test::make_tagged_operand(welded);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto shell = tf::test::outer_shell_of(graph);
  REQUIRE(shell.faces().size() == welded.faces().size());
  REQUIRE_THAT(tf::signed_volume(shell.polygons()),
               Catch::Matchers::WithinRel(tf::signed_volume(welded.polygons()),
                                          ring_real_t(1e-9)));

  auto shell_operand = tf::test::make_tagged_operand(shell);
  auto shell_graph = tf::test::build_self_csg_graph(shell_operand.form(), {});
  auto shell_again = tf::test::outer_shell_of(shell_graph);
  REQUIRE(shell_again.faces().size() == shell.faces().size());
  REQUIRE_THAT(tf::signed_volume(shell_again.polygons()),
               Catch::Matchers::WithinRel(tf::signed_volume(shell.polygons()),
                                          ring_real_t(1e-9)));
}

TEST_CASE("a corner cutter leaves three sheets uncut and the ring still "
          "sectors",
          "[csg][nm-ring]") {
  auto welded = ring_welded_boxes();
  // crosses the diagonal of box A's y face, away from the pinch line:
  // exactly one of the four sheets at the pinch edge is cut
  auto cutter = ring_box(ring_real_t(0.5), ring_real_t(0.2), ring_real_t(0.5),
                         ring_real_t(0.1), ring_real_t(0.5), ring_real_t(0.1));
  auto welded_operand = tf::test::make_tagged_operand(welded);
  auto cutter_operand = tf::test::make_tagged_operand(cutter);
  std::vector<tf::test::form_t<ring_index_t, ring_real_t, 3>> forms{
      welded_operand.form(), cutter_operand.form()};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph);

  auto [arr_mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(welded_operand.form(), cutter_operand.form());
  auto clean = tf::cleaned(arr_mesh.polygons(), ring_real_t(1e-6));
  auto dl = tf::make_domain_labels(
      clean.polygons(), tf::domain_config::exclude_outer_shell |
                            tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  ring_check_parity(cells, ocells);
  REQUIRE(cells.size() == 4);
}

TEST_CASE("a degenerate sheet is skipped and the ring stands",
          "[csg][nm-ring]") {
  auto welded = ring_welded_boxes();
  auto fans = tf::make_non_manifold_edge_fans(welded.polygons());
  REQUIRE(fans.edges.size() == 1);
  const auto i = fans.edges[0][0];
  const auto j = fans.edges[0][1];
  const auto pi = welded.points()[std::size_t(i)];
  const auto pj = welded.points()[std::size_t(j)];
  // a fifth sheet collapsed onto the pinch line: its third vertex sits
  // on the edge's own line, past the segment, so it states no plane and
  // no contact
  const auto m = ring_index_t(welded.points().size());
  welded.points_buffer().push_back(tf::point<ring_real_t, 3>{
      pj[0] + (pj[0] - pi[0]), pj[1] + (pj[1] - pi[1]),
      pj[2] + (pj[2] - pi[2])});
  welded.faces_buffer().push_back(std::array<ring_index_t, 3>{i, j, m});

  auto operand = tf::test::make_tagged_operand(welded);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph);
  REQUIRE(cells.size() == 2);
  auto volumes = ring_sorted_volumes(cells);
  for (auto v : volumes)
    REQUIRE_THAT(v, Catch::Matchers::WithinRel(ring_real_t(1), //
                                               ring_real_t(0.02)));
  for (auto &c : cells) {
    REQUIRE(tf::is_closed(c.polygons()));
    REQUIRE(tf::is_manifold(c.polygons()));
  }
}
