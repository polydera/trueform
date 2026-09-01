/**
 * @file test_csg_domains.cpp
 * @brief Per-domain CSG extraction: read the N-ary csg_graph as
 *        watertight per-domain cells.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/csg/expression/operators.hpp>
#include <trueform/csg/graph/compute_domain_partition.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "arrangement_readers.hpp"
#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <set>

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using domains_index_t = int;
using domains_real_t = double;
using domains_mesh_t =
    tf::polygons_buffer<domains_index_t, domains_real_t, 3, 3>;

auto frame0() {
  return tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<domains_real_t, 3>{0, 0, 0}));
}

auto domains_frame_at(domains_real_t x, domains_real_t y, domains_real_t z) {
  return tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<domains_real_t, 3>{x, y, z}));
}

using domains_frame_t = decltype(frame0());
using domains_operand_t =
    tf::test::tagged_operand<domains_index_t, domains_real_t>;
using domains_form_t = tf::test::form_t<domains_index_t, domains_real_t, 3>;

/// The placement a frame states, as the operand carries it.
auto placement_of(const domains_frame_t &frame)
    -> tf::transformation<domains_real_t, 3> {
  return tf::transformation<domains_real_t, 3>(frame.transformation());
}

template <typename Cells> auto sorted_volumes(const Cells &cells) {
  std::vector<domains_real_t> v;
  v.reserve(cells.size());
  for (auto &c : cells)
    v.push_back(std::abs(tf::signed_volume(c.polygons())));
  std::sort(v.begin(), v.end());
  return v;
}

template <typename CellsA, typename CellsB>
void print_and_check_parity(const char *name, const CellsA &cells,
                            const CellsB &ocells) {
  auto v1 = sorted_volumes(cells);
  auto v2 = sorted_volumes(ocells);

  std::printf("[parity] %s\n", name);
  std::printf("  csg    count=%zu volumes=", cells.size());
  for (auto x : v1)
    std::printf("%.6f ", x);
  std::printf("\n  oracle count=%zu volumes=", ocells.size());
  for (auto x : v2)
    std::printf("%.6f ", x);
  std::printf("\n");

  REQUIRE(cells.size() == ocells.size());
  REQUIRE(v1.size() == v2.size());
  for (std::size_t i = 0; i < v1.size(); ++i)
    REQUIRE_THAT(v1[i],
                 Catch::Matchers::WithinRel(v2[i], domains_real_t(0.02)));

  for (auto &c : cells) {
    REQUIRE(tf::is_closed(c.polygons()));
    REQUIRE(tf::is_manifold(c.polygons()));
  }
  for (auto &c : ocells) {
    REQUIRE(tf::is_closed(c.polygons()));
    REQUIRE(tf::is_manifold(c.polygons()));
  }
}

} // namespace

TEST_CASE("compute_domain_partition keeps a label per kept domain side",
          "[domains]") {
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 24, 24);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});

  auto inc = graph.inclusion();
  auto blocks = inc.make_range();
  tf::buffer<bool> keep;
  keep.allocate(blocks.size());
  for (std::size_t d = 0; d < blocks.size(); ++d) {
    std::uint32_t any = 0;
    for (auto w : blocks[d])
      any |= w;
    keep[d] = any != 0;
  }

  auto part = tf::csg::graph::compute_domain_partition(graph.descriptor(), keep);

  std::size_t n_keep = 0;
  for (std::size_t d = 0; d < keep.size(); ++d)
    n_keep += keep[d];
  REQUIRE(part.n_kept == static_cast<domains_index_t>(n_keep));

  // The two bounded sphere cells (upper/lower hemisphere) carry the
  // sphere's operand bit (bit 0); the sheet also tags the open half-space
  // behind its normal, so the raw non-zero keep rule retains that too. The
  // bounded-only filter lives in the emission (Task 2), not here.
  std::size_t n_bounded = 0;
  for (std::size_t d = 0; d < blocks.size(); ++d)
    if ((blocks[d][0] & 0x1u) != 0)
      ++n_bounded;
  REQUIRE(n_bounded == 2);

  // dense_of_domain is a valid remap: kept -> [0,n_kept), dropped -> -1.
  for (std::size_t d = 0; d < keep.size(); ++d) {
    if (keep[d])
      REQUIRE(part.dense_of_domain[d] >= 0);
    else
      REQUIRE(part.dense_of_domain[d] == domains_index_t(-1));
  }
}

TEST_CASE("make_csg_domains splits a plane-cut sphere into two cells",
          "[domains]") {
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph);

  REQUIRE(cells.size() == 2);
  REQUIRE(ids.size() == 2);

  domains_real_t total_volume = 0;
  for (auto &cell : cells) {
    auto polys = cell.polygons();
    REQUIRE(tf::is_closed(polys));
    REQUIRE(tf::is_manifold(polys));
    total_volume += std::abs(tf::signed_volume(polys));
  }

  const domains_real_t ball =
      domains_real_t(4) / domains_real_t(3) * tf::pi<domains_real_t>;
  REQUIRE(std::abs(total_volume - ball) < domains_real_t(0.02) * ball);
}

TEST_CASE("make_csg_domains default config fuses sheet open halves",
          "[domains]") {
  // Headline change: under the default config (exclude_outer_shell |
  // ignore_open_fragments) the sheet's two open halves are merged into
  // the universe and dropped, leaving exactly the two closed hemispheres.
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph);

  REQUIRE(cells.size() == 2);
  REQUIRE(ids.size() == 2);

  int n_closed = 0;
  int n_open = 0;
  domains_real_t total_volume = 0;
  for (auto &cell : cells) {
    auto polys = cell.polygons();
    if (tf::is_closed(polys))
      ++n_closed;
    else
      ++n_open;
    REQUIRE(tf::is_manifold(polys));
    total_volume += std::abs(tf::signed_volume(polys));
  }
  REQUIRE(n_closed == 2);
  REQUIRE(n_open == 0);

  const domains_real_t ball =
      domains_real_t(4) / domains_real_t(3) * tf::pi<domains_real_t>;
  REQUIRE(std::abs(total_volume - ball) < domains_real_t(0.02) * ball);
}

TEST_CASE("make_csg_domains filters domains without merging them",
          "[domains]") {
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph, tf::csg::op(0));

  REQUIRE(cells.size() == 2);
  for (auto &cell : cells)
    REQUIRE(tf::is_closed(cell.polygons()));
}

TEST_CASE("make_csg_domains filter still selects hemispheres for a sheet",
          "[domains]") {
  // Sheet graph, default config + op(0): open halves fused away, only the
  // two closed sphere-interior hemispheres survive the filter.
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph, tf::csg::op(0));

  REQUIRE(cells.size() == 2);
  for (auto &cell : cells)
    REQUIRE(tf::is_closed(cell.polygons()));
}

TEST_CASE("make_csg_domains recovers the outer domain when not excluded",
          "[domains]") {
  // Sheet graph; opens merged (ignore_open_fragments) but the universe is
  // NOT excluded, so ~op(0) & ~op(1) selects the outside. Observed: 1 cell
  // (id 0), closed + manifold, signed_volume = -4.152 (inward-facing outer
  // shell of the [-1.5,1.5]^2 sheet box with the sphere carved out).
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});

  auto [cells, ids] =
      tf::test::csg_domains_of(graph, ~tf::csg::op(0) & ~tf::csg::op(1),
                               tf::domain_config::ignore_open_fragments);

  REQUIRE(cells.size() >= 1);
  bool found_outer = false;
  for (auto &cell : cells)
    if (tf::signed_volume(cell.polygons()) < domains_real_t(0))
      found_outer = true;
  REQUIRE(found_outer);
}

TEST_CASE("make_csg_domains raw config emits every arrangement domain",
          "[domains]") {
  // config = none: no open merge, no universe drop. Observed for the sheet
  // graph: 4 domains (2 closed hemispheres + 2 open sheet halves).
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph, tf::domain_config::none);

  REQUIRE(cells.size() == 4);
  REQUIRE(ids.size() == 4);

  int n_closed = 0;
  int n_open = 0;
  for (auto &cell : cells) {
    if (tf::is_closed(cell.polygons()))
      ++n_closed;
    else
      ++n_open;
  }
  REQUIRE(n_closed == 2);
  REQUIRE(n_open == 2);
}

TEST_CASE("make_csg_domains matches materialized split: two spheres",
          "[domains][parity]") {
  auto af = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto bf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto a_operand = tf::test::make_tagged_operand(af, placement_of(frame0()));
  auto a = a_operand.form();
  auto b_operand = tf::test::make_tagged_operand(
      bf, placement_of(domains_frame_at(domains_real_t(1), 0, 0)));
  auto b = b_operand.form();
  std::vector<domains_form_t> forms{a, b};

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);

  auto [arr_mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(a, b);
  auto clean = tf::cleaned(arr_mesh.polygons(), domains_real_t(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  print_and_check_parity("two overlapping spheres", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("make_csg_domains matches materialized split: sphere + plane sheet",
          "[domains][parity]") {
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(domains_real_t(3),
                                                 domains_real_t(3));
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand =
      tf::test::make_tagged_operand(pf, placement_of(frame0()));
  auto plane = plane_operand.form();
  std::vector<domains_form_t> forms{sphere, plane};

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::sheets_of({1}), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);

  auto [arr_mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(sphere, plane);
  auto clean = tf::cleaned(arr_mesh.polygons(), domains_real_t(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  print_and_check_parity("sphere + plane sheet", cells, ocells);
  REQUIRE(cells.size() == 2);
}

TEST_CASE("make_csg_domains matches materialized split: nested spheres",
          "[domains][parity]") {
  auto of = tf::make_sphere_mesh<domains_index_t>(domains_real_t(2), 32, 32);
  auto inf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto outer_operand =
      tf::test::make_tagged_operand(of, placement_of(frame0()));
  auto outer = outer_operand.form();
  auto inner_operand =
      tf::test::make_tagged_operand(inf, placement_of(frame0()));
  auto inner = inner_operand.form();
  std::vector<domains_form_t> forms{outer, inner};

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);

  auto [arr_mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(outer, inner);
  auto clean = tf::cleaned(arr_mesh.polygons(), domains_real_t(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  // Contact-free nested shells: the seeding-cast nesting merge
  // (seed_inclusion_bits nesting merge) fuses the false shell split, so the implicit
  // path now matches the materialized oracle - inner ball + shell.
  print_and_check_parity("nested spheres", cells, ocells);
  REQUIRE(cells.size() == 2);
}

namespace {

// Oracle for an N-form scene: merge the same tagged forms, clean, label
// domains under the default config, split into per-domain cells.
template <typename Range> auto oracle_domains(const Range &forms) {
  auto [arr_mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(forms);
  auto clean = tf::cleaned(arr_mesh.polygons(), domains_real_t(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  return tf::split_into_domains(clean.polygons(), dl);
}

auto sphere_form(domains_real_t r, domains_frame_t frame,
                 std::vector<domains_operand_t> &storage) -> domains_form_t {
  storage.push_back(tf::test::make_tagged_operand(
      tf::make_sphere_mesh<domains_index_t>(r, 32, 32), placement_of(frame)));
  return storage.back().form();
}

} // namespace

TEST_CASE("nesting: 3-level concentric spheres", "[domains][nesting]") {
  std::vector<domains_operand_t> storage;
  storage.reserve(3);
  std::vector<domains_form_t> forms;
  forms.push_back(sphere_form(domains_real_t(3), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(2), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(1), frame0(), storage));

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("3-level concentric", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("nesting: off-center fully nested sphere", "[domains][nesting]") {
  std::vector<domains_operand_t> storage;
  storage.reserve(2);
  std::vector<domains_form_t> forms;
  forms.push_back(sphere_form(domains_real_t(3), frame0(), storage));
  forms.push_back(sphere_form(
      domains_real_t(1), domains_frame_at(domains_real_t(1.2), 0, 0), storage));

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("off-center nested", cells, ocells);
  REQUIRE(cells.size() == 2);
}

TEST_CASE("nesting: two sibling spheres in one parent", "[domains][nesting]") {
  std::vector<domains_operand_t> storage;
  storage.reserve(3);
  std::vector<domains_form_t> forms;
  forms.push_back(sphere_form(domains_real_t(3), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(0.6),
                              domains_frame_at(domains_real_t(1.5), 0, 0),
                              storage));
  forms.push_back(sphere_form(domains_real_t(0.6),
                              domains_frame_at(domains_real_t(-1.5), 0, 0),
                              storage));

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("two siblings in parent", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("nesting: nested plus intersecting mix", "[domains][nesting]") {
  std::vector<domains_operand_t> storage;
  storage.reserve(3);
  std::vector<domains_form_t> forms;
  forms.push_back(sphere_form(domains_real_t(3), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(1), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(2.5),
                              domains_frame_at(domains_real_t(2.5), 0, 0),
                              storage));

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("nested + intersecting", cells, ocells);
}

TEST_CASE("nesting: deep 4-level concentric chain", "[domains][nesting]") {
  std::vector<domains_operand_t> storage;
  storage.reserve(4);
  std::vector<domains_form_t> forms;
  forms.push_back(sphere_form(domains_real_t(4), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(3), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(2), frame0(), storage));
  forms.push_back(sphere_form(domains_real_t(1), frame0(), storage));

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("4-level concentric", cells, ocells);
  REQUIRE(cells.size() == 4);
}

// An open cut whose free edge ends inside a closed volume makes a slit - the
// cut region runs in along one wall and back along the other, folding both
// walls into one edge incidence. That incidence must still separate: the
// sphere's interior and exterior are two domains, so the union stays the
// watertight sphere.
TEST_CASE("make_csg_domains: open partial wall (slit) does not collapse",
          "[domains][slit]") {
  auto sf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 32, 32);
  auto pf = tf::make_plane_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(4)); // partial wall
  auto sphere_operand =
      tf::test::make_tagged_operand(sf, placement_of(frame0()));
  auto sphere = sphere_operand.form();
  auto plane_operand = tf::test::make_tagged_operand(
      pf, placement_of(domains_frame_at(-1, 0, 0)));
  auto plane = plane_operand.form(); // free edge inside
  std::vector<domains_form_t> forms{sphere, plane};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  // The union solid must stay the watertight sphere, not collapse to empty.
  auto u = tf::test::csg_mesh_of(
      graph, tf::csg::any_of(tf::make_sequence_range(0, 2)));
  REQUIRE(tf::is_closed(u.polygons()));
  REQUIRE(tf::is_manifold(u.polygons()));
  const domains_real_t ball =
      domains_real_t(4) / domains_real_t(3) * tf::pi<domains_real_t>;
  const domains_real_t vol = std::abs(tf::signed_volume(u.polygons()));
  REQUIRE(std::abs(vol - ball) < domains_real_t(0.03) * ball);

  // The wall divides the interior: keeping open fragments yields >= 2 cells
  // (it was a single collapsed cell before the fix).
  auto [cells, ids] =
      tf::test::csg_domains_of(graph, tf::domain_config::ignore_open_fragments);
  REQUIRE(cells.size() >= 2);
}

// make_intersection_curves walks region-loop edges and emits the cross-tag
// seam network; two overlapping spheres meet along exactly one closed circle.
TEST_CASE("make_intersection_curves: two spheres meet on one closed loop",
          "[intersection_curves]") {
  auto af = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 24, 24);
  auto bf = tf::make_sphere_mesh<domains_index_t>(domains_real_t(1), 24, 24);
  auto a_operand = tf::test::make_tagged_operand(af, placement_of(frame0()));
  auto a = a_operand.form();
  auto b_operand = tf::test::make_tagged_operand(
      bf, placement_of(domains_frame_at(domains_real_t(0.8), 0, 0)));
  auto b = b_operand.form();
  std::vector<domains_form_t> forms{a, b};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto curves = tf::test::arrangement_curves_of(graph);
  std::size_t n_paths = 0, n_pts = 0;
  for (auto &&path : curves.paths()) {
    ++n_paths;
    n_pts += path.size();
  }
  REQUIRE(n_paths == 1); // a single intersection circle
  REQUIRE(n_pts > 3);    // a real polyline, not a degenerate point
}

// Bridged / patched-hole region: a cylinder drilled through the interior of a
// single (large) box face leaves that face an annulus - a hole inside the
// face boundary. The planar arrangement represents that as ONE loop with a
// bridge edge joining the hole to the outer boundary, so the bridge edge is
// carried twice. Its endpoints are ordinary vertices (not a flanked free tip
// like a slit), so the non-manifold-edge fan builder must *count* a loop's
// edge incidences, not flank-test for a tip. This pins clean handling of
// holed face regions through the boolean.
TEST_CASE("make_csg_mesh: cylinder drilled through a box face (bridged hole)",
          "[domains][bridge]") {
  auto bf = tf::make_box_mesh<domains_index_t>(
      domains_real_t(4), domains_real_t(4), domains_real_t(4));
  auto cf = tf::make_cylinder_mesh<domains_index_t>(domains_real_t(0.5),
                                                    domains_real_t(6), 32);
  auto box_operand = tf::test::make_tagged_operand(bf, placement_of(frame0()));
  auto box = box_operand.form();
  auto cyl_operand = tf::test::make_tagged_operand(
      cf, placement_of(domains_frame_at(1, -1, 0))); // offset into one face
  auto cyl = cyl_operand.form();
  std::vector<domains_form_t> forms{box, cyl};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto drilled = tf::test::csg_mesh_of(graph, tf::csg::op(0) & ~tf::csg::op(1));
  REQUIRE(tf::is_closed(drilled.polygons()));
  REQUIRE(tf::is_manifold(drilled.polygons()));
  // box (4^3 = 64) minus a cylinder bored straight through (pi r^2 * 4).
  const domains_real_t expect = domains_real_t(64) - tf::pi<domains_real_t> *
                                                         domains_real_t(0.25) *
                                                         domains_real_t(4);
  const domains_real_t vol = std::abs(tf::signed_volume(drilled.polygons()));
  REQUIRE(std::abs(vol - expect) <
          domains_real_t(0.1)); // 32-segment facet tolerance
}

TEST_CASE("make_csg_domains return_source_ids: per-cell tag + face provenance",
          "[domains][source_ids]") {
  // Two overlapping cubes (side 2, offset by 1,1,1) -> three solid cells.
  auto af = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  auto bf = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  for (std::size_t i = 0; i < bf.points_buffer().size(); ++i) {
    auto p = bf.points_buffer()[i];
    bf.points_buffer()[i] = tf::point<domains_real_t, 3>{
        p[0] + domains_real_t(1), p[1] + domains_real_t(1),
        p[2] + domains_real_t(1)};
  }
  auto a_operand = tf::test::make_tagged_operand(af, placement_of(frame0()));
  auto a = a_operand.form();
  auto b_operand = tf::test::make_tagged_operand(bf, placement_of(frame0()));
  auto b = b_operand.form();
  std::vector<domains_form_t> forms{a, b};
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto [cells0, ids0] = tf::test::csg_domains_of(graph);
  auto [cells, ids, tag_blocks, face_blocks] =
      tf::make_csg_domains(graph, tf::return_source_ids);

  // Provenance blocks run parallel to cells; the cells match the plain build.
  REQUIRE(cells.size() == cells0.size());
  REQUIRE(std::size_t(tag_blocks.size()) == cells.size());
  REQUIRE(std::size_t(face_blocks.size()) == cells.size());

  bool saw0 = false, saw1 = false;
  for (std::size_t c = 0; c < cells.size(); ++c) {
    auto tblk = tag_blocks[c];
    auto fblk = face_blocks[c];
    // One (tag, face) per cell face.
    REQUIRE(std::size_t(tblk.size()) == cells[c].polygons().faces().size());
    REQUIRE(std::size_t(fblk.size()) == cells[c].polygons().faces().size());
    for (std::size_t j = 0; j < std::size_t(tblk.size()); ++j) {
      auto t = tblk[j];
      auto fl = fblk[j];
      REQUIRE((t == domains_index_t(0) || t == domains_index_t(1)));
      saw0 = saw0 || (t == domains_index_t(0));
      saw1 = saw1 || (t == domains_index_t(1));
      REQUIRE(fl >= domains_index_t(0));
      REQUIRE(std::size_t(fl) < forms[std::size_t(t)].faces().size());
      // Order-independent correctness: cell face j lies in the plane of its
      // claimed source face (frame is identity here).
      auto src_plane = tf::make_plane(forms[std::size_t(t)][std::size_t(fl)]);
      for (auto v : cells[c].polygons()[j]) {
        auto d = double(tf::distance(src_plane, v));
        REQUIRE(d < 1e-3);
        REQUIRE(d > -1e-3);
      }
    }
  }
  // Three cells over two overlapping cubes use faces from both operands.
  REQUIRE(saw0);
  REQUIRE(saw1);
}

TEST_CASE("make_csg_domains return_source_ids: dynamic-arity (quad) provenance",
          "[domains][source_ids]") {
  // Quad-cube input exercises the dynamic (non-triangle) cell path: uncut
  // faces stay quads, cut faces become triangles.
  using qmesh_t = tf::polygons_buffer<domains_index_t, domains_real_t, 3, 4>;
  auto quad_cube = [](domains_real_t s, domains_real_t ox, domains_real_t oy,
                      domains_real_t oz) {
    qmesh_t m;
    m.points_buffer().allocate(8);
    domains_index_t idx = 0;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          m.points_buffer()[std::size_t(idx++)] = tf::point<domains_real_t, 3>{
              ox + domains_real_t(x) * s, oy + domains_real_t(y) * s,
              oz + domains_real_t(z) * s};
    // Outward-oriented quad faces (vertex bits: x=1, y=2, z=4).
    const std::array<std::array<domains_index_t, 4>, 6> faces = {
        {{0, 2, 3, 1},
         {4, 5, 7, 6},
         {0, 1, 5, 4},
         {2, 6, 7, 3},
         {0, 4, 6, 2},
         {1, 3, 7, 5}}};
    m.faces_buffer().allocate(6);
    for (std::size_t i = 0; i < 6; ++i)
      m.faces_buffer()[i] = faces[i];
    return m;
  };
  auto af = quad_cube(domains_real_t(2), 0, 0, 0);
  auto bf = quad_cube(domains_real_t(2), domains_real_t(1), domains_real_t(1),
                      domains_real_t(1));
  std::vector<decltype(tf::test::make_tagged_operand(af))> forms_operands;
  forms_operands.reserve(2);
  forms_operands.push_back(
      tf::test::make_tagged_operand(af, placement_of(frame0())));
  forms_operands.push_back(
      tf::test::make_tagged_operand(bf, placement_of(frame0())));
  auto forms = tf::test::tagged_forms(forms_operands);
  auto qgraph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                                tf::test::no_sheets(), {});

  // plain call first: the dynamic emit must not touch the (empty)
  // provenance machinery when no labels are requested
  {
    auto [pcells, pids] = tf::test::csg_domains_of(qgraph);
    REQUIRE(pcells.size() >= std::size_t(1));
  }

  auto [cells, ids, tag_blocks, face_blocks] =
      tf::make_csg_domains(qgraph, tf::return_source_ids);

  REQUIRE(cells.size() >= std::size_t(1));
  REQUIRE(std::size_t(tag_blocks.size()) == cells.size());
  REQUIRE(std::size_t(face_blocks.size()) == cells.size());
  for (std::size_t c = 0; c < cells.size(); ++c) {
    auto tblk = tag_blocks[c];
    auto fblk = face_blocks[c];
    REQUIRE(std::size_t(tblk.size()) == cells[c].polygons().faces().size());
    REQUIRE(std::size_t(fblk.size()) == cells[c].polygons().faces().size());
    for (std::size_t j = 0; j < std::size_t(tblk.size()); ++j) {
      auto t = tblk[j];
      auto fl = fblk[j];
      REQUIRE((t == domains_index_t(0) || t == domains_index_t(1)));
      REQUIRE(fl >= domains_index_t(0));
      REQUIRE(std::size_t(fl) < forms[std::size_t(t)].faces().size());
      auto src_plane = tf::make_plane(forms[std::size_t(t)][std::size_t(fl)]);
      for (auto v : cells[c].polygons()[j]) {
        auto d = double(tf::distance(src_plane, v));
        REQUIRE(d < 1e-3);
        REQUIRE(d > -1e-3);
      }
    }
  }
}

TEST_CASE("make_csg_domains: coincident-plane cubes emit no phantom domains",
          "[domains][subulp]") {
  // Two 2x2x2 cubes offset by 1 in x share the y=+-1 and z=+-1 planes, and
  // a knife plane bisects everything at z=0. Rim corners lying exactly in
  // a coincident wall plane construct the same triple point through
  // several primitive pairs; without the sub-ulp record fuse the twins
  // survive as 2-vertex sliver loops whose private domains emit empty
  // cells. Expect exactly the 6 real cells, each closed with volume 2.
  auto c0 = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  auto c1 = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  auto plane = tf::make_plane_mesh<domains_index_t>(domains_real_t(4),
                                                    domains_real_t(4));

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<domains_real_t, 3>{domains_real_t(-0.5), 0, 0}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<domains_real_t, 3>{domains_real_t(0.5), 0, 0}));
  std::vector<domains_operand_t> forms_operands;
  forms_operands.reserve(3);
  forms_operands.push_back(tf::test::make_tagged_operand(c0, placement_of(f0)));
  forms_operands.push_back(tf::test::make_tagged_operand(c1, placement_of(f1)));
  forms_operands.push_back(
      tf::test::make_tagged_operand(plane, placement_of(frame0())));
  auto forms = tf::test::tagged_forms(forms_operands);

  for (bool with_sheet : {true, false}) {
    DYNAMIC_SECTION((with_sheet ? "knife as sheet" : "knife as volume")) {
      const std::vector<int> sheets{2};
      auto graph =
          with_sheet
              ? tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                                tf::test::sheets_of(sheets), {})
              : tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                                tf::test::no_sheets(), {});
      auto [cells, ids] = tf::test::csg_domains_of(graph);
      REQUIRE(cells.size() == 6);
      for (auto &cell : cells) {
        REQUIRE(cell.polygons().size() > 0);
        REQUIRE(tf::is_closed(cell.polygons()));
        REQUIRE_THAT(std::abs(double(tf::signed_volume(cell.polygons()))),
                     Catch::Matchers::WithinAbs(2.0, 1e-3));
      }
    }
  }
}

TEST_CASE("make_csg_domains return_index_map: per-cell point + face maps",
          "[domains][index_map]") {
  // Two overlapping cubes -> three cells; verify the per-cell index map.
  auto af = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  auto bf = tf::make_box_mesh<domains_index_t>(
      domains_real_t(2), domains_real_t(2), domains_real_t(2));
  for (std::size_t i = 0; i < bf.points_buffer().size(); ++i) {
    auto p = bf.points_buffer()[i];
    bf.points_buffer()[i] = tf::point<domains_real_t, 3>{
        p[0] + domains_real_t(1), p[1] + domains_real_t(1),
        p[2] + domains_real_t(1)};
  }
  std::vector<decltype(tf::test::make_tagged_operand(af))> forms_operands;
  forms_operands.reserve(2);
  forms_operands.push_back(
      tf::test::make_tagged_operand(af, placement_of(frame0())));
  forms_operands.push_back(
      tf::test::make_tagged_operand(bf, placement_of(frame0())));
  auto forms = tf::test::tagged_forms(forms_operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto [cells, ids, imap] = tf::make_csg_domains(graph, tf::return_index_map);

  REQUIRE(imap.n_tags == domains_index_t(forms.size()));
  REQUIRE(std::size_t(imap.face_tag_blocks.size()) == cells.size());
  REQUIRE(std::size_t(imap.face_blocks.size()) == cells.size());
  REQUIRE(std::size_t(imap.point_tag_blocks.size()) == cells.size());
  REQUIRE(std::size_t(imap.point_blocks.size()) == cells.size());

  for (std::size_t k = 0; k < cells.size(); ++k) {
    auto polys = cells[k].polygons();

    // Face maps parallel to cell faces; each a valid (form, face).
    auto ftag = imap.face_tag_blocks[k];
    auto fid = imap.face_blocks[k];
    REQUIRE(std::size_t(ftag.size()) == polys.faces().size());
    REQUIRE(std::size_t(fid.size()) == polys.faces().size());
    for (std::size_t j = 0; j < std::size_t(ftag.size()); ++j) {
      REQUIRE((ftag[j] >= domains_index_t(0) &&
               std::size_t(ftag[j]) < forms.size()));
      REQUIRE(std::size_t(fid[j]) < forms[std::size_t(ftag[j])].faces().size());
    }

    // Point maps parallel to cell points. A kept original must equal its
    // claimed input point (identity frame); a created point carries both
    // end sentinels.
    auto ptag = imap.point_tag_blocks[k];
    auto pid = imap.point_blocks[k];
    REQUIRE(std::size_t(ptag.size()) == polys.points().size());
    REQUIRE(std::size_t(pid.size()) == polys.points().size());
    for (std::size_t p = 0; p < std::size_t(ptag.size()); ++p) {
      if (ptag[p] == imap.n_tags) {
        REQUIRE(pid[p] == imap.n_output_points);
        continue;
      }
      REQUIRE((ptag[p] >= domains_index_t(0) &&
               std::size_t(ptag[p]) < forms.size()));
      REQUIRE(std::size_t(pid[p]) <
              forms[std::size_t(ptag[p])].points().size());
      auto cp = polys.points()[p];
      auto ip = forms[std::size_t(ptag[p])].points()[std::size_t(pid[p])];
      REQUIRE(std::abs(double(cp[0]) - double(ip[0])) < 1e-9);
      REQUIRE(std::abs(double(cp[1]) - double(ip[1])) < 1e-9);
      REQUIRE(std::abs(double(cp[2]) - double(ip[2])) < 1e-9);
    }
  }

  // Inclusion matrix: one row per cell, one column per form; each column
  // must agree with the expression-filtered query (ids are stable across
  // queries on one graph + config).
  REQUIRE(imap.inclusion.block_size() == forms.size());
  REQUIRE(std::size_t(imap.inclusion.size()) == cells.size());
  const auto &inc = imap.inclusion;
  auto [a_cells, a_ids] = tf::test::csg_domains_of(graph, tf::csg::op(0));
  auto [b_cells, b_ids] = tf::test::csg_domains_of(graph, tf::csg::op(1));
  auto has = [](const auto &dom_ids, domains_index_t id) {
    for (auto v : dom_ids)
      if (v == id)
        return true;
    return false;
  };
  for (std::size_t k = 0; k < cells.size(); ++k) {
    REQUIRE((inc[k][0] != 0) == has(a_ids, ids[k]));
    REQUIRE((inc[k][1] != 0) == has(b_ids, ids[k]));
  }
}

namespace {

auto sphere_at(domains_real_t cx, domains_real_t cy, domains_real_t cz,
               domains_real_t r = domains_real_t(1)) -> domains_mesh_t {
  domains_mesh_t m = tf::make_sphere_mesh<domains_index_t>(r, 32, 32);
  tf::ensure_positive_orientation(m.polygons());
  auto &p = m.points_buffer();
  for (std::size_t i = 0; i < p.size(); ++i) {
    auto q = p[i];
    p[i] = tf::point<domains_real_t, 3>{q[0] + cx, q[1] + cy, q[2] + cz};
  }
  return m;
}

constexpr auto within_config = tf::intersect_config{
    tf::intersect_mode::primitives |
    tf::intersect_mode::resolve_crossing_contours |
    tf::intersect_mode::within};

template <typename VolsA, typename VolsB>
void require_same_volumes(const VolsA &a, const VolsB &b) {
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    REQUIRE_THAT(a[i], Catch::Matchers::WithinRel(b[i], 1e-9));
}

} // namespace

TEST_CASE("make_csg_domains within: concatenated operand matches separate "
          "operands",
          "[domains][within]") {
  // Three mutually intersecting spheres. Reference: three operands.
  // Test: two of them concatenated into ONE self-overlapping operand,
  // arranged against the third with the within bit. The double-covered
  // pocket (inside both B and C, outside A) must survive the parity read.
  auto A = sphere_at(0, 0, 0);
  auto B = sphere_at(domains_real_t(0.9), 0, 0);
  auto C = sphere_at(domains_real_t(0.45), domains_real_t(0.8), 0);

  std::vector<domains_operand_t> three_operands;
  three_operands.reserve(3);
  three_operands.push_back(
      tf::test::make_tagged_operand(A, placement_of(frame0())));
  three_operands.push_back(
      tf::test::make_tagged_operand(B, placement_of(frame0())));
  three_operands.push_back(
      tf::test::make_tagged_operand(C, placement_of(frame0())));
  auto three = tf::test::tagged_forms(three_operands);
  auto g3 = tf::test::build_range_csg_graph(tf::test::forms_range(three),
                                            tf::test::no_sheets(), {});
  auto [cells3, ids3] = tf::test::csg_domains_of(g3);
  auto v3 = sorted_volumes(cells3);
  REQUIRE(cells3.size() == 7);

  auto bc = tf::concatenated(B.polygons(), C.polygons());
  std::vector<domains_operand_t> two_operands;
  two_operands.reserve(2);
  two_operands.push_back(
      tf::test::make_tagged_operand(bc, placement_of(frame0())));
  two_operands.push_back(
      tf::test::make_tagged_operand(A, placement_of(frame0())));
  auto two = tf::test::tagged_forms(two_operands);
  auto g2 = tf::test::build_range_csg_graph(
      tf::test::forms_range(two), tf::test::no_sheets(), within_config);
  auto [cells2, ids2] = tf::test::csg_domains_of(g2);
  auto v2 = sorted_volumes(cells2);

  require_same_volumes(v3, v2);
}

TEST_CASE("make_csg_domains within: flag on clean operands changes nothing",
          "[domains][within]") {
  // No self-overlap anywhere: the within bit must not change the cells.
  auto A = sphere_at(0, 0, 0);
  auto B = sphere_at(domains_real_t(1), 0, 0);

  std::vector<domains_operand_t> forms_operands;
  forms_operands.reserve(2);
  forms_operands.push_back(
      tf::test::make_tagged_operand(A, placement_of(frame0())));
  forms_operands.push_back(
      tf::test::make_tagged_operand(B, placement_of(frame0())));
  auto forms = tf::test::tagged_forms(forms_operands);
  auto g_plain = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                                 tf::test::no_sheets(), {});
  auto [cells_p, ids_p] = tf::test::csg_domains_of(g_plain);

  auto g_within = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::no_sheets(), within_config);
  auto [cells_w, ids_w] = tf::test::csg_domains_of(g_within);

  REQUIRE(cells_p.size() == 3);
  require_same_volumes(sorted_volumes(cells_p), sorted_volumes(cells_w));
}

TEST_CASE("make_csg_domains within: nested pair concatenated into one "
          "operand keeps the cavity",
          "[domains][within][nesting]") {
  // Hollow operand: outer + inner sphere concatenated (nested, no
  // crossing), plus a far disjoint sphere as the second operand.
  // Reference: the same three as separate operands.
  auto outer = sphere_at(0, 0, 0);
  auto inner = sphere_at(0, 0, 0, domains_real_t(0.5));
  auto far_sphere = sphere_at(domains_real_t(5), 0, 0);

  std::vector<domains_operand_t> three_operands;
  three_operands.reserve(3);
  three_operands.push_back(
      tf::test::make_tagged_operand(outer, placement_of(frame0())));
  three_operands.push_back(
      tf::test::make_tagged_operand(inner, placement_of(frame0())));
  three_operands.push_back(
      tf::test::make_tagged_operand(far_sphere, placement_of(frame0())));
  auto three = tf::test::tagged_forms(three_operands);
  auto g3 = tf::test::build_range_csg_graph(tf::test::forms_range(three),
                                            tf::test::no_sheets(), {});
  auto [cells3, ids3] = tf::test::csg_domains_of(g3);
  auto v3 = sorted_volumes(cells3);
  REQUIRE(cells3.size() == 3);

  auto oi = tf::concatenated(outer.polygons(), inner.polygons());
  std::vector<domains_operand_t> two_operands;
  two_operands.reserve(2);
  two_operands.push_back(
      tf::test::make_tagged_operand(oi, placement_of(frame0())));
  two_operands.push_back(
      tf::test::make_tagged_operand(far_sphere, placement_of(frame0())));
  auto two = tf::test::tagged_forms(two_operands);
  auto g2 = tf::test::build_range_csg_graph(
      tf::test::forms_range(two), tf::test::no_sheets(), within_config);
  auto [cells2, ids2] = tf::test::csg_domains_of(g2);

  require_same_volumes(v3, sorted_volumes(cells2));
}

TEST_CASE("one-form graph: nested spheres in a single soup keep cavity and "
          "outer shell",
          "[domains][nesting][within]") {
  // Everything concatenated into ONE operand: two hollow spheres (outer
  // + inner, and a translated copy). Nesting resolves purely by
  // containment casts -- there are no intersection records at all --
  // and the universe must be exactly the two outer spheres reversed.
  auto outer_a = sphere_at(0, 0, 0);
  auto inner_a = sphere_at(0, 0, 0, domains_real_t(0.5));
  auto outer_b = sphere_at(domains_real_t(3), 0, 0);
  auto inner_b = sphere_at(domains_real_t(3), 0, 0, domains_real_t(0.5));
  const double vo = double(tf::signed_volume(outer_a.polygons()));
  const double vi = double(tf::signed_volume(inner_a.polygons()));

  auto ab = tf::concatenated(outer_a.polygons(), inner_a.polygons());
  auto abc = tf::concatenated(ab.polygons(), outer_b.polygons());
  auto soup = tf::concatenated(abc.polygons(), inner_b.polygons());
  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});

  auto [cells, ids] = tf::test::csg_domains_of(graph);
  std::vector<double> want{vi, vi, vo - vi, vo - vi};
  require_same_volumes(want, sorted_volumes(cells));

  // raw keeps the universe: ONE domain bounded by both outer spheres
  // reversed, SIGNED volume exactly -2 * vo (sorted_volumes above takes
  // abs, so check the universe sign directly)
  auto [raw, raw_ids] =
      tf::test::csg_domains_of(graph, tf::domain_config::none);
  REQUIRE(raw.size() == 5);
  double universe = 0;
  for (auto &c : raw)
    universe = std::min(universe, double(tf::signed_volume(c.polygons())));
  REQUIRE_THAT(universe, Catch::Matchers::WithinRel(-2 * vo, 1e-9));
  std::vector<double> want_raw{vi, vi, vo - vi, vo - vi, 2 * vo};
  require_same_volumes(want_raw, sorted_volumes(raw));
}

namespace {

auto cube_hole_operands(bool inside)
    -> std::pair<tf::polygons_buffer<domains_index_t, domains_real_t, 3, 3>,
                 tf::polygons_buffer<domains_index_t, domains_real_t, 3, 3>> {
  auto big = tf::triangulated(
      tf::make_box_mesh<domains_index_t, domains_real_t>(
          domains_real_t(2), domains_real_t(2), domains_real_t(2))
          .polygons());
  tf::ensure_positive_orientation(big.polygons());
  auto small = tf::triangulated(
      tf::make_box_mesh<domains_index_t, domains_real_t>(
          domains_real_t(0.4), domains_real_t(0.4), domains_real_t(0.4))
          .polygons());
  tf::ensure_positive_orientation(small.polygons());
  // footprint centre (0.4, -0.4): strictly inside ONE top triangle
  const domains_real_t z_off =
      inside ? domains_real_t(0.8) : domains_real_t(1.2);
  for (auto &&p : small.points()) {
    p[0] += domains_real_t(0.4);
    p[1] += domains_real_t(-0.4);
    p[2] += z_off;
  }
  return {std::move(big), std::move(small)};
}

auto faces_at_z(
    const tf::polygons_buffer<domains_index_t, domains_real_t, 3, 3> &m,
    domains_real_t z) -> std::set<domains_index_t> {
  std::set<domains_index_t> out;
  auto faces = m.faces();
  for (domains_index_t f = 0; f < domains_index_t(faces.size()); ++f) {
    bool all = true;
    for (int c = 0; c < 3; ++c)
      all = all && m.points()[std::size_t(faces[f][std::size_t(c)])][2] == z;
    if (all)
      out.insert(f);
  }
  return out;
}

} // namespace

TEST_CASE("cube-on-cube hole: domains, volumes, and face membership "
          "across the route matrix",
          "[csg][domains][hole]") {
  // A small cube whose footprint lies strictly inside ONE triangle of
  // the big cube's top face — the hole path. Two contacts: sitting on
  // top (opposing coplanar) and hanging inside (aligned coplanar).
  const double v_small_ref = 0.4 * 0.4 * 0.4;
  for (int cfgi = 0; cfgi < 2; ++cfgi) {
    const bool inside = cfgi == 1;
    const double v_big_ref = inside ? 8.0 - v_small_ref : 8.0;
    auto operands = cube_hole_operands(inside);
    auto &big = operands.first;
    auto &small = operands.second;
    const domains_real_t z_contact = domains_real_t(1);
    const auto small_contact =
        faces_at_z(small, z_contact); // bottom (top cfg) or top (inside)
    const auto big_top = faces_at_z(big, domains_real_t(1));
    const domains_index_t n_big = domains_index_t(big.faces().size());

    for (auto tri : {tf::triangulation_type::cdt,
                     tf::triangulation_type::refined_cdt}) {
      const tf::arrangement_config cfg{
          tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_crossing_contours},
          tri};
      for (int nary = 0; nary < 2; ++nary) {
        DYNAMIC_SECTION((inside ? "inside" : "top")
                        << (nary ? ", csg n-ary" : ", csg concat")
                        << ", tri=" << int(tri)) {
          auto run = [&](auto &&graph) {
            auto [cells, ids, imap_b] =
                tf::make_csg_domains(graph, tf::return_index_map);
            // structured bindings cannot be captured pre-C++20 (MSVC)
            auto &imap = imap_b;
            REQUIRE(cells.size() == 2);
            std::size_t bi = 0, si = 1;
            double v0 = std::fabs(double(tf::signed_volume(cells[0].polygons())));
            double v1 = std::fabs(double(tf::signed_volume(cells[1].polygons())));
            if (v0 < v1)
              std::swap(bi, si);
            const double vb = std::max(v0, v1), vs = std::min(v0, v1);
            REQUIRE_THAT(vb, Catch::Matchers::WithinRel(v_big_ref, 1e-6));
            REQUIRE_THAT(vs, Catch::Matchers::WithinRel(v_small_ref, 1e-6));
            for (auto &c : cells) {
              REQUIRE(tf::is_closed(c.polygons()));
              REQUIRE(tf::is_manifold(c.polygons()));
            }
            // face membership: which input faces bound which domain
            auto classify = [&](std::size_t k) {
              std::set<domains_index_t> bigf, smallf;
              auto ftag = imap.face_tag_blocks[k];
              auto fid = imap.face_blocks[k];
              for (std::size_t j = 0; j < std::size_t(ftag.size()); ++j) {
                const bool is_big =
                    nary ? ftag[j] == domains_index_t(0) : fid[j] < n_big;
                if (is_big)
                  bigf.insert(nary ? fid[j] : fid[j]);
                else
                  smallf.insert(nary ? fid[j] : fid[j] - n_big);
              }
              return std::make_pair(bigf, smallf);
            };
            auto cb = classify(bi);
            auto cs = classify(si);
            auto &big_of_big = cb.first;
            auto &small_of_big = cb.second;
            auto &big_of_small = cs.first;
            auto &small_of_small = cs.second;
            // the big domain is bounded by every big-cube face...
            REQUIRE(big_of_big.size() == std::size_t(n_big));
            // ...and touches the small cube only through walls at the
            // contact plane (top cfg) or the cubby walls (inside cfg)
            if (!inside)
              for (auto f : small_of_big)
                REQUIRE(small_contact.count(f) == 1);
            // the small domain: every non-contact small face bounds it
            for (domains_index_t f = 0;
                 f < domains_index_t(small.faces().size()); ++f)
              if (!small_contact.count(f))
                REQUIRE(small_of_small.count(f) == 1);
            // any big-cube face on the small domain lies in the top face
            for (auto f : big_of_small)
              REQUIRE(big_top.count(f) == 1);
          };
          if (nary) {
            std::vector<domains_operand_t> ops;
            ops.reserve(2);
            ops.push_back(tf::test::make_tagged_operand(big));
            ops.push_back(tf::test::make_tagged_operand(small));
            auto forms = tf::test::tagged_forms(ops);
            run(tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                                tf::test::no_sheets(), cfg));
          } else {
            auto soup = tf::concatenated(big.polygons(), small.polygons());
            auto soup_operand = tf::test::make_tagged_operand(soup);
            run(tf::test::build_self_csg_graph(soup_operand.form(), cfg));
          }
        }
      }
      // arrangement routes: domain volumes through the free path
      DYNAMIC_SECTION((inside ? "inside" : "top")
                      << ", arrangements, tri=" << int(tri)) {
        std::vector<decltype(big.polygons())> forms{big.polygons(),
                                                    small.polygons()};
        auto [m, t, f] = tf::make_mesh_arrangements(
            tf::make_range(forms.data(), forms.data() + forms.size()), cfg);
        auto bb = tf::aabb_from(m.polygons().points());
        auto cl = tf::cleaned(m.polygons(), bb.diagonal().length() * 1e-9);
        auto labels = tf::make_domain_labels(
            cl.polygons(), tf::domain_config::ignore_open_fragments |
                               tf::domain_config::exclude_outer_shell);
        auto [comps, clabels] = tf::split_into_domains(cl.polygons(), labels);
        REQUIRE(comps.size() == 2);
        double v0 = std::fabs(double(tf::signed_volume(comps[0].polygons())));
        double v1 = std::fabs(double(tf::signed_volume(comps[1].polygons())));
        REQUIRE_THAT(std::max(v0, v1), Catch::Matchers::WithinRel(v_big_ref, 1e-6));
        REQUIRE_THAT(std::min(v0, v1), Catch::Matchers::WithinRel(v_small_ref, 1e-6));
        for (auto &c : comps) {
          REQUIRE(tf::is_closed(c.polygons()));
          REQUIRE(tf::is_manifold(c.polygons()));
        }
      }
    }
  }
}
