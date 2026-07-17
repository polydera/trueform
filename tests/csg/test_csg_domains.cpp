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

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using Index = int;
using Real = double;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

auto frame0() {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
}

auto frame_at(Real x, Real y, Real z) {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

using frame_t = decltype(frame0());
using form_t = decltype(std::declval<mesh_t &>().polygons() |
                        tf::tag(std::declval<frame_t>()));

template <typename Cells> auto sorted_volumes(const Cells &cells) {
  std::vector<Real> v;
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
    REQUIRE_THAT(v1[i], Catch::Matchers::WithinRel(v2[i], Real(0.02)));

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
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 24, 24);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph =
      tf::make_csg_graph(tf::make_range(forms), tf::make_range(std::array<int, 1>{1}));

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
  REQUIRE(part.n_kept == static_cast<Index>(n_keep));

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
      REQUIRE(part.dense_of_domain[d] == Index(-1));
  }
}

TEST_CASE("make_csg_domains splits a plane-cut sphere into two cells",
          "[domains]") {
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto [cells, ids] = tf::make_csg_domains(graph);

  REQUIRE(cells.size() == 2);
  REQUIRE(ids.size() == 2);

  Real total_volume = 0;
  for (auto &cell : cells) {
    auto polys = cell.polygons();
    REQUIRE(tf::is_closed(polys));
    REQUIRE(tf::is_manifold(polys));
    total_volume += std::abs(tf::signed_volume(polys));
  }

  const Real ball = Real(4) / Real(3) * tf::pi<Real>;
  REQUIRE(std::abs(total_volume - ball) < Real(0.02) * ball);
}

TEST_CASE("make_csg_domains default config fuses sheet open halves",
          "[domains]") {
  // Headline change: under the default config (exclude_outer_shell |
  // ignore_open_fragments) the sheet's two open halves are merged into
  // the universe and dropped, leaving exactly the two closed hemispheres.
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms),
                                  tf::make_range(std::array<int, 1>{1}));

  auto [cells, ids] = tf::make_csg_domains(graph);

  REQUIRE(cells.size() == 2);
  REQUIRE(ids.size() == 2);

  int n_closed = 0;
  int n_open = 0;
  Real total_volume = 0;
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

  const Real ball = Real(4) / Real(3) * tf::pi<Real>;
  REQUIRE(std::abs(total_volume - ball) < Real(0.02) * ball);
}

TEST_CASE("make_csg_domains filters domains without merging them",
          "[domains]") {
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto [cells, ids] = tf::make_csg_domains(graph, tf::csg::op(0));

  REQUIRE(cells.size() == 2);
  for (auto &cell : cells)
    REQUIRE(tf::is_closed(cell.polygons()));
}

TEST_CASE("make_csg_domains filter still selects hemispheres for a sheet",
          "[domains]") {
  // Sheet graph, default config + op(0): open halves fused away, only the
  // two closed sphere-interior hemispheres survive the filter.
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms),
                                  tf::make_range(std::array<int, 1>{1}));

  auto [cells, ids] = tf::make_csg_domains(graph, tf::csg::op(0));

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
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms),
                                  tf::make_range(std::array<int, 1>{1}));

  auto [cells, ids] = tf::make_csg_domains(
      graph, ~tf::csg::op(0) & ~tf::csg::op(1),
      tf::domain_config::ignore_open_fragments);

  REQUIRE(cells.size() >= 1);
  bool found_outer = false;
  for (auto &cell : cells)
    if (tf::signed_volume(cell.polygons()) < Real(0))
      found_outer = true;
  REQUIRE(found_outer);
}

TEST_CASE("make_csg_domains raw config emits every arrangement domain",
          "[domains]") {
  // config = none: no open merge, no universe drop. Observed for the sheet
  // graph: 4 domains (2 closed hemispheres + 2 open sheet halves).
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms),
                                  tf::make_range(std::array<int, 1>{1}));

  auto [cells, ids] = tf::make_csg_domains(graph, tf::domain_config::none);

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
  auto af = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto bf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto a = af.polygons() | tf::tag(frame0());
  auto b = bf.polygons() | tf::tag(frame_at(Real(1), 0, 0));
  std::vector<form_t> forms{a, b};

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);

  auto [arr_mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(a, b);
  auto clean = tf::cleaned(arr_mesh.polygons(), Real(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  print_and_check_parity("two overlapping spheres", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("make_csg_domains matches materialized split: sphere + plane sheet",
          "[domains][parity]") {
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(3), Real(3));
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{sphere, plane};

  auto graph = tf::make_csg_graph(tf::make_range(forms),
                                  tf::make_range(std::array<int, 1>{1}));
  auto [cells, ids] = tf::make_csg_domains(graph);

  auto [arr_mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(sphere, plane);
  auto clean = tf::cleaned(arr_mesh.polygons(), Real(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  auto [ocells, oids] = tf::split_into_domains(clean.polygons(), dl);

  print_and_check_parity("sphere + plane sheet", cells, ocells);
  REQUIRE(cells.size() == 2);
}

TEST_CASE("make_csg_domains matches materialized split: nested spheres",
          "[domains][parity]") {
  auto of = tf::make_sphere_mesh<Index>(Real(2), 32, 32);
  auto inf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto outer = of.polygons() | tf::tag(frame0());
  auto inner = inf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{outer, inner};

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);

  auto [arr_mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(outer, inner);
  auto clean = tf::cleaned(arr_mesh.polygons(), Real(1e-6));
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
  auto clean = tf::cleaned(arr_mesh.polygons(), Real(1e-6));
  auto dl = tf::make_domain_labels(clean.polygons(),
                                   tf::domain_config::exclude_outer_shell |
                                       tf::domain_config::ignore_open_fragments);
  return tf::split_into_domains(clean.polygons(), dl);
}

auto sphere_form(Real r, frame_t frame, std::vector<mesh_t> &storage) {
  storage.push_back(tf::make_sphere_mesh<Index>(r, 32, 32));
  return storage.back().polygons() | tf::tag(frame);
}

} // namespace

TEST_CASE("nesting: 3-level concentric spheres", "[domains][nesting]") {
  std::vector<mesh_t> storage;
  storage.reserve(3);
  std::vector<form_t> forms;
  forms.push_back(sphere_form(Real(3), frame0(), storage));
  forms.push_back(sphere_form(Real(2), frame0(), storage));
  forms.push_back(sphere_form(Real(1), frame0(), storage));

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("3-level concentric", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("nesting: off-center fully nested sphere", "[domains][nesting]") {
  std::vector<mesh_t> storage;
  storage.reserve(2);
  std::vector<form_t> forms;
  forms.push_back(sphere_form(Real(3), frame0(), storage));
  forms.push_back(sphere_form(Real(1), frame_at(Real(1.2), 0, 0), storage));

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("off-center nested", cells, ocells);
  REQUIRE(cells.size() == 2);
}

TEST_CASE("nesting: two sibling spheres in one parent", "[domains][nesting]") {
  std::vector<mesh_t> storage;
  storage.reserve(3);
  std::vector<form_t> forms;
  forms.push_back(sphere_form(Real(3), frame0(), storage));
  forms.push_back(sphere_form(Real(0.6), frame_at(Real(1.5), 0, 0), storage));
  forms.push_back(sphere_form(Real(0.6), frame_at(Real(-1.5), 0, 0), storage));

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("two siblings in parent", cells, ocells);
  REQUIRE(cells.size() == 3);
}

TEST_CASE("nesting: nested plus intersecting mix", "[domains][nesting]") {
  std::vector<mesh_t> storage;
  storage.reserve(3);
  std::vector<form_t> forms;
  forms.push_back(sphere_form(Real(3), frame0(), storage));
  forms.push_back(sphere_form(Real(1), frame0(), storage));
  forms.push_back(sphere_form(Real(2.5), frame_at(Real(2.5), 0, 0), storage));

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("nested + intersecting", cells, ocells);
}

TEST_CASE("nesting: deep 4-level concentric chain", "[domains][nesting]") {
  std::vector<mesh_t> storage;
  storage.reserve(4);
  std::vector<form_t> forms;
  forms.push_back(sphere_form(Real(4), frame0(), storage));
  forms.push_back(sphere_form(Real(3), frame0(), storage));
  forms.push_back(sphere_form(Real(2), frame0(), storage));
  forms.push_back(sphere_form(Real(1), frame0(), storage));

  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [cells, ids] = tf::make_csg_domains(graph);
  auto [ocells, oids] = oracle_domains(tf::make_range(forms));

  print_and_check_parity("4-level concentric", cells, ocells);
  REQUIRE(cells.size() == 4);
}

// Regression: an open cut whose free edge ends inside a closed volume makes a
// slit - the cut region runs in along one wall and back along the other,
// folding both walls into one edge incidence. Before the slit-aware
// non-manifold-edge fan builder (tf::cut::make_non_manifold_edge_fans), that
// lost incidence collapsed the sphere's interior and exterior into a single
// domain: the union came out empty / open. This pins the fixed behaviour.
TEST_CASE("make_csg_domains: open partial wall (slit) does not collapse",
          "[domains][slit]") {
  auto sf = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto pf = tf::make_plane_mesh<Index>(Real(2), Real(4)); // partial wall
  auto sphere = sf.polygons() | tf::tag(frame0());
  auto plane = pf.polygons() | tf::tag(frame_at(-1, 0, 0)); // free edge inside
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  // The union solid must stay the watertight sphere, not collapse to empty.
  auto u =
      tf::make_csg_mesh(graph, tf::csg::any_of(tf::make_sequence_range(0, 2)));
  REQUIRE(tf::is_closed(u.polygons()));
  REQUIRE(tf::is_manifold(u.polygons()));
  const Real ball = Real(4) / Real(3) * tf::pi<Real>;
  const Real vol = std::abs(tf::signed_volume(u.polygons()));
  REQUIRE(std::abs(vol - ball) < Real(0.03) * ball);

  // The wall divides the interior: keeping open fragments yields >= 2 cells
  // (it was a single collapsed cell before the fix).
  auto [cells, ids] =
      tf::make_csg_domains(graph, tf::domain_config::ignore_open_fragments);
  REQUIRE(cells.size() >= 2);
}

// make_intersection_curves walks region-loop edges and emits the cross-tag
// seam network; two overlapping spheres meet along exactly one closed circle.
TEST_CASE("make_intersection_curves: two spheres meet on one closed loop",
          "[intersection_curves]") {
  auto af = tf::make_sphere_mesh<Index>(Real(1), 24, 24);
  auto bf = tf::make_sphere_mesh<Index>(Real(1), 24, 24);
  auto a = af.polygons() | tf::tag(frame0());
  auto b = bf.polygons() | tf::tag(frame_at(Real(0.8), 0, 0));
  std::vector<form_t> forms{a, b};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto curves = tf::make_intersection_curves(graph);
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
  auto bf = tf::make_box_mesh<Index>(Real(4), Real(4), Real(4));
  auto cf = tf::make_cylinder_mesh<Index>(Real(0.5), Real(6), 32);
  auto box = bf.polygons() | tf::tag(frame0());
  auto cyl = cf.polygons() | tf::tag(frame_at(1, -1, 0)); // offset into one face
  std::vector<form_t> forms{box, cyl};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto drilled = tf::make_csg_mesh(graph, tf::csg::op(0) & ~tf::csg::op(1));
  REQUIRE(tf::is_closed(drilled.polygons()));
  REQUIRE(tf::is_manifold(drilled.polygons()));
  // box (4^3 = 64) minus a cylinder bored straight through (pi r^2 * 4).
  const Real expect = Real(64) - tf::pi<Real> * Real(0.25) * Real(4);
  const Real vol = std::abs(tf::signed_volume(drilled.polygons()));
  REQUIRE(std::abs(vol - expect) < Real(0.1)); // 32-segment facet tolerance
}

TEST_CASE("make_csg_domains return_source_ids: per-cell tag + face provenance",
          "[domains][source_ids]") {
  // Two overlapping cubes (side 2, offset by 1,1,1) -> three solid cells.
  auto af = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto bf = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  for (std::size_t i = 0; i < bf.points_buffer().size(); ++i) {
    auto p = bf.points_buffer()[i];
    bf.points_buffer()[i] =
        tf::point<Real, 3>{p[0] + Real(1), p[1] + Real(1), p[2] + Real(1)};
  }
  auto a = af.polygons() | tf::tag(frame0());
  auto b = bf.polygons() | tf::tag(frame0());
  std::vector<form_t> forms{a, b};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto [cells0, ids0] = tf::make_csg_domains(graph);
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
      REQUIRE((t == Index(0) || t == Index(1)));
      saw0 = saw0 || (t == Index(0));
      saw1 = saw1 || (t == Index(1));
      REQUIRE(fl >= Index(0));
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
  using qmesh_t = tf::polygons_buffer<Index, Real, 3, 4>;
  auto quad_cube = [](Real s, Real ox, Real oy, Real oz) {
    qmesh_t m;
    m.points_buffer().allocate(8);
    Index idx = 0;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          m.points_buffer()[std::size_t(idx++)] = tf::point<Real, 3>{
              ox + Real(x) * s, oy + Real(y) * s, oz + Real(z) * s};
    // Outward-oriented quad faces (vertex bits: x=1, y=2, z=4).
    const std::array<std::array<Index, 4>, 6> faces = {{{0, 2, 3, 1},
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
  auto af = quad_cube(Real(2), 0, 0, 0);
  auto bf = quad_cube(Real(2), Real(1), Real(1), Real(1));
  using qform_t = decltype(af.polygons() | tf::tag(frame0()));
  std::vector<qform_t> forms{af.polygons() | tf::tag(frame0()),
                             bf.polygons() | tf::tag(frame0())};
  auto qgraph = tf::make_csg_graph(tf::make_range(forms));

  // plain call first: the dynamic emit must not touch the (empty)
  // provenance machinery when no labels are requested
  {
    auto [pcells, pids] = tf::make_csg_domains(qgraph);
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
      REQUIRE((t == Index(0) || t == Index(1)));
      REQUIRE(fl >= Index(0));
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
  auto c0 = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto c1 = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto plane = tf::make_plane_mesh<Index>(Real(4), Real(4));

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(-0.5), 0, 0}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0.5), 0, 0}));
  std::vector<form_t> forms{c0.polygons() | tf::tag(f0),
                            c1.polygons() | tf::tag(f1),
                            plane.polygons() | tf::tag(frame0())};

  for (bool with_sheet : {true, false}) {
    DYNAMIC_SECTION((with_sheet ? "knife as sheet" : "knife as volume")) {
      std::array<int, 1> sheets{2};
      auto graph = with_sheet
                       ? tf::make_csg_graph(tf::make_range(forms),
                                            tf::make_range(sheets))
                       : tf::make_csg_graph(tf::make_range(forms));
      auto [cells, ids] = tf::make_csg_domains(graph);
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
  auto af = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto bf = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  for (std::size_t i = 0; i < bf.points_buffer().size(); ++i) {
    auto p = bf.points_buffer()[i];
    bf.points_buffer()[i] =
        tf::point<Real, 3>{p[0] + Real(1), p[1] + Real(1), p[2] + Real(1)};
  }
  std::vector<form_t> forms{af.polygons() | tf::tag(frame0()),
                            bf.polygons() | tf::tag(frame0())};
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto [cells, ids, imap] = tf::make_csg_domains(graph, tf::return_index_map);

  REQUIRE(imap.n_tags == Index(forms.size()));
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
      REQUIRE((ftag[j] >= Index(0) && std::size_t(ftag[j]) < forms.size()));
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
      REQUIRE((ptag[p] >= Index(0) && std::size_t(ptag[p]) < forms.size()));
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
  auto [a_cells, a_ids] = tf::make_csg_domains(graph, tf::csg::op(0));
  auto [b_cells, b_ids] = tf::make_csg_domains(graph, tf::csg::op(1));
  auto has = [](const auto &dom_ids, Index id) {
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

auto sphere_at(Real cx, Real cy, Real cz, Real r = Real(1)) -> mesh_t {
  mesh_t m = tf::make_sphere_mesh<Index>(r, 32, 32);
  tf::ensure_positive_orientation(m.polygons());
  auto &p = m.points_buffer();
  for (std::size_t i = 0; i < p.size(); ++i) {
    auto q = p[i];
    p[i] = tf::point<Real, 3>{q[0] + cx, q[1] + cy, q[2] + cz};
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
  auto B = sphere_at(Real(0.9), 0, 0);
  auto C = sphere_at(Real(0.45), Real(0.8), 0);

  std::vector<form_t> three{A.polygons() | tf::tag(frame0()),
                            B.polygons() | tf::tag(frame0()),
                            C.polygons() | tf::tag(frame0())};
  auto g3 = tf::make_csg_graph(tf::make_range(three));
  auto [cells3, ids3] = tf::make_csg_domains(g3);
  auto v3 = sorted_volumes(cells3);
  REQUIRE(cells3.size() == 7);

  auto bc = tf::concatenated(B.polygons(), C.polygons());
  std::vector<form_t> two{bc.polygons() | tf::tag(frame0()),
                          A.polygons() | tf::tag(frame0())};
  auto g2 = tf::make_csg_graph(tf::make_range(two), within_config);
  auto [cells2, ids2] = tf::make_csg_domains(g2);
  auto v2 = sorted_volumes(cells2);

  require_same_volumes(v3, v2);
}

TEST_CASE("make_csg_domains within: flag on clean operands changes nothing",
          "[domains][within]") {
  // No self-overlap anywhere: the within bit must not change the cells.
  auto A = sphere_at(0, 0, 0);
  auto B = sphere_at(Real(1), 0, 0);

  std::vector<form_t> forms{A.polygons() | tf::tag(frame0()),
                            B.polygons() | tf::tag(frame0())};
  auto g_plain = tf::make_csg_graph(tf::make_range(forms));
  auto [cells_p, ids_p] = tf::make_csg_domains(g_plain);

  auto g_within = tf::make_csg_graph(tf::make_range(forms), within_config);
  auto [cells_w, ids_w] = tf::make_csg_domains(g_within);

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
  auto inner = sphere_at(0, 0, 0, Real(0.5));
  auto far_sphere = sphere_at(Real(5), 0, 0);

  std::vector<form_t> three{outer.polygons() | tf::tag(frame0()),
                            inner.polygons() | tf::tag(frame0()),
                            far_sphere.polygons() | tf::tag(frame0())};
  auto g3 = tf::make_csg_graph(tf::make_range(three));
  auto [cells3, ids3] = tf::make_csg_domains(g3);
  auto v3 = sorted_volumes(cells3);
  REQUIRE(cells3.size() == 3);

  auto oi = tf::concatenated(outer.polygons(), inner.polygons());
  std::vector<form_t> two{oi.polygons() | tf::tag(frame0()),
                          far_sphere.polygons() | tf::tag(frame0())};
  auto g2 = tf::make_csg_graph(tf::make_range(two), within_config);
  auto [cells2, ids2] = tf::make_csg_domains(g2);

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
  auto inner_a = sphere_at(0, 0, 0, Real(0.5));
  auto outer_b = sphere_at(Real(3), 0, 0);
  auto inner_b = sphere_at(Real(3), 0, 0, Real(0.5));
  const double vo = double(tf::signed_volume(outer_a.polygons()));
  const double vi = double(tf::signed_volume(inner_a.polygons()));

  auto ab = tf::concatenated(outer_a.polygons(), inner_a.polygons());
  auto abc = tf::concatenated(ab.polygons(), outer_b.polygons());
  auto soup = tf::concatenated(abc.polygons(), inner_b.polygons());
  auto graph = tf::make_csg_graph(soup.polygons());

  auto [cells, ids] = tf::make_csg_domains(graph);
  std::vector<double> want{vi, vi, vo - vi, vo - vi};
  require_same_volumes(want, sorted_volumes(cells));

  // raw keeps the universe: ONE domain bounded by both outer spheres
  // reversed, SIGNED volume exactly -2 * vo (sorted_volumes above takes
  // abs, so check the universe sign directly)
  auto [raw, raw_ids] =
      tf::make_csg_domains(graph, tf::domain_config::none);
  REQUIRE(raw.size() == 5);
  double universe = 0;
  for (auto &c : raw)
    universe = std::min(universe, double(tf::signed_volume(c.polygons())));
  REQUIRE_THAT(universe, Catch::Matchers::WithinRel(-2 * vo, 1e-9));
  std::vector<double> want_raw{vi, vi, vo - vi, vo - vi, 2 * vo};
  require_same_volumes(want_raw, sorted_volumes(raw));
}
