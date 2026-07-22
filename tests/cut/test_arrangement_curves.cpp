/*
 * Copyright (c) 2026 XLAB
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <trueform/trueform.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace {

using Index = int;
using Real = double;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

template <typename Mesh>
auto require_no_stray_vertices(const Mesh &mesh) -> void {
  std::vector<char> used(std::size_t(mesh.points_buffer().size()), 0);
  for (auto f : mesh.faces())
    for (std::size_t k = 0; k < f.size(); ++k)
      used[std::size_t(f[k])] = 1;
  std::size_t stray = 0;
  for (auto u : used)
    stray += !u;
  REQUIRE(stray == 0);
}

auto require_no_stray_points(const tf::curves_buffer<Index, Real, 3> &cb)
    -> void {
  std::vector<char> used(std::size_t(cb.points().size()), 0);
  for (auto &&path_ids : cb.paths_buffer())
    for (auto id : path_ids)
      used[std::size_t(id)] = 1;
  for (auto u : used)
    REQUIRE(u == 1);
}

auto curve_length(const tf::curves_buffer<Index, Real, 3> &cb) -> Real {
  Real total = 0;
  for (auto &&path : cb.curves())
    for (std::size_t i = 0; i + 1 < path.size(); ++i)
      total += std::sqrt(Real(tf::distance2(path[i], path[i + 1])));
  return total;
}

} // namespace

TEST_CASE("arrangement curves: transversal crossing, graph and csg agree",
          "[cut][curves]") {
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto plane = tf::make_plane_mesh(4.0, 4.0, 2, 2);
  std::vector<decltype(box.polygons())> forms{box.polygons(),
                                              plane.polygons()};
  auto rng = tf::make_range(forms.data(), forms.data() + forms.size());

  auto graph = tf::make_arrangement_graph(rng);
  auto cb = tf::make_intersection_curves(graph);
  // the plane crosses the box at z=0: the curve is the 2x2 square
  REQUIRE(cb.curves().size() > 0);
  REQUIRE(curve_length(cb) == Catch::Approx(8.0).epsilon(1e-9));
  // curves carry only referenced points — a retired or unreferenced
  // created id never becomes a stray output vertex
  require_no_stray_points(cb);

  auto cb_sos = tf::make_intersection_curves(box.polygons(), plane.polygons());
  REQUIRE(cb_sos.curves().size() > 0);
  require_no_stray_points(cb_sos);

  auto csg = tf::make_csg_graph(rng);
  auto cb2 = tf::make_intersection_curves(csg);
  REQUIRE(curve_length(cb2) == Catch::Approx(curve_length(cb)).epsilon(1e-12));

  // the standalone read (no triangulation, no graph) rides the same
  // scan and must agree
  auto cb3 = tf::make_intersection_curves(
      box.polygons(), plane.polygons(),
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours});
  REQUIRE(curve_length(cb3) == Catch::Approx(curve_length(cb)).epsilon(1e-12));

  auto cb4 = tf::make_self_intersection_curves(
      tf::concatenated(box.polygons(), plane.polygons()).polygons(),
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours});
  REQUIRE(curve_length(cb4) == Catch::Approx(curve_length(cb)).epsilon(1e-9));

  // a weld-retired point must not be copied into the output mesh:
  // every output point is referenced by at least one face
  auto [mesh, tags, faces] =
      tf::make_mesh_arrangements(box.polygons(), plane.polygons());
  std::vector<char> used(std::size_t(mesh.points_buffer().size()), 0);
  for (auto f : mesh.faces())
    for (int k = 0; k < 3; ++k)
      used[std::size_t(f[k])] = 1;
  for (auto u : used)
    REQUIRE(u == 1);
}

TEST_CASE("arrangement curves: coincident overlay contributes its border",
          "[cut][curves][coplanar]") {
  // a small tiled plane lying IN a bigger tiled plane: the curve is
  // the small plane's rim — the overlap interior stays silent even
  // though every interior edge is doubled before the collapse
  auto big = tf::make_plane_mesh(8.0, 8.0, 4, 4);
  auto small = tf::make_plane_mesh(2.0, 2.0, 2, 2);
  std::vector<decltype(big.polygons())> forms{big.polygons(),
                                              small.polygons()};
  auto graph = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()));
  REQUIRE(graph.dead_loops().size() > 0);
  auto cb = tf::make_intersection_curves(graph);
  REQUIRE(cb.curves().size() > 0);
  // rim of the 2x2 square, nothing more
  REQUIRE(curve_length(cb) == Catch::Approx(8.0).epsilon(1e-9));
  // every curve point sits ON the rim (max-norm distance 1 from center
  // in the plane, z = 0)
  for (auto &&path : cb.curves())
    for (auto &&pt : path) {
      REQUIRE(pt[2] == Catch::Approx(0.0).margin(1e-12));
      const Real m = std::max(std::abs(pt[0]), std::abs(pt[1]));
      REQUIRE(m == Catch::Approx(1.0).epsilon(1e-9));
    }
}

TEST_CASE("arrangement curves: stacked solids yield the shared-wall rim",
          "[cut][curves][coplanar]") {
  auto lo = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto hi = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  for (auto &&p : hi.points())
    p[2] += 2.0;
  std::vector<decltype(lo.polygons())> forms{lo.polygons(), hi.polygons()};
  auto graph = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()));
  auto cb = tf::make_intersection_curves(graph);
  REQUIRE(cb.curves().size() > 0);
  // contact = the shared wall at z=1: its rim is the 2x2 square
  REQUIRE(curve_length(cb) == Catch::Approx(8.0).epsilon(1e-9));
  for (auto &&path : cb.curves())
    for (auto &&pt : path)
      REQUIRE(pt[2] == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("arrangement curves: coincident overlay WITHIN one mesh is "
          "border-only",
          "[cut][curves][coplanar][self]") {
  // one mesh containing a doubled sub-patch: the single-tag rule must
  // report exactly the patch border — stack interiors stay silent
  auto big = tf::make_plane_mesh(8.0, 8.0, 4, 4);
  auto small = tf::make_plane_mesh(2.0, 2.0, 2, 2);
  auto soup = tf::concatenated(big.polygons(), small.polygons());
  auto graph = tf::make_arrangement_graph(soup.polygons());
  REQUIRE(graph.dead_loops().size() > 0);
  auto cb = tf::make_intersection_curves(graph);
  REQUIRE(cb.curves().size() > 0);
  REQUIRE(curve_length(cb) == Catch::Approx(8.0).epsilon(1e-9));
  for (auto &&path : cb.curves())
    for (auto &&pt : path) {
      REQUIRE(pt[2] == Catch::Approx(0.0).margin(1e-12));
      const Real m = std::max(std::abs(pt[0]), std::abs(pt[1]));
      REQUIRE(m == Catch::Approx(1.0).epsilon(1e-9));
    }
}

TEST_CASE("arrangement curves: self arrangement uses the non-manifold rule",
          "[cut][curves][self]") {
  // one mesh containing two crossing sheets: the seam is where 4 walks
  // meet — invisible to the cross-tag rule, found by the non-manifold
  // one
  auto h = tf::make_plane_mesh(2.0, 2.0, 1, 1);
  auto v = tf::make_plane_mesh(2.0, 2.0, 1, 1);
  for (auto &&p : v.points()) {
    const Real y = p[1];
    p[1] = p[2];
    p[2] = y;
  }
  auto soup = tf::concatenated(h.polygons(), v.polygons());
  auto graph = tf::make_arrangement_graph(soup.polygons());
  auto cb = tf::make_intersection_curves(graph);
  REQUIRE(cb.curves().size() > 0);
  // the crossing segment: the x axis, length 2
  REQUIRE(curve_length(cb) == Catch::Approx(2.0).epsilon(1e-9));
  for (auto &&path : cb.curves())
    for (auto &&pt : path) {
      REQUIRE(pt[1] == Catch::Approx(0.0).margin(1e-12));
      REQUIRE(pt[2] == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("arrangement curves: byte-deterministic across runs",
          "[cut][curves][determinism]") {
  // parallel emission must not leak thread timing into the output:
  // two builds of the same input compare equal point-for-point
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto sheet = tf::make_plane_mesh(2.0, 2.0, 2, 2);
  for (auto &&p : sheet.points()) {
    const Real y = p[1];
    p[1] = 0.3;
    p[2] = y;
  }
  auto run = [&] {
    std::vector<decltype(box.polygons())> forms{box.polygons(),
                                                sheet.polygons()};
    auto graph = tf::make_csg_graph(
        tf::make_range(forms.data(), forms.data() + forms.size()));
    return tf::make_intersection_curves(graph);
  };
  auto a = run();
  for (int r = 0; r < 3; ++r) {
    auto b = run();
    REQUIRE(a.points().size() == b.points().size());
    for (std::size_t i = 0; i < std::size_t(a.points().size()); ++i)
      for (int c = 0; c < 3; ++c)
        REQUIRE(a.points()[i][c] == b.points()[i][c]);
    REQUIRE(a.paths_buffer().size() == b.paths_buffer().size());
    auto pa = a.paths_buffer();
    auto pb = b.paths_buffer();
    for (std::size_t k = 0; k < std::size_t(pa.size()); ++k) {
      REQUIRE(pa[k].size() == pb[k].size());
      for (std::size_t j = 0; j < std::size_t(pa[k].size()); ++j)
        REQUIRE(pa[k][j] == pb[k][j]);
    }
  }
}

TEST_CASE("arrangement curves: synthetic DFN oracle across the full route "
          "matrix",
          "[cut][curves][dfn]") {
  // The DFN shape, synthetic: a box and a vertical sheet whose rim
  // lies exactly IN the box walls (T-contact slit all around). The
  // oracle (Žiga): the curves must equal the non-manifold edges of
  // the returned arrangement — checked on every route and both
  // triangulations.
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto sheet_src = tf::make_plane_mesh(2.0, 2.0, 2, 2);
  auto sheet = sheet_src;
  for (auto &&p : sheet.points()) {
    const Real y = p[1];
    p[1] = 0.3;
    p[2] = y; // vertical, rim on x=+-1 and z=+-1 walls
  }

  auto nm_length = [](const auto &mesh) -> Real {
    auto bb = tf::aabb_from(mesh.polygons().points());
    auto cl = tf::cleaned(mesh.polygons(), bb.diagonal().length() * 1e-9);
    auto nm = tf::make_non_manifold_edges(cl.polygons());
    Real total = 0;
    for (auto &&e : tf::make_edges(nm)) {
      auto p0 = cl.polygons().points()[std::size_t(e[0])];
      auto p1 = cl.polygons().points()[std::size_t(e[1])];
      total += std::sqrt(Real(tf::distance2(p0, p1)));
    }
    return total;
  };

  const auto tris = {tf::triangulation_type::cdt,
                     tf::triangulation_type::refined_cdt};
  for (auto tri : tris) {
    const tf::arrangement_config cfg{
        tf::intersect_config{tf::intersect_mode::primitives |
                             tf::intersect_mode::resolve_crossing_contours},
        tri};
    DYNAMIC_SECTION("pair, tri=" << int(tri)) {
      auto [m, tags, faces, cb] = tf::make_mesh_arrangements(
          box.polygons(), sheet.polygons(), cfg, tf::return_curves);
      REQUIRE(cb.curves().size() > 0);
      REQUIRE(curve_length(cb) == Catch::Approx(nm_length(m)).epsilon(1e-6));
      require_no_stray_points(cb);
      require_no_stray_vertices(m);
    }
    DYNAMIC_SECTION("n-ary, tri=" << int(tri)) {
      std::vector<decltype(box.polygons())> forms{box.polygons(),
                                                  sheet.polygons()};
      auto [m, tags, faces, cb] = tf::make_mesh_arrangements(
          tf::make_range(forms.data(), forms.data() + forms.size()), cfg,
          tf::return_curves);
      REQUIRE(cb.curves().size() > 0);
      REQUIRE(curve_length(cb) == Catch::Approx(nm_length(m)).epsilon(1e-6));
      require_no_stray_points(cb);
      require_no_stray_vertices(m);
    }
    DYNAMIC_SECTION("polygon soup, tri=" << int(tri)) {
      auto soup = tf::concatenated(box.polygons(), sheet.polygons());
      auto [m, faces, cb] = tf::make_polygon_arrangements(
          soup.polygons(),
          {tf::intersect_config{tf::intersect_mode::primitives |
                                tf::intersect_mode::resolve_contours |
                                tf::intersect_mode::within},
           tri},
          tf::return_curves);
      REQUIRE(cb.curves().size() > 0);
      REQUIRE(curve_length(cb) == Catch::Approx(nm_length(m)).epsilon(1e-6));
      require_no_stray_points(cb);
      require_no_stray_vertices(m);
    }
    DYNAMIC_SECTION("csg n-ary, tri=" << int(tri)) {
      std::vector<decltype(box.polygons())> forms{box.polygons(),
                                                  sheet.polygons()};
      auto graph = tf::make_csg_graph(
          tf::make_range(forms.data(), forms.data() + forms.size()), cfg);
      auto m = tf::make_csg_mesh(graph);
      auto cb = tf::make_intersection_curves(graph);
      REQUIRE(cb.curves().size() > 0);
      REQUIRE(curve_length(cb) == Catch::Approx(nm_length(m)).epsilon(1e-6));
      require_no_stray_points(cb);
      require_no_stray_vertices(m);
    }
    DYNAMIC_SECTION("csg concatenated, tri=" << int(tri)) {
      auto soup = tf::concatenated(box.polygons(), sheet.polygons());
      auto graph = tf::make_csg_graph(soup.polygons(), cfg);
      auto m = tf::make_csg_mesh(graph);
      auto cb = tf::make_intersection_curves(graph);
      REQUIRE(cb.curves().size() > 0);
      REQUIRE(curve_length(cb) == Catch::Approx(nm_length(m)).epsilon(1e-6));
      require_no_stray_points(cb);
      require_no_stray_vertices(m);
    }
  }
}
