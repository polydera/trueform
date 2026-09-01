/**
 * @file test_iso_face_cuts.cpp
 * @brief Tests for tf::iso::build_iso_cuts at the region grain.
 *
 * Verifies the per-face cut: a face with no interior chord stays one region,
 * chords split it into the expected bands, one cut value may state several
 * chords on one face, a chord may end on an on-cut vertex, a non-convex
 * face's pocket never survives, and the emitted triangles are wound with
 * their source face.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/iso/cut/build_iso_cuts.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

using Index = int;
using Real = double;
using mesh_t = tf::polygons_buffer<Index, Real, 3, tf::dynamic_size>;

namespace {

struct cut_result {
  std::vector<Index> band_of_region;
  std::vector<Index> face_of_region;
  std::vector<std::vector<std::array<Index, 3>>> triangles;
  std::vector<tf::point<Real, 3>> points;
  std::vector<Index> refused;
};

/// Run the cut and flatten its product into plain rows: the band a region
/// carries is the partition block it lands in.
auto cut(const mesh_t &mesh, const std::vector<Real> &scalars,
         const std::vector<Real> &values) -> cut_result {
  auto [sfi, regions, pids] = tf::iso::build_iso_cuts<Index>(
      mesh.polygons(), tf::make_range(scalars), tf::make_range(values));

  cut_result out;
  const auto n_regions = regions.faces.size();
  out.band_of_region.assign(n_regions, Index(-1));
  for (std::size_t band = 0; band < std::size_t(pids.cut_faces.size()); ++band)
    for (const auto region : pids.cut_faces[band])
      out.band_of_region[std::size_t(region)] = static_cast<Index>(band);

  for (std::size_t r = 0; r < n_regions; ++r) {
    out.face_of_region.push_back(regions.faces[r]);
    std::vector<std::array<Index, 3>> block;
    for (const auto &triangle : regions.triangles[r])
      block.push_back(triangle);
    out.triangles.push_back(block);
  }

  for (std::size_t i = 0; i < std::size_t(mesh.points().size()); ++i) {
    const auto p = mesh.points()[i];
    out.points.push_back({p[0], p[1], p[2]});
  }
  for (std::size_t i = 0; i < std::size_t(sfi.intersection_points().size());
       ++i) {
    const auto p = sfi.intersection_points()[i];
    out.points.push_back({p[0], p[1], p[2]});
  }
  for (const auto &p : regions.minted_points)
    out.points.push_back({p[0], p[1], p[2]});
  for (const auto id : regions.refused)
    out.refused.push_back(id);
  return out;
}

auto cross(const tf::point<Real, 3> &a, const tf::point<Real, 3> &b,
           const tf::point<Real, 3> &c) -> std::array<Real, 3> {
  const std::array<Real, 3> u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const std::array<Real, 3> v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

auto region_area(const cut_result &r, std::size_t region) -> Real {
  Real sum = 0;
  for (const auto &t : r.triangles[region]) {
    const auto n = cross(r.points[std::size_t(t[0])], r.points[std::size_t(t[1])],
                         r.points[std::size_t(t[2])]);
    sum += Real(0.5) * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  }
  return sum;
}

auto total_area(const cut_result &r) -> Real {
  Real sum = 0;
  for (std::size_t i = 0; i < r.triangles.size(); ++i)
    sum += region_area(r, i);
  return sum;
}

/// Every emitted triangle's normal agrees in SIGN with its source face's.
auto winding_agrees(const mesh_t &mesh, const cut_result &r) -> bool {
  for (std::size_t region = 0; region < r.triangles.size(); ++region) {
    const auto face = mesh.polygons()[std::size_t(r.face_of_region[region])];
    const tf::point<Real, 3> f0{face[0][0], face[0][1], face[0][2]};
    const tf::point<Real, 3> f1{face[1][0], face[1][1], face[1][2]};
    const tf::point<Real, 3> f2{face[2][0], face[2][1], face[2][2]};
    const auto reference = cross(f0, f1, f2);
    for (const auto &t : r.triangles[region]) {
      const auto n =
          cross(r.points[std::size_t(t[0])], r.points[std::size_t(t[1])],
                r.points[std::size_t(t[2])]);
      const Real dot = n[0] * reference[0] + n[1] * reference[1] +
                       n[2] * reference[2];
      if (dot <= Real(0))
        return false;
    }
  }
  return true;
}

auto count_bands(const cut_result &r) -> std::vector<Index> {
  auto bands = r.band_of_region;
  std::sort(bands.begin(), bands.end());
  return bands;
}

/// The band and the area of every region, band-major. A region's position is
/// an implementation fact; the band it carries and the area it covers are the
/// contract, so the assertions read them sorted.
auto banded_areas(const cut_result &r) -> std::vector<std::pair<Index, Real>> {
  std::vector<std::pair<Index, Real>> out;
  for (std::size_t i = 0; i < r.triangles.size(); ++i)
    out.emplace_back(r.band_of_region[i], region_area(r, i));
  std::sort(out.begin(), out.end());
  return out;
}

auto triangle(std::array<tf::point<Real, 3>, 3> pts) -> mesh_t {
  mesh_t mesh;
  for (const auto &p : pts)
    mesh.points_buffer().push_back(p);
  mesh.faces_buffer().push_back({Index(0), Index(1), Index(2)});
  return mesh;
}

/// A U — two prongs standing on a bar: area 20 inside a hull of 24, the
/// missing 4 being the notch. A level set above the notch floor meets the
/// boundary FOUR times, so ONE cut value states TWO chords on this one face.
auto u_face() -> mesh_t {
  mesh_t mesh;
  const std::array<std::array<Real, 2>, 8> corners{
      {{0, 0}, {6, 0}, {6, 4}, {4, 4}, {4, 2}, {2, 2}, {2, 4}, {0, 4}}};
  for (const auto &c : corners)
    mesh.points_buffer().push_back(tf::point<Real, 3>{c[0], c[1], 0});
  mesh.faces_buffer().push_back({Index(0), Index(1), Index(2), Index(3),
                                 Index(4), Index(5), Index(6), Index(7)});
  return mesh;
}

/// The field is the height, so a cut value is a horizontal line.
auto height_field(const mesh_t &mesh) -> std::vector<Real> {
  std::vector<Real> scalars;
  for (std::size_t i = 0; i < std::size_t(mesh.points().size()); ++i)
    scalars.push_back(mesh.points()[i][1]);
  return scalars;
}

} // namespace

TEST_CASE("iso face cut: no interior chord leaves one region",
          "[iso_face_cuts]") {
  // Two adjacent corners sit exactly on the cut, the third above: the chord
  // runs along a mesh edge and cuts nothing.
  auto mesh = triangle({tf::point<Real, 3>{0, 0, 0}, tf::point<Real, 3>{1, 0, 0},
                        tf::point<Real, 3>{0, 1, 0}});
  const std::vector<Real> scalars{1.0, 1.0, 2.0};
  const std::vector<Real> values{1.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() == 1);
  REQUIRE(r.band_of_region[0] == Index(1));
  REQUIRE(r.face_of_region[0] == Index(0));
  REQUIRE(std::abs(total_area(r) - Real(0.5)) < Real(1e-12));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: one chord makes two regions in two bands",
          "[iso_face_cuts]") {
  auto mesh = triangle({tf::point<Real, 3>{0, 0, 0}, tf::point<Real, 3>{1, 0, 0},
                        tf::point<Real, 3>{0, 1, 0}});
  const std::vector<Real> scalars{0.0, 2.0, 2.0};
  const std::vector<Real> values{1.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() == 2);
  REQUIRE(count_bands(r) == std::vector<Index>{Index(0), Index(1)});
  REQUIRE(std::abs(total_area(r) - Real(0.5)) < Real(1e-12));
  // The below-the-cut corner keeps a quarter of the triangle.
  for (std::size_t i = 0; i < 2; ++i)
    if (r.band_of_region[i] == Index(0))
      REQUIRE(std::abs(region_area(r, i) - Real(0.125)) < Real(1e-12));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: two chords make three bands", "[iso_face_cuts]") {
  auto mesh = triangle({tf::point<Real, 3>{0, 0, 0}, tf::point<Real, 3>{1, 0, 0},
                        tf::point<Real, 3>{0, 1, 0}});
  const std::vector<Real> scalars{0.0, 3.0, 3.0};
  const std::vector<Real> values{1.0, 2.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() == 3);
  REQUIRE(count_bands(r) ==
          std::vector<Index>{Index(0), Index(1), Index(2)});
  REQUIRE(std::abs(total_area(r) - Real(0.5)) < Real(1e-12));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: one cut value states two chords on one face",
          "[iso_face_cuts]") {
  // The line y = 3 meets the U's boundary four times, so the single cut value
  // owns TWO chords: one across each prong. They must pair prong-wise, and
  // the two prong tops are separate pieces of the SAME band.
  auto mesh = u_face();
  const auto scalars = height_field(mesh);
  const std::vector<Real> values{3.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.refused.empty());
  REQUIRE(r.triangles.size() == 3);
  REQUIRE(count_bands(r) ==
          std::vector<Index>{Index(0), Index(1), Index(1)});
  // The notch is a pocket of 4, so the hull's 24 come back as the U's 20.
  REQUIRE(std::abs(total_area(r) - Real(20.0)) < Real(1e-9));

  const auto banded = banded_areas(r);
  REQUIRE(banded[0].first == Index(0));
  REQUIRE(std::abs(banded[0].second - Real(16.0)) < Real(1e-9));
  REQUIRE(banded[1].first == Index(1));
  REQUIRE(std::abs(banded[1].second - Real(2.0)) < Real(1e-9));
  REQUIRE(banded[2].first == Index(1));
  REQUIRE(std::abs(banded[2].second - Real(2.0)) < Real(1e-9));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: a value chording twice joins one chording once",
          "[iso_face_cuts]") {
  // y = 1 crosses the bar once; y = 3 crosses each prong. Three chords from
  // two values on one face, and the middle band stays ONE piece because the
  // bar joins the prongs below the notch floor.
  auto mesh = u_face();
  const auto scalars = height_field(mesh);
  const std::vector<Real> values{1.0, 3.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.refused.empty());
  REQUIRE(r.triangles.size() == 4);
  REQUIRE(count_bands(r) ==
          std::vector<Index>{Index(0), Index(1), Index(2), Index(2)});
  REQUIRE(std::abs(total_area(r) - Real(20.0)) < Real(1e-9));

  const auto banded = banded_areas(r);
  REQUIRE(banded[0].first == Index(0));
  REQUIRE(std::abs(banded[0].second - Real(6.0)) < Real(1e-9));
  REQUIRE(banded[1].first == Index(1));
  REQUIRE(std::abs(banded[1].second - Real(10.0)) < Real(1e-9));
  REQUIRE(banded[2].first == Index(2));
  REQUIRE(std::abs(banded[2].second - Real(2.0)) < Real(1e-9));
  REQUIRE(banded[3].first == Index(2));
  REQUIRE(std::abs(banded[3].second - Real(2.0)) < Real(1e-9));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: a degenerate multi-chord face holds no piece",
          "[iso_face_cuts]") {
  // Face 0 is collinear, so its projection has no area and it can hold no
  // piece — yet both cut values chord it. It states its emptiness without
  // refusing, and the neighbour it shares an edge with cuts whole.
  mesh_t mesh;
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{1, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{2, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 2, 0});
  mesh.faces_buffer().push_back({Index(0), Index(1), Index(2)});
  mesh.faces_buffer().push_back({Index(0), Index(2), Index(3)});
  const std::vector<Real> scalars{0.0, 1.0, 2.0, 2.0};
  const std::vector<Real> values{0.5, 1.5};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.refused.empty());
  REQUIRE(r.triangles.size() == 3);
  for (const auto face : r.face_of_region)
    REQUIRE(face == Index(1));
  REQUIRE(count_bands(r) ==
          std::vector<Index>{Index(0), Index(1), Index(2)});
  REQUIRE(std::abs(total_area(r) - Real(2.0)) < Real(1e-9));

  const auto banded = banded_areas(r);
  REQUIRE(std::abs(banded[0].second - Real(0.125)) < Real(1e-9));
  REQUIRE(std::abs(banded[1].second - Real(1.0)) < Real(1e-9));
  REQUIRE(std::abs(banded[2].second - Real(0.875)) < Real(1e-9));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: a chord may end on an on-cut vertex",
          "[iso_face_cuts]") {
  // Corner 0 sits on the cut with a below-the-cut neighbour, so it anchors a
  // chord that lands on the opposite edge.
  mesh_t mesh;
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{1, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{1, 1, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 1, 0});
  mesh.faces_buffer().push_back({Index(0), Index(1), Index(2), Index(3)});
  const std::vector<Real> scalars{1.0, 2.0, 2.0, 0.0};
  const std::vector<Real> values{1.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() == 2);
  REQUIRE(count_bands(r) == std::vector<Index>{Index(0), Index(1)});
  REQUIRE(std::abs(total_area(r) - Real(1.0)) < Real(1e-12));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: a non-convex face keeps no pocket",
          "[iso_face_cuts]") {
  // An arrowhead: corner 3 is reflex, so the hull holds a pocket that must
  // stay outside every region.
  mesh_t mesh;
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{4, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{4, 4, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{2, 1, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 4, 0});
  mesh.faces_buffer().push_back(
      {Index(0), Index(1), Index(2), Index(3), Index(4)});
  // area = |shoelace| / 2 = 10
  const std::vector<Real> scalars{0.0, 4.0, 4.0, 1.0, 0.0};
  const std::vector<Real> values{2.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() >= 2);
  REQUIRE(std::abs(total_area(r) - Real(10.0)) < Real(1e-9));
  REQUIRE(winding_agrees(mesh, r));
}

TEST_CASE("iso face cut: an n-gon with collinear leading corners cuts",
          "[iso_face_cuts]") {
  // The face lies in x = 0, so a projection chosen from its first three
  // corners — which are collinear — drops the only two axes that separate it.
  mesh_t mesh;
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 2, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 4, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 4, 4});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 4});
  mesh.faces_buffer().push_back(
      {Index(0), Index(1), Index(2), Index(3), Index(4)});
  const std::vector<Real> scalars{0.0, 0.0, 0.0, 4.0, 4.0};
  const std::vector<Real> values{2.0};

  auto r = cut(mesh, scalars, values);
  REQUIRE(r.triangles.size() == 2);
  REQUIRE(count_bands(r) == std::vector<Index>{Index(0), Index(1)});
  REQUIRE(std::abs(total_area(r) - Real(16.0)) < Real(1e-9));
  for (std::size_t i = 0; i < 2; ++i)
    REQUIRE(std::abs(region_area(r, i) - Real(8.0)) < Real(1e-9));
}

TEST_CASE("iso face cut: a reversed source face keeps its own winding",
          "[iso_face_cuts]") {
  auto forward =
      triangle({tf::point<Real, 3>{0, 0, 0}, tf::point<Real, 3>{1, 0, 0},
                tf::point<Real, 3>{0, 1, 0}});
  auto reversed =
      triangle({tf::point<Real, 3>{0, 0, 0}, tf::point<Real, 3>{0, 1, 0},
                tf::point<Real, 3>{1, 0, 0}});
  const std::vector<Real> forward_scalars{0.0, 2.0, 2.0};
  const std::vector<Real> reversed_scalars{0.0, 2.0, 2.0};
  const std::vector<Real> values{1.0};

  auto rf = cut(forward, forward_scalars, values);
  auto rr = cut(reversed, reversed_scalars, values);
  REQUIRE(winding_agrees(forward, rf));
  REQUIRE(winding_agrees(reversed, rr));
  REQUIRE(std::abs(total_area(rf) - total_area(rr)) < Real(1e-12));
}
