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
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/volume/make_mesh_sdf.hpp>
#include <vector>

namespace {

/// Signed distance of a point to a convex mesh stated as half-spaces: the
/// largest signed plane distance, negative strictly inside every face plane.
auto sdfsign_halfspace(const std::vector<std::array<double, 4>> &planes,
                       const std::array<double, 3> &p) -> double {
  double worst = -1e300;
  for (const auto &pl : planes)
    worst = std::max(worst, pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3]);
  return worst;
}

auto sdfsign_plane(const std::array<double, 3> &a,
                   const std::array<double, 3> &b,
                   const std::array<double, 3> &c) -> std::array<double, 4> {
  const std::array<double, 3> u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const std::array<double, 3> v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  std::array<double, 3> n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]};
  const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (auto &c2 : n)
    c2 /= len;
  return {n[0], n[1], n[2], -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2])};
}

} // namespace

TEST_CASE("mesh sdf sign is exact on the sliver tetrahedron",
          "[volume][sdf][sign]") {
  // The pseudonormal failure class: a thin watertight tetrahedron and a dense
  // sample battery. Every sample clear of the surface must carry the
  // half-space oracle's sign.
  tf::polygons_buffer<int, float, 3, 3> tetra;
  const std::array<std::array<double, 3>, 4> vs{{{0.0, 0.0, 0.0},
                                                 {1.0, 0.0, 0.0},
                                                 {0.5, 1.0, 0.0},
                                                 {0.5, 0.4, 0.005}}};
  for (const auto &v : vs)
    tetra.points_buffer().push_back({float(v[0]), float(v[1]), float(v[2])});
  const std::array<std::array<int, 3>, 4> fs{
      {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}}};
  for (const auto &f : fs)
    tetra.faces_buffer().push_back({f[0], f[1], f[2]});

  std::vector<std::array<double, 4>> planes;
  for (const auto &f : fs)
    planes.push_back(sdfsign_plane(vs[std::size_t(f[0])], vs[std::size_t(f[1])],
                                   vs[std::size_t(f[2])]));

  tf::aabb_tree<int, float, 3> tree(tetra.polygons(), tf::config_tree(4, 4));
  auto form = tetra.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{64, 64, 32};
  const tf::point<float, 3> spacing{0.03f, 0.025f, 0.002f};
  const tf::point<float, 3> origin{-0.45f, -0.3f, -0.026f};
  auto vol = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto field = vol.volume();

  std::size_t wrong = 0, checked = 0;
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const auto p = field.point_at<double>(x, y, z);
        const double oracle = sdfsign_halfspace(planes, {p[0], p[1], p[2]});
        if (std::abs(oracle) < 1e-6)
          continue;
        ++checked;
        if ((oracle < 0.0) != (field(x, y, z) < 0.0f))
          ++wrong;
      }
  REQUIRE(checked > 100000u);
  REQUIRE(wrong == 0u);
}

TEST_CASE("mesh sdf sign matches the convex oracle on a sphere mesh",
          "[volume][sdf][sign]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  std::vector<std::array<double, 4>> planes;
  for (std::size_t f = 0; f < sphere.size(); ++f) {
    auto face = sphere.faces()[f];
    auto pt = [&](int k) -> std::array<double, 3> {
      auto p = sphere.points()[std::size_t(face[std::size_t(k)])];
      return {double(p[0]), double(p[1]), double(p[2])};
    };
    planes.push_back(sdfsign_plane(pt(0), pt(1), pt(2)));
  }

  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{41, 41, 41};
  const tf::point<float, 3> spacing{0.065f, 0.065f, 0.065f};
  const tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};
  auto vol = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto field = vol.volume();

  std::size_t wrong = 0, checked = 0;
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const auto p = field.point_at<double>(x, y, z);
        const double oracle = sdfsign_halfspace(planes, {p[0], p[1], p[2]});
        if (std::abs(oracle) < 1e-5)
          continue;
        ++checked;
        if ((oracle < 0.0) != (field(x, y, z) < 0.0f))
          ++wrong;
      }
  REQUIRE(checked > 50000u);
  REQUIRE(wrong == 0u);
}

TEST_CASE("mesh sdf sign holds at the reflex edge of an L prism",
          "[volume][sdf][sign]") {
  // The L outline (0,0)(2,0)(2,1)(1,1)(1,2)(0,2) extruded to z in [0,1]:
  // a reflex edge at (1,1), where angle-weighted pseudonormals nearly cancel.
  tf::polygons_buffer<int, float, 3, 3> prism;
  const std::array<std::array<float, 2>, 6> ring{
      {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}}};
  for (int level = 0; level < 2; ++level)
    for (const auto &q : ring)
      prism.points_buffer().push_back({q[0], q[1], float(level)});
  const auto cap = std::array<std::array<int, 3>, 4>{
      {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}}};
  for (const auto &t : cap) {
    prism.faces_buffer().push_back({t[0], t[2], t[1]});
    prism.faces_buffer().push_back({t[0] + 6, t[1] + 6, t[2] + 6});
  }
  for (int i = 0; i < 6; ++i) {
    const int j = (i + 1) % 6;
    prism.faces_buffer().push_back({i, j, j + 6});
    prism.faces_buffer().push_back({i, j + 6, i + 6});
  }

  auto inside_l = [](const std::array<double, 3> &p) {
    const bool xy = (p[0] > 0 && p[0] < 2 && p[1] > 0 && p[1] < 1) ||
                    (p[0] > 0 && p[0] < 1 && p[1] > 0 && p[1] < 2);
    return xy && p[2] > 0 && p[2] < 1;
  };
  auto boundary_close = [](const std::array<double, 3> &p) {
    auto near_wall = [](double v, double w) { return std::abs(v - w) < 1e-6; };
    return near_wall(p[0], 0) || near_wall(p[0], 1) || near_wall(p[0], 2) ||
           near_wall(p[1], 0) || near_wall(p[1], 1) || near_wall(p[1], 2) ||
           near_wall(p[2], 0) || near_wall(p[2], 1);
  };

  tf::aabb_tree<int, float, 3> tree(prism.polygons(), tf::config_tree(4, 4));
  auto form = prism.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{45, 45, 25};
  const tf::point<float, 3> spacing{0.061f, 0.061f, 0.063f};
  const tf::point<float, 3> origin{-0.35f, -0.35f, -0.27f};
  auto vol = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto field = vol.volume();

  std::size_t wrong = 0, checked = 0;
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const auto pf = field.point_at<double>(x, y, z);
        const std::array<double, 3> p{pf[0], pf[1], pf[2]};
        if (boundary_close(p))
          continue;
        ++checked;
        if (inside_l(p) != (field(x, y, z) < 0.0f))
          ++wrong;
      }
  REQUIRE(checked > 40000u);
  REQUIRE(wrong == 0u);
}

TEST_CASE("mesh sdf sign is deterministic on samples landing on the surface",
          "[volume][sdf][sign]") {
  // A 2x2x2 box with the grid arranged so samples land exactly on faces,
  // edges, and corners: the sign there is the perturbation's to decide, and
  // the decision must be the same on every run.
  auto box = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f);
  tf::aabb_tree<int, float, 3> tree(box.polygons(), tf::config_tree(4, 4));
  auto form = box.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{9, 9, 9};
  const tf::point<float, 3> spacing{0.5f, 0.5f, 0.5f};
  const tf::point<float, 3> origin{-2.0f, -2.0f, -2.0f};

  auto vol0 = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto vol1 = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto f0 = vol0.volume();
  auto f1 = vol1.volume();

  for (int z = 0; z < 9; ++z)
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 9; ++x) {
        const float a = f0(x, y, z);
        REQUIRE(a == f1(x, y, z));
        const auto p = f0.point_at<double>(x, y, z);
        const double m =
            std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
        if (m < 1.0 - 1e-9)
          REQUIRE(a < 0.0f);
        else if (m > 1.0 + 1e-9)
          REQUIRE(a > 0.0f);
        else
          REQUIRE(std::abs(a) < 1e-5f);
      }
}

TEST_CASE("the field's sign is direction-free", "[volume][sdf][sign]") {
  // Negative spacing runs a grid axis against the lattice order; the
  // binning normalizes, so every axis direction signs alike.
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  std::vector<std::array<double, 4>> planes;
  for (std::size_t f = 0; f < sphere.size(); ++f) {
    auto face = sphere.faces()[f];
    auto pt = [&](int k) -> std::array<double, 3> {
      auto p = sphere.points()[std::size_t(face[std::size_t(k)])];
      return {double(p[0]), double(p[1]), double(p[2])};
    };
    planes.push_back(sdfsign_plane(pt(0), pt(1), pt(2)));
  }
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  const int g = 25;
  const float h = 2.6f / float(g - 1);
  for (int neg = 0; neg < 3; ++neg) {
    tf::point<float, 3> spacing{h, h, h};
    tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};
    spacing[std::size_t(neg)] = -h;
    origin[std::size_t(neg)] = 1.3f;
    auto vol = tf::make_mesh_sdf<float>(form, {g, g, g}, spacing, origin);
    auto field = vol.volume();
    std::size_t wrong = 0, checked = 0;
    for (int z = 0; z < g; ++z)
      for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
          const auto p = field.point_at<double>(x, y, z);
          const double oracle = sdfsign_halfspace(planes, {p[0], p[1], p[2]});
          if (std::abs(oracle) < 1e-5)
            continue;
          ++checked;
          if ((oracle < 0.0) != (field(x, y, z) < 0.0f))
            ++wrong;
        }
    REQUIRE(checked > 10000u);
    REQUIRE(wrong == 0u);
  }
}

TEST_CASE("winding signs the field", "[volume][sdf][sign]") {
  // Negative inside by winding: an inverted shell inverts its field,
  // nested outward shells stay solid inside, and a hollow shell's cavity
  // is outside.
  const auto sphere_at = [](float r, bool inverted) {
    auto mesh = tf::make_sphere_mesh<int>(r, 16, 16);
    if (inverted)
      for (std::size_t f = 0; f < mesh.size(); ++f) {
        auto face = mesh.faces_buffer()[f];
        std::swap(face[1], face[2]);
      }
    return mesh;
  };
  const auto merged = [](tf::polygons_buffer<int, float, 3, 3> a,
                         const tf::polygons_buffer<int, float, 3, 3> &b) {
    const int base = int(a.points_buffer().size());
    for (std::size_t i = 0; i < b.points_buffer().size(); ++i)
      a.points_buffer().push_back(b.points_buffer()[i]);
    for (std::size_t f = 0; f < b.size(); ++f) {
      auto face = b.faces_buffer()[f];
      a.faces_buffer().push_back(
          {face[0] + base, face[1] + base, face[2] + base});
    }
    return a;
  };
  const std::array<int, 3> dims{25, 25, 25};
  const tf::point<float, 3> spacing{0.1f, 0.1f, 0.1f};
  const tf::point<float, 3> origin{-1.2f, -1.2f, -1.2f};
  const auto field_of = [&](tf::polygons_buffer<int, float, 3, 3> &mesh) {
    tf::aabb_tree<int, float, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    return tf::make_mesh_sdf<float>(mesh.polygons() | tf::tag(tree), dims,
                                    spacing, origin);
  };

  auto inverted = sphere_at(1.0f, true);
  auto vi = field_of(inverted);
  REQUIRE(vi.volume()(12, 12, 12) > 0.0f); // center: inversion inverts
  REQUIRE(vi.volume()(0, 0, 0) > 0.0f);    // corner outside either way

  auto nested = merged(sphere_at(1.0f, false), sphere_at(0.5f, false));
  auto vn = field_of(nested);
  REQUIRE(vn.volume()(12, 12, 12) < 0.0f); // inside both: stays solid
  REQUIRE(vn.volume()(12, 12, 4) < 0.0f);  // between the shells
  REQUIRE(vn.volume()(0, 0, 0) > 0.0f);

  auto hollow = merged(sphere_at(1.0f, false), sphere_at(0.5f, true));
  auto vh = field_of(hollow);
  REQUIRE(vh.volume()(12, 12, 12) > 0.0f); // the cavity is outside
  REQUIRE(vh.volume()(12, 12, 4) < 0.0f);  // the wall is solid
  REQUIRE(vh.volume()(0, 0, 0) > 0.0f);
}

TEST_CASE("the field holds for double coordinates and dynamic faces",
          "[volume][sdf][sign]") {
  {
    auto sphere = tf::make_sphere_mesh<int>(1.0, 16, 16);
    tf::aabb_tree<int, double, 3> tree(sphere.polygons(),
                                       tf::config_tree(4, 4));
    auto vol = tf::make_mesh_sdf<double>(
        sphere.polygons() | tf::tag(tree), {15, 15, 15},
        tf::point<double, 3>{0.2, 0.2, 0.2},
        tf::point<double, 3>{-1.4, -1.4, -1.4});
    REQUIRE(vol.volume()(7, 7, 7) < 0.0);
    REQUIRE(vol.volume()(0, 0, 0) > 0.0);
  }
  {
    // the 2x2x2 box as six quads
    tf::polygons_buffer<int, float, 3, tf::dynamic_size> box;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          box.points_buffer().push_back(
              {float(2 * x - 1), float(2 * y - 1), float(2 * z - 1)});
    const std::array<std::array<int, 4>, 6> quads{{{0, 2, 3, 1},
                                                   {4, 5, 7, 6},
                                                   {0, 1, 5, 4},
                                                   {2, 6, 7, 3},
                                                   {0, 4, 6, 2},
                                                   {1, 3, 7, 5}}};
    for (const auto &qd : quads)
      box.faces_buffer().push_back(tf::make_range(qd));
    tf::aabb_tree<int, float, 3> tree(box.polygons(), tf::config_tree(4, 4));
    auto vol = tf::make_mesh_sdf<float>(
        box.polygons() | tf::tag(tree), {15, 15, 15},
        tf::point<float, 3>{0.2f, 0.2f, 0.2f},
        tf::point<float, 3>{-1.4f, -1.4f, -1.4f});
    REQUIRE(vol.volume()(7, 7, 7) < 0.0f);
    REQUIRE(vol.volume()(0, 0, 0) > 0.0f);
    REQUIRE(vol.volume()(14, 14, 14) > 0.0f);
  }
}
