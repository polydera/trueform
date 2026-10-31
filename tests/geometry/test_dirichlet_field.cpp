/**
 * @file test_dirichlet_field.cpp
 * @brief Tests for the Dirichlet field over a weighted Laplacian
 *
 * Tests for:
 * - tf::make_dirichlet_field
 * - tf::geometry::laplacian_is_anchored
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <trueform/core/buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/make_dirichlet_field.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <type_traits>
#include <vector>

namespace {

/**
 * @brief The rings of a plane grid of `ticks` cells per side, counted from
 *        its border inwards: ring 0 is the border.
 */
auto dirichlet_grid_ring(float x, float y, float half, float step) -> int {
  const float inset = half - std::max(std::abs(x), std::abs(y));
  return int(std::lround(inset / step));
}

/**
 * @brief The Laplacian and lumped mass of a triangle list under stated
 *        corner weights, assembled densely and independently of the library,
 *        and its Dirichlet field solved by Gaussian elimination.
 */
auto dirichlet_dense_field(const std::vector<std::array<double, 3>> &points,
                           const std::vector<std::array<int, 3>> &faces,
                           const std::vector<double> &corner_weights,
                           const std::vector<int> &fixed,
                           const std::vector<double> &values, int order)
    -> std::vector<double> {
  const std::size_t n = points.size();
  std::vector<double> l(n * n, 0.0), mass(n, 0.0);
  for (std::size_t f = 0; f < faces.size(); ++f) {
    const auto &t = faces[f];
    double e1[3], e2[3];
    for (int a = 0; a < 3; ++a) {
      e1[a] = points[std::size_t(t[1])][std::size_t(a)] -
              points[std::size_t(t[0])][std::size_t(a)];
      e2[a] = points[std::size_t(t[2])][std::size_t(a)] -
              points[std::size_t(t[0])][std::size_t(a)];
    }
    const double cx = e1[1] * e2[2] - e1[2] * e2[1];
    const double cy = e1[2] * e2[0] - e1[0] * e2[2];
    const double cz = e1[0] * e2[1] - e1[1] * e2[0];
    const double area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    for (int c = 0; c < 3; ++c) {
      mass[std::size_t(t[std::size_t(c)])] += area / 3.0;
      const auto a = std::size_t(t[std::size_t((c + 1) % 3)]);
      const auto b = std::size_t(t[std::size_t((c + 2) % 3)]);
      const double w = corner_weights[3 * f + std::size_t(c)];
      l[a * n + b] -= w;
      l[b * n + a] -= w;
      l[a * n + a] += w;
      l[b * n + b] += w;
    }
  }
  std::vector<double> q = l;
  if (order == 2)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double value = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          value += l[i * n + k] * l[k * n + j] / mass[k];
        q[i * n + j] = value;
      }
  std::vector<double> x(n, 0.0);
  std::vector<char> held(n, 0);
  for (std::size_t k = 0; k < fixed.size(); ++k) {
    x[std::size_t(fixed[k])] = values[k];
    held[std::size_t(fixed[k])] = 1;
  }
  std::vector<std::size_t> free_rows;
  for (std::size_t i = 0; i < n; ++i)
    if (!held[i])
      free_rows.push_back(i);
  const std::size_t m = free_rows.size();
  std::vector<double> a(m * (m + 1), 0.0);
  for (std::size_t r = 0; r < m; ++r) {
    double rhs = 0.0;
    for (std::size_t j = 0; j < n; ++j)
      if (held[j])
        rhs -= q[free_rows[r] * n + j] * x[j];
    for (std::size_t c = 0; c < m; ++c)
      a[r * (m + 1) + c] = q[free_rows[r] * n + free_rows[c]];
    a[r * (m + 1) + m] = rhs;
  }
  for (std::size_t p = 0; p < m; ++p) {
    std::size_t pivot = p;
    for (std::size_t r = p + 1; r < m; ++r)
      if (std::abs(a[r * (m + 1) + p]) > std::abs(a[pivot * (m + 1) + p]))
        pivot = r;
    for (std::size_t c = 0; c <= m; ++c)
      std::swap(a[p * (m + 1) + c], a[pivot * (m + 1) + c]);
    for (std::size_t r = 0; r < m; ++r) {
      if (r == p)
        continue;
      const double factor = a[r * (m + 1) + p] / a[p * (m + 1) + p];
      for (std::size_t c = p; c <= m; ++c)
        a[r * (m + 1) + c] -= factor * a[p * (m + 1) + c];
    }
  }
  for (std::size_t r = 0; r < m; ++r)
    x[free_rows[r]] = a[r * (m + 1) + m] / a[r * (m + 1) + r];
  return x;
}

/**
 * @brief A sphere cap's points, faces and a corner weight per face corner
 *        that varies by two orders of magnitude, so a weighted field departs
 *        measurably from the cotangent one.
 */
struct dirichlet_cap {
  std::vector<std::array<double, 3>> points;
  std::vector<std::array<int, 3>> faces;
  std::vector<double> weights;
  std::vector<int> fixed;
  std::vector<double> values;
};

auto dirichlet_cap_of() -> dirichlet_cap {
  const auto sphere = tf::make_sphere_mesh<int>(1.0f, 12, 12);
  dirichlet_cap cap;
  std::vector<int> remap(sphere.points().size(), -1);
  for (std::size_t f = 0; f < sphere.faces().size(); ++f) {
    const auto face = sphere.faces()[f];
    bool upper = true;
    for (int c = 0; c < 3; ++c)
      upper = upper && sphere.points()[face[c]][2] > -0.2f;
    if (!upper)
      continue;
    std::array<int, 3> mapped{};
    for (int c = 0; c < 3; ++c) {
      auto &slot = remap[std::size_t(face[c])];
      if (slot < 0) {
        slot = int(cap.points.size());
        const auto p = sphere.points()[face[c]];
        cap.points.push_back({double(p[0]), double(p[1]), double(p[2])});
      }
      mapped[std::size_t(c)] = slot;
    }
    cap.faces.push_back(mapped);
  }
  for (std::size_t f = 0; f < cap.faces.size(); ++f)
    for (int c = 0; c < 3; ++c)
      cap.weights.push_back(0.05 +
                            double((7 * f + 3 * std::size_t(c)) % 11) / 2.0);
  for (std::size_t k = 0; k < cap.points.size(); ++k) {
    const auto &p = cap.points[k];
    if (p[2] < 0.05) {
      cap.fixed.push_back(int(k));
      cap.values.push_back(0.0);
    } else if (p[2] > 0.95) {
      cap.fixed.push_back(int(k));
      cap.values.push_back(1.0);
    }
  }
  return cap;
}

auto dirichlet_cap_mesh(const dirichlet_cap &cap)
    -> tf::polygons_buffer<int, double, 3, 3> {
  tf::polygons_buffer<int, double, 3, 3> mesh;
  mesh.points_buffer().allocate(cap.points.size());
  for (std::size_t k = 0; k < cap.points.size(); ++k)
    for (std::size_t a = 0; a < 3; ++a)
      mesh.points()[k][a] = cap.points[k][a];
  mesh.faces_buffer().allocate(cap.faces.size());
  for (std::size_t f = 0; f < cap.faces.size(); ++f)
    for (std::size_t c = 0; c < 3; ++c)
      mesh.faces()[f][c] = cap.faces[f][c];
  return mesh;
}

} // namespace

TEST_CASE("dirichlet_field: the cotangent field reproduces a linear one on a "
          "plane",
          "[geometry][dirichlet]") {
  const auto plane = tf::make_plane_mesh<int>(4.0f, 4.0f, 12, 12);
  const auto linear = [](float x, float y) {
    return 2.0 * double(x) - 3.0 * double(y) + 0.5;
  };
  for (const auto order :
       {tf::dirichlet_order::harmonic, tf::dirichlet_order::biharmonic}) {
    // The biharmonic field is held on two rings, the harmonic one on one.
    const int rings = order == tf::dirichlet_order::biharmonic ? 2 : 1;
    std::vector<int> fixed;
    std::vector<double> values;
    for (std::size_t k = 0; k < plane.points().size(); ++k) {
      const auto p = plane.points()[k];
      if (dirichlet_grid_ring(p[0], p[1], 2.0f, 4.0f / 12.0f) < rings) {
        fixed.push_back(int(k));
        values.push_back(linear(p[0], p[1]));
      }
    }
    const auto result = tf::make_dirichlet_field<double>(
        plane.polygons(), tf::none, fixed, values, order);
    REQUIRE(result.status == tf::dirichlet_field_status::solved);
    REQUIRE(result.field.size() == plane.points().size());
    std::size_t free_count = 0;
    for (std::size_t k = 0; k < plane.points().size(); ++k) {
      const auto p = plane.points()[k];
      REQUIRE(std::abs(result.field[k] - linear(p[0], p[1])) < 1e-6);
      free_count +=
          dirichlet_grid_ring(p[0], p[1], 2.0f, 4.0f / 12.0f) >= rings;
    }
    REQUIRE(free_count > 50);
  }
}

TEST_CASE("dirichlet_field: stated weights answer the dense solve",
          "[geometry][dirichlet]") {
  const auto cap = dirichlet_cap_of();
  const auto mesh = dirichlet_cap_mesh(cap);
  REQUIRE(cap.fixed.size() > 5);
  REQUIRE(cap.points.size() - cap.fixed.size() > 20);
  for (const auto order :
       {tf::dirichlet_order::harmonic, tf::dirichlet_order::biharmonic}) {
    const auto dense = dirichlet_dense_field(
        cap.points, cap.faces, cap.weights, cap.fixed, cap.values,
        order == tf::dirichlet_order::biharmonic ? 2 : 1);
    const auto result = tf::make_dirichlet_field(mesh.polygons(), cap.weights,
                                                 cap.fixed, cap.values, order);
    REQUIRE(result.status == tf::dirichlet_field_status::solved);
    double spread = 0.0;
    for (std::size_t k = 0; k < cap.points.size(); ++k) {
      REQUIRE(std::abs(result.field[k] - dense[k]) < 1e-10);
      spread = std::max(spread, std::abs(dense[k] - 0.5));
    }
    REQUIRE(spread > 0.3);
  }
}

TEST_CASE("dirichlet_field: stated weights are read, not replaced by the "
          "cotangent ones",
          "[geometry][dirichlet]") {
  const auto cap = dirichlet_cap_of();
  const auto mesh = dirichlet_cap_mesh(cap);
  const auto weighted =
      tf::make_dirichlet_field(mesh.polygons(), cap.weights, cap.fixed,
                               cap.values, tf::dirichlet_order::harmonic);
  const auto cotangent =
      tf::make_dirichlet_field(mesh.polygons(), tf::none, cap.fixed, cap.values,
                               tf::dirichlet_order::harmonic);
  REQUIRE(weighted.status == tf::dirichlet_field_status::solved);
  REQUIRE(cotangent.status == tf::dirichlet_field_status::solved);
  double apart = 0.0;
  for (std::size_t k = 0; k < cap.points.size(); ++k)
    apart = std::max(apart, std::abs(weighted.field[k] - cotangent.field[k]));
  REQUIRE(apart > 0.05);
}

TEST_CASE("dirichlet_field: a run of free points that reaches no fixed one "
          "refuses and holds its fixed rows",
          "[geometry][dirichlet]") {
  auto cap = dirichlet_cap_of();
  const int base = int(cap.points.size());
  cap.points.push_back({4.0, 0.0, 0.0});
  cap.points.push_back({5.0, 0.0, 0.0});
  cap.points.push_back({4.0, 1.0, 0.0});
  cap.faces.push_back({base, base + 1, base + 2});
  // These weights leave the island's singular block a last pivot that rounds
  // positive, so only the anchoring proof refuses it.
  for (double w : {0.1, 0.3, 0.7})
    cap.weights.push_back(w);
  const auto mesh = dirichlet_cap_mesh(cap);
  for (const auto order :
       {tf::dirichlet_order::harmonic, tf::dirichlet_order::biharmonic}) {
    const auto result = tf::make_dirichlet_field(mesh.polygons(), cap.weights,
                                                 cap.fixed, cap.values, order);
    REQUIRE(result.status == tf::dirichlet_field_status::refused_unanchored);
    REQUIRE(result.field.size() == cap.points.size());
    for (std::size_t k = 0; k < cap.fixed.size(); ++k)
      REQUIRE(result.field[std::size_t(cap.fixed[k])] == cap.values[k]);
  }
  cap.fixed.push_back(base);
  cap.values.push_back(0.25);
  const auto held = dirichlet_cap_mesh(cap);
  REQUIRE(tf::make_dirichlet_field(held.polygons(), cap.weights, cap.fixed,
                                   cap.values, tf::dirichlet_order::biharmonic)
              .status == tf::dirichlet_field_status::solved);
}

TEST_CASE("dirichlet_field: weights that leave the system indefinite refuse",
          "[geometry][dirichlet]") {
  auto cap = dirichlet_cap_of();
  for (auto &w : cap.weights)
    w = -w;
  const auto mesh = dirichlet_cap_mesh(cap);
  const auto result =
      tf::make_dirichlet_field(mesh.polygons(), cap.weights, cap.fixed,
                               cap.values, tf::dirichlet_order::harmonic);
  REQUIRE(result.status == tf::dirichlet_field_status::refused_indefinite);
  for (std::size_t k = 0; k < cap.fixed.size(); ++k)
    REQUIRE(result.field[std::size_t(cap.fixed[k])] == cap.values[k]);
}

TEST_CASE("dirichlet_field: a point no triangle holds states no field",
          "[geometry][dirichlet]") {
  auto cap = dirichlet_cap_of();
  cap.points.push_back({5.0, 5.0, 5.0});
  const auto mesh = dirichlet_cap_mesh(cap);
  const auto result =
      tf::make_dirichlet_field(mesh.polygons(), cap.weights, cap.fixed,
                               cap.values, tf::dirichlet_order::harmonic);
  REQUIRE(result.status == tf::dirichlet_field_status::refused_operator);
  for (std::size_t k = 0; k < cap.fixed.size(); ++k)
    REQUIRE(result.field[std::size_t(cap.fixed[k])] == cap.values[k]);
}

TEST_CASE("dirichlet_field: the field is stated in the resolved output type",
          "[geometry][dirichlet]") {
  const auto plane = tf::make_plane_mesh<int>(4.0f, 4.0f, 6, 6);
  std::vector<int> fixed;
  std::vector<double> values;
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    if (dirichlet_grid_ring(plane.points()[k][0], plane.points()[k][1], 2.0f,
                            4.0f / 6.0f) < 1) {
      fixed.push_back(int(k));
      values.push_back(double(plane.points()[k][0]));
    }
  const auto own = tf::make_dirichlet_field(
      plane.polygons(), tf::none, fixed, values, tf::dirichlet_order::harmonic);
  const auto wide = tf::make_dirichlet_field<double>(
      plane.polygons(), tf::none, fixed, values, tf::dirichlet_order::harmonic);
  static_assert(std::is_same_v<decltype(own.field[0]), float &> ||
                    std::is_same_v<decltype(own.field[0]), const float &>,
                "a float mesh states a float field");
  static_assert(std::is_same_v<decltype(wide.field[0]), double &> ||
                    std::is_same_v<decltype(wide.field[0]), const double &>,
                "a stated double is a double field");
  REQUIRE(own.status == tf::dirichlet_field_status::solved);
  REQUIRE(wide.status == tf::dirichlet_field_status::solved);
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    REQUIRE(own.field[k] == float(wide.field[k]));
}
