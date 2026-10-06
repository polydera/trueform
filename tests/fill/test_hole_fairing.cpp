/**
 * @file test_hole_fairing.cpp
 * @brief Tests for the fairing operator, its solve and the seam it publishes
 *
 * Tests for:
 * - tf::fill::assemble_hole_fairing_system
 * - tf::fill::apply_hole_fairing_operator
 * - tf::fill::hole_fairing_form / hole_fairing_form_diagonal
 * - tf::fill::hole_fairing_is_anchored
 * - tf::fill::solve_hole_fairing
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/cross.hpp>
#include <trueform/core/dot.hpp>
#include <trueform/core/point.hpp>
#include <trueform/fill/holes/apply_hole_fairing_operator.hpp>
#include <trueform/fill/holes/hole_fairing_is_anchored.hpp>
#include <trueform/fill/holes/hole_fairing_stencil.hpp>
#include <trueform/fill/holes/hole_fairing_system.hpp>
#include <trueform/fill/holes/hole_metric.hpp>
#include <trueform/fill/holes/solve_hole_fairing.hpp>
#include <trueform/fill/holes/solve_hole_fairing_axis.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using fair_index = int;
using fair_triangle = std::array<fair_index, 3>;

/**
 * @brief A stencil stated directly: positions, triangles and the vertices the
 *        solve may move, canonicalised the way the gatherer canonicalises.
 */
auto fair_stencil_of(const std::vector<tf::point<double, 3>> &positions,
                     const std::vector<fair_triangle> &triangles,
                     const std::vector<fair_index> &free_vertices)
    -> tf::fill::hole_fairing_stencil<fair_index> {
  tf::fill::hole_fairing_stencil<fair_index> stencil;
  for (std::size_t k = 0; k < positions.size(); ++k) {
    stencil.names.push_back(fair_index(k));
    stencil.positions.push_back(positions[k]);
  }
  std::vector<fair_triangle> canonical;
  for (const auto &triangle : triangles) {
    const int first = tf::fill::hole_canonical_rotation(
        triangle[0], triangle[1], triangle[2]);
    canonical.push_back({triangle[std::size_t(first)],
                         triangle[std::size_t((first + 1) % 3)],
                         triangle[std::size_t((first + 2) % 3)]});
  }
  std::sort(canonical.begin(), canonical.end());
  for (const auto &triangle : canonical) {
    stencil.triangles.push_back(triangle);
    stencil.from_patch.push_back(char(0));
  }
  stencil.free_of.allocate(positions.size());
  for (std::size_t k = 0; k < positions.size(); ++k)
    stencil.free_of[k] = fair_index(-1);
  for (auto vertex : free_vertices) {
    stencil.free_of[std::size_t(vertex)] =
        fair_index(stencil.free_ids.size());
    stencil.free_ids.push_back(vertex);
  }
  return stencil;
}

/**
 * @brief The cotangent Laplacian and the lumped mass of a triangle list,
 *        assembled densely and independently of the library.
 */
struct fair_dense {
  std::size_t n;
  std::vector<double> laplacian;
  std::vector<double> mass;

  auto at(std::size_t row, std::size_t column) const -> double {
    return laplacian[row * n + column];
  }
};

auto fair_dense_of(const std::vector<tf::point<double, 3>> &positions,
                   const std::vector<fair_triangle> &triangles) -> fair_dense {
  fair_dense dense{positions.size(),
                   std::vector<double>(positions.size() * positions.size(),
                                       0.0),
                   std::vector<double>(positions.size(), 0.0)};
  for (const auto &triangle : triangles) {
    const auto p0 = positions[std::size_t(triangle[0])];
    const auto p1 = positions[std::size_t(triangle[1])];
    const auto p2 = positions[std::size_t(triangle[2])];
    const auto normal = tf::cross(p1 - p0, p2 - p0);
    const double two_area = std::sqrt(tf::dot(normal, normal));
    for (int corner = 0; corner < 3; ++corner)
      dense.mass[std::size_t(triangle[std::size_t(corner)])] += two_area / 6.0;
    const tf::point<double, 3> p[3] = {p0, p1, p2};
    for (int slot = 0; slot < 3; ++slot) {
      const std::size_t a = std::size_t(triangle[std::size_t(slot)]);
      const std::size_t b = std::size_t(triangle[std::size_t((slot + 1) % 3)]);
      const auto u = p[std::size_t(slot)] - p[std::size_t((slot + 2) % 3)];
      const auto v =
          p[std::size_t((slot + 1) % 3)] - p[std::size_t((slot + 2) % 3)];
      const double weight = 0.5 * tf::dot(u, v) / two_area;
      dense.laplacian[a * dense.n + b] -= weight;
      dense.laplacian[b * dense.n + a] -= weight;
      dense.laplacian[a * dense.n + a] += weight;
      dense.laplacian[b * dense.n + b] += weight;
    }
  }
  return dense;
}

/// `out = (k_s L + k_b L M^-1 L) x` over the whole dense stencil.
auto fair_dense_form(const fair_dense &dense, const std::vector<double> &x)
    -> std::vector<double> {
  std::vector<double> stretch(dense.n, 0.0), lumped(dense.n, 0.0),
      bend(dense.n, 0.0), out(dense.n, 0.0);
  for (std::size_t i = 0; i < dense.n; ++i)
    for (std::size_t j = 0; j < dense.n; ++j)
      stretch[i] += dense.at(i, j) * x[j];
  for (std::size_t i = 0; i < dense.n; ++i)
    lumped[i] = stretch[i] / dense.mass[i];
  for (std::size_t i = 0; i < dense.n; ++i)
    for (std::size_t j = 0; j < dense.n; ++j)
      bend[i] += dense.at(i, j) * lumped[j];
  for (std::size_t i = 0; i < dense.n; ++i)
    out[i] = tf::fill::hole_fairing_stiffness * stretch[i] +
             tf::fill::hole_fairing_bending * bend[i];
  return out;
}

auto fair_close(double a, double b) -> bool {
  return std::abs(a - b) <= 1e-9 * (1.0 + std::abs(a) + std::abs(b));
}

/// The round-6 middle-fan neighbourhood: a quad patch with two interior
/// points, and the host fan at one rim vertex whose middle triangle touches
/// the rim at that vertex alone.
const std::vector<tf::point<double, 3>> fair_fan_positions{
    {-2.0, 0.0, 0.0},  {0.0, -2.0, 0.0}, {2.0, 0.0, 0.0},
    {0.0, 2.0, 0.0},   {-0.5, 0.0, 0.5}, {0.5, 0.0, 0.5},
    {-2.0, -2.0, -0.5}, {0.0, -4.0, -0.5}};
const std::vector<fair_triangle> fair_fan_patch{
    {0, 1, 4}, {1, 5, 4}, {1, 2, 5}, {2, 3, 5}, {3, 4, 5}, {3, 0, 4}};
const std::vector<fair_triangle> fair_fan_host{{1, 0, 6}, {1, 6, 7}, {1, 7, 2}};

auto fair_fan_triangles(bool with_middle) -> std::vector<fair_triangle> {
  auto triangles = fair_fan_patch;
  for (std::size_t k = 0; k < fair_fan_host.size(); ++k)
    if (with_middle || k != 1)
      triangles.push_back(fair_fan_host[k]);
  return triangles;
}

const std::vector<fair_index> fair_fan_free{4, 5};

} // namespace

TEST_CASE("fill: the matrix-free action answers what a dense assembly does",
          "[fill][fairing]") {
  const auto triangles = fair_fan_triangles(true);
  const auto stencil =
      fair_stencil_of(fair_fan_positions, triangles, fair_fan_free);
  const auto dense = fair_dense_of(fair_fan_positions, triangles);

  tf::fill::hole_fairing_assembly_scratch<fair_index> assembly;
  tf::fill::hole_fairing_system<fair_index> system;
  REQUIRE(tf::fill::assemble_hole_fairing_system(
      stencil.triangles, stencil.positions, assembly, system));
  REQUIRE(system.size() == dense.n);
  for (std::size_t i = 0; i < dense.n; ++i) {
    REQUIRE(fair_close(system.mass[i], dense.mass[i]));
    REQUIRE(fair_close(system.diagonal[i], dense.at(i, i)));
  }

  const double tether = 0.037;
  const std::vector<double> state{0.37, -0.91};
  tf::buffer<double> in;
  for (auto value : state)
    in.push_back(value);
  tf::fill::hole_fairing_form_scratch form;
  tf::buffer<double> acted;
  tf::fill::apply_hole_fairing_operator(
      system, tf::fill::hole_fairing_stiffness, tf::fill::hole_fairing_bending,
      tether, stencil.free_ids, in, form, acted);

  std::vector<double> extended(dense.n, 0.0);
  for (std::size_t k = 0; k < fair_fan_free.size(); ++k)
    extended[std::size_t(fair_fan_free[k])] = state[k];
  const auto dense_acted = fair_dense_form(dense, extended);
  for (std::size_t k = 0; k < fair_fan_free.size(); ++k) {
    const std::size_t row = std::size_t(fair_fan_free[k]);
    REQUIRE(fair_close(acted[k], dense_acted[row] +
                                     tether * dense.mass[row] * state[k]));
  }

  for (std::size_t axis = 0; axis < 3; ++axis) {
    tf::buffer<double> coordinate;
    std::vector<double> reference(dense.n, 0.0);
    for (std::size_t k = 0; k < dense.n; ++k) {
      coordinate.push_back(fair_fan_positions[k][axis]);
      reference[k] = fair_fan_positions[k][axis];
    }
    tf::buffer<double> stated;
    tf::fill::hole_fairing_form(system, tf::fill::hole_fairing_stiffness,
                                tf::fill::hole_fairing_bending, coordinate,
                                form, stated);
    const auto dense_stated = fair_dense_form(dense, reference);
    for (std::size_t k = 0; k < fair_fan_free.size(); ++k) {
      const std::size_t row = std::size_t(fair_fan_free[k]);
      REQUIRE(fair_close(-stated[row], -dense_stated[row]));
    }
  }

  for (std::size_t k = 0; k < fair_fan_free.size(); ++k) {
    const std::size_t row = std::size_t(fair_fan_free[k]);
    double bend = 0.0;
    for (std::size_t j = 0; j < dense.n; ++j)
      bend += dense.at(row, j) * dense.at(row, j) / dense.mass[j];
    const double expected =
        tf::fill::hole_fairing_stiffness * dense.at(row, row) +
                            tf::fill::hole_fairing_bending * bend +
                            tether * dense.mass[row];
    REQUIRE(fair_close(tf::fill::hole_fairing_form_diagonal(
                           system, tf::fill::hole_fairing_stiffness,
                           tf::fill::hole_fairing_bending, row) +
                           tether * system.mass[row],
                       expected));
  }
}

TEST_CASE("fill: a reference without the middle host triangle disagrees",
          "[fill][fairing]") {
  const auto triangles = fair_fan_triangles(true);
  const auto stencil =
      fair_stencil_of(fair_fan_positions, triangles, fair_fan_free);
  const auto dense = fair_dense_of(fair_fan_positions, triangles);
  const auto partial =
      fair_dense_of(fair_fan_positions, fair_fan_triangles(false));

  tf::fill::hole_fairing_assembly_scratch<fair_index> assembly;
  tf::fill::hole_fairing_system<fair_index> system;
  REQUIRE(tf::fill::assemble_hole_fairing_system(
      stencil.triangles, stencil.positions, assembly, system));

  const std::vector<double> state{0.37, -0.91};
  std::vector<double> extended(dense.n, 0.0);
  for (std::size_t k = 0; k < fair_fan_free.size(); ++k)
    extended[std::size_t(fair_fan_free[k])] = state[k];
  const auto whole = fair_dense_form(dense, extended);
  const auto without = fair_dense_form(partial, extended);

  bool differs = false;
  for (auto free_vertex : fair_fan_free) {
    const std::size_t row = std::size_t(free_vertex);
    differs = differs || !fair_close(whole[row], without[row]);
  }
  REQUIRE(differs);
}

TEST_CASE("fill: a permuted triangle block states the same operator",
          "[fill][fairing]") {
  auto permuted = fair_fan_triangles(true);
  std::reverse(permuted.begin(), permuted.end());
  for (auto &triangle : permuted)
    triangle = {triangle[1], triangle[2], triangle[0]};

  const auto ordered = fair_stencil_of(fair_fan_positions,
                                       fair_fan_triangles(true), fair_fan_free);
  const auto shuffled =
      fair_stencil_of(fair_fan_positions, permuted, fair_fan_free);
  REQUIRE(ordered.triangles.size() == shuffled.triangles.size());
  for (std::size_t k = 0; k < ordered.triangles.size(); ++k)
    REQUIRE(ordered.triangles[k] == shuffled.triangles[k]);

  tf::fill::hole_fairing_scratch<fair_index> first, second;
  tf::buffer<tf::point<double, 3>> left, right;
  REQUIRE(tf::fill::solve_hole_fairing(ordered, first, left));
  REQUIRE(tf::fill::solve_hole_fairing(shuffled, second, right));
  REQUIRE(left.size() == right.size());
  for (std::size_t k = 0; k < left.size(); ++k)
    for (std::size_t axis = 0; axis < 3; ++axis)
      REQUIRE(left[k][axis] == right[k][axis]);
}

TEST_CASE("fill: a run of free vertices that reaches no fixed one takes the "
          "tether",
          "[fill][fairing]") {
  const std::vector<tf::point<double, 3>> corners{
      {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 4.0}};
  const std::vector<fair_triangle> faces{
      {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  const auto stencil = fair_stencil_of(corners, faces, {0, 1, 2, 3});

  tf::fill::hole_fairing_assembly_scratch<fair_index> assembly;
  tf::fill::hole_fairing_system<fair_index> system;
  REQUIRE(tf::fill::assemble_hole_fairing_system(
      stencil.triangles, stencil.positions, assembly, system));
  tf::buffer<fair_index> stack;
  tf::buffer<char> seen;
  REQUIRE_FALSE(tf::fill::hole_fairing_is_anchored(system, stencil.free_ids,
                                                   stencil.free_of, stack,
                                                   seen));

  const auto anchored = fair_stencil_of(corners, faces, {3});
  REQUIRE(tf::fill::hole_fairing_is_anchored(system, anchored.free_ids,
                                             anchored.free_of, stack, seen));
}

TEST_CASE("fill: the tethered solve answers a translated stencil alike",
          "[fill][fairing]") {
  const std::vector<tf::point<double, 3>> corners{
      {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 4.0}};
  const std::vector<fair_triangle> faces{
      {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  const tf::point<double, 3> shift{8.0, -16.0, 32.0};
  std::vector<tf::point<double, 3>> moved;
  for (const auto &corner : corners)
    moved.push_back({corner[0] + shift[0], corner[1] + shift[1],
                     corner[2] + shift[2]});

  const auto here = fair_stencil_of(corners, faces, {0, 1, 2, 3});
  const auto there = fair_stencil_of(moved, faces, {0, 1, 2, 3});
  tf::fill::hole_fairing_scratch<fair_index> first, second;
  tf::buffer<tf::point<double, 3>> left, right;
  REQUIRE(tf::fill::solve_hole_fairing(here, first, left));
  REQUIRE(tf::fill::solve_hole_fairing(there, second, right));
  REQUIRE(left.size() == 4);
  REQUIRE(right.size() == 4);
  bool moved_at_all = false;
  for (std::size_t k = 0; k < left.size(); ++k) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      REQUIRE(std::abs(right[k][axis] - (left[k][axis] + shift[axis])) <
              1e-9 * (1.0 + std::abs(shift[axis])));
      moved_at_all = moved_at_all ||
                     std::abs(left[k][axis] - corners[k][axis]) > 1e-12;
    }
  }
  REQUIRE(moved_at_all);
}

TEST_CASE("fill: a stencil with no free vertex is answered without moving",
          "[fill][fairing]") {
  const auto stencil =
      fair_stencil_of(fair_fan_positions, fair_fan_triangles(true), {});
  tf::fill::hole_fairing_scratch<fair_index> scratch;
  tf::buffer<tf::point<double, 3>> out;
  REQUIRE(tf::fill::solve_hole_fairing(stencil, scratch, out));
  REQUIRE(out.size() == 0);
}

TEST_CASE("fill: a stencil triangle that states no area refuses the solve",
          "[fill][fairing]") {
  auto positions = fair_fan_positions;
  positions[6] = tf::point<double, 3>{-2.0, 0.0, 0.0};
  positions[7] = tf::point<double, 3>{-2.0, 0.0, 0.0};
  const auto stencil =
      fair_stencil_of(positions, fair_fan_triangles(true), fair_fan_free);
  tf::fill::hole_fairing_scratch<fair_index> scratch;
  tf::buffer<tf::point<double, 3>> out;
  REQUIRE_FALSE(tf::fill::solve_hole_fairing(stencil, scratch, out));
  REQUIRE(out.size() == 0);
}
