/**
 * @file test_cholesky.cpp
 * @brief Tests for the sparse Cholesky factor, its pattern and its solves
 *
 * Tests for:
 * - tf::analyze_cholesky
 * - tf::factor_cholesky
 * - tf::solve_cholesky, against the dense factor's triangular solves
 * - tf::linalg::analyze_cholesky_order
 * - tf::linalg::factor_front_columns
 * - tf::linalg::refine_once
 * - tf::linalg::partition_elimination_tree
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/geometry/laplacian/assemble_laplacian_system.hpp>
#include <trueform/geometry/laplacian/laplacian_form.hpp>
#include <trueform/geometry/laplacian/make_laplacian_form_pattern.hpp>
#include <trueform/geometry/laplacian/state_laplacian_form_values.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/linalg/analyze_cholesky.hpp>
#include <trueform/linalg/cholesky/analyze_cholesky_order.hpp>
#include <trueform/linalg/cholesky/factor_front_columns.hpp>
#include <trueform/linalg/cholesky/make_symmetric_adjacency.hpp>
#include <trueform/linalg/cholesky/partition_elimination_tree.hpp>
#include <trueform/linalg/cholesky/refine_once.hpp>
#include <trueform/linalg/factor_cholesky.hpp>
#include <trueform/linalg/solve_cholesky.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <vector>

namespace {

using chol_index = int;

/**
 * @brief One fixture: a mesh's free rows of `k_s L + k_b L M^-1 L + tau M`,
 *        their weights log-uniform over `decades` and every component held.
 */
struct chol_fixture_spec {
  bool sphere;
  int ticks;
  int order;
  double tether;
  double decades;
  unsigned seed;
};

auto chol_form_of(const chol_fixture_spec &spec)
    -> tf::geometry::laplacian_form<chol_index> {
  const auto mesh =
      spec.sphere
          ? tf::make_sphere_mesh<chol_index>(1.0f, spec.ticks, spec.ticks + 2)
          : tf::make_plane_mesh<chol_index>(2.0f, 2.0f, spec.ticks, spec.ticks);
  std::mt19937 random(spec.seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> weights(3 * mesh.faces().size());
  for (auto &w : weights)
    w = std::pow(10.0, -spec.decades / 2.0 + spec.decades * unit(random));
  tf::face_membership<chol_index> membership;
  membership.build(mesh.polygons());
  tf::geometry::laplacian_assembly_scratch assembly;
  tf::geometry::laplacian_system<chol_index> system;
  REQUIRE(tf::geometry::assemble_laplacian_system(
      mesh.polygons() | tf::tag(membership),
      [&weights](std::size_t t, std::size_t slot,
                 const tf::point<double, 3>(&)[3],
                 double) { return weights[3 * t + slot]; },
      assembly, system));
  const std::size_t n = mesh.points().size();
  tf::buffer<chol_index> free_of;
  free_of.allocate_and_initialize(n, chol_index(0));
  free_of[0] = -1;
  for (std::size_t k = 1; k < n; ++k)
    if (unit(random) < 0.1)
      free_of[k] = -1;
  tf::buffer<chol_index> free_ids;
  for (std::size_t k = 0; k < n; ++k)
    if (free_of[k] == 0) {
      free_of[k] = chol_index(free_ids.size());
      free_ids.push_back(chol_index(k));
    }
  tf::geometry::laplacian_form<chol_index> form;
  tf::geometry::make_laplacian_form_pattern(system, free_ids, free_of,
                                            spec.order == 2, form);
  tf::geometry::state_laplacian_form_values(
      system, free_ids, free_of, spec.order == 1 ? 1.0 : 0.1,
      spec.order == 1 ? 0.0 : 1.0, spec.tether, form);
  return form;
}

const std::vector<chol_fixture_spec> chol_fixtures{
    {false, 8, 1, 0.0, 0.0, 1},  {false, 8, 2, 0.0, 0.0, 2},
    {true, 6, 1, 0.0, 0.0, 3},   {true, 6, 2, 0.0, 0.0, 4},
    {false, 12, 1, 0.0, 9.0, 5}, {false, 12, 2, 0.0, 9.0, 6},
    {false, 12, 2, 0.3, 9.0, 7}, {true, 9, 1, 0.0, 9.0, 8},
    {true, 9, 2, 0.0, 9.0, 9},   {true, 9, 1, 0.5, 9.0, 10}};

/**
 * @brief A pattern read under a stated order, without the minimum degree
 *        walk, so the factor is checked apart from it.
 */
auto chol_pattern_under(const tf::symmetric_lower<chol_index> &matrix,
                        const std::vector<chol_index> &order,
                        tf::cholesky_scratch<chol_index> &scratch)
    -> tf::cholesky_pattern<chol_index> {
  tf::buffer<chol_index> cursor, elimination;
  for (auto row : order)
    elimination.push_back(row);
  tf::cholesky_pattern<chol_index> pattern;
  tf::linalg::make_symmetric_adjacency(
      matrix, cursor, pattern.adjacency_offsets, pattern.adjacency,
      pattern.adjacency_entries);
  REQUIRE(tf::linalg::analyze_cholesky_order(matrix, elimination, scratch,
                                             pattern));
  return pattern;
}

auto chol_pattern_under(const tf::symmetric_lower<chol_index> &matrix,
                        const std::vector<chol_index> &order)
    -> tf::cholesky_pattern<chol_index> {
  tf::cholesky_scratch<chol_index> scratch;
  return chol_pattern_under(matrix, order, scratch);
}

/**
 * @brief Cut a pattern's supernodal tree into `target` blocks, each
 *        supernode weighing its columns' squared row counts.
 */
auto chol_partition(tf::cholesky_pattern<chol_index> &pattern,
                    std::size_t target) -> void {
  tf::buffer<double> work;
  for (std::size_t s = 0; s < pattern.number_of_supernodes(); ++s) {
    const double k =
        double(pattern.supernode_offsets[s + 1] - pattern.supernode_offsets[s]);
    const double m =
        double(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
    double w = 0.0;
    for (double t = 0.0; t < k; t += 1.0)
      w += (m - t) * (m - t);
    work.push_back(w);
  }
  tf::buffer<chol_index> size, heap;
  tf::linalg::partition_elimination_tree(pattern.supernode_parent, target, work,
                                         size, heap, pattern.schedule,
                                         pattern.block_offsets);
}

/**
 * @brief The entries of a factor: each value with its row and column, the
 *        triangle above each block's diagonal included as row -1.
 */
template <typename F>
auto chol_each_entry(const tf::cholesky_pattern<chol_index> &pattern,
                     const tf::cholesky_factor &factor, F &&f) -> void {
  for (std::size_t s = 0; s < pattern.number_of_supernodes(); ++s) {
    const auto first = std::size_t(pattern.supernode_offsets[s]);
    const auto k = std::size_t(pattern.supernode_offsets[s + 1]) - first;
    const auto m =
        std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
    REQUIRE(std::size_t(pattern.value_offsets[s + 1] -
                        pattern.value_offsets[s]) == m * k);
    for (std::size_t t = 0; t < k; ++t)
      for (std::size_t p = 0; p < m; ++p)
        f(p < t ? chol_index(-1)
                : pattern.rows[std::size_t(pattern.row_offsets[s]) + p],
          chol_index(first + t),
          factor.values[std::size_t(pattern.value_offsets[s]) + t * m + p]);
  }
}

auto chol_bitwise(const tf::buffer<double> &a, const tf::buffer<double> &b)
    -> bool {
  return a.size() == b.size() &&
         std::memcmp(a.begin(), b.begin(), a.size() * sizeof(double)) == 0;
}

auto chol_natural(std::size_t n) -> std::vector<chol_index> {
  std::vector<chol_index> order(n);
  for (std::size_t k = 0; k < n; ++k)
    order[k] = chol_index(k);
  return order;
}

auto chol_shuffled(std::size_t n, unsigned seed) -> std::vector<chol_index> {
  auto order = chol_natural(n);
  std::mt19937 random(seed);
  std::shuffle(order.begin(), order.end(), random);
  return order;
}

auto chol_dense(const tf::symmetric_lower<chol_index> &matrix)
    -> std::vector<double> {
  const std::size_t n = matrix.size();
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (auto at = matrix.row_offsets[i]; at < matrix.row_offsets[i + 1];
         ++at) {
      const std::size_t j = std::size_t(matrix.columns[std::size_t(at)]);
      a[i * n + j] = a[j * n + i] = matrix.values[std::size_t(at)];
    }
  return a;
}

/**
 * @brief Solve over a pattern: factor, solve, refine once, and require the
 *        answer accepted.
 */
auto chol_solve(const tf::symmetric_lower<chol_index> &matrix,
                const tf::cholesky_pattern<chol_index> &pattern,
                const tf::buffer<double> &b, std::size_t r)
    -> tf::buffer<double> {
  tf::cholesky_scratch<chol_index> scratch;
  tf::cholesky_factor factor;
  REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));
  tf::buffer<double> x = b;
  tf::solve_cholesky(pattern, factor, x, r, scratch);
  REQUIRE(tf::linalg::refine_once(matrix, pattern, factor, b, r, scratch, x));
  return x;
}

auto chol_rhs(std::size_t n, std::size_t r, unsigned seed)
    -> tf::buffer<double> {
  std::mt19937 random(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  tf::buffer<double> b;
  for (std::size_t k = 0; k < n * r; ++k)
    b.push_back(unit(random));
  return b;
}

auto chol_max_abs(const tf::buffer<double> &x) -> double {
  double value = 0.0;
  for (auto v : x)
    value = std::max(value, std::abs(v));
  return value;
}

/**
 * @brief The 2-norm condition number, by power and inverse iteration over
 *        the factor itself.
 */
auto chol_condition(const tf::symmetric_lower<chol_index> &matrix,
                    const tf::cholesky_pattern<chol_index> &pattern) -> double {
  const std::size_t n = matrix.size();
  tf::cholesky_scratch<chol_index> scratch;
  tf::cholesky_factor factor;
  REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));
  const auto norm = [](const tf::buffer<double> &v) {
    double s = 0.0;
    for (auto x : v)
      s += x * x;
    return std::sqrt(s);
  };
  tf::buffer<double> zero, v, w, scale;
  zero.allocate_and_initialize(n, 0.0);
  v = chol_rhs(n, 1, 77);
  double largest = 0.0;
  for (int it = 0; it < 200; ++it) {
    tf::linalg::symmetric_lower_residual(matrix, pattern, zero, v, 1, w, scale);
    largest = norm(w) / norm(v);
    for (std::size_t k = 0; k < n; ++k)
      v[k] = -w[k] / norm(w);
  }
  v = chol_rhs(n, 1, 78);
  double smallest = 0.0;
  for (int it = 0; it < 200; ++it) {
    w = v;
    tf::solve_cholesky(pattern, factor, w, 1, scratch);
    smallest = norm(v) / norm(w);
    const double scale = norm(w);
    for (std::size_t k = 0; k < n; ++k)
      v[k] = w[k] / scale;
  }
  return largest / smallest;
}

} // namespace

TEST_CASE("cholesky: the factor is the dense factor of the permuted matrix",
          "[linalg][cholesky]") {
  for (std::size_t f = 0; f < 4; ++f) {
    const auto form = chol_form_of(chol_fixtures[f]);
    const auto &matrix = form.free;
    const std::size_t n = matrix.size();
    REQUIRE(n > 20);
    REQUIRE(n <= 400);
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> pattern;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, pattern));
    tf::cholesky_factor factor;
    REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));

    const auto a = chol_dense(matrix);
    std::vector<double> l(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        l[i * n + j] = a[std::size_t(pattern.order[i]) * n +
                         std::size_t(pattern.order[j])];
    for (std::size_t j = 0; j < n; ++j) {
      l[j * n + j] = std::sqrt(l[j * n + j]);
      for (std::size_t i = j + 1; i < n; ++i)
        l[i * n + j] /= l[j * n + j];
      for (std::size_t c = j + 1; c < n; ++c)
        for (std::size_t i = c; i < n; ++i)
          l[i * n + c] -= l[i * n + j] * l[c * n + j];
    }
    double largest = 0.0;
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = j; i < n; ++i)
        largest = std::max(largest, std::abs(l[i * n + j]));
    std::vector<char> stated(n * n, 0);
    std::size_t padded = 0;
    chol_each_entry(
        pattern, factor, [&](chol_index row, chol_index column, double value) {
          if (row == -1) {
            REQUIRE(value == 0.0);
            return;
          }
          const std::size_t i = std::size_t(row), j = std::size_t(column);
          REQUIRE(!stated[i * n + j]);
          stated[i * n + j] = 1;
          REQUIRE(std::abs(value - l[i * n + j]) <= 1e-12 * largest);
          // A structural zero is exactly zero in both: each of its products has
          // a structurally zero factor.
          if (l[i * n + j] == 0.0) {
            REQUIRE(value == 0.0);
            ++padded;
          }
        });
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = j; i < n; ++i)
        if (!stated[i * n + j])
          REQUIRE(l[i * n + j] == 0.0);
    // The relaxed supernodes hold explicit zeros on every fixture here.
    REQUIRE(padded > 0);
  }
}

TEST_CASE("cholesky: a solve is the dense factor's two triangular solves",
          "[linalg][cholesky]") {
  for (std::size_t f = 0; f < 4; ++f) {
    const auto form = chol_form_of(chol_fixtures[f]);
    const auto &matrix = form.free;
    const std::size_t n = matrix.size();
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> pattern;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, pattern));
    tf::cholesky_factor factor;
    REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));
    std::vector<double> l(n * n, 0.0);
    chol_each_entry(pattern, factor,
                    [&](chol_index row, chol_index column, double value) {
                      if (row != -1)
                        l[std::size_t(row) * n + std::size_t(column)] = value;
                    });

    const std::size_t r = 2;
    const auto b = chol_rhs(n, r, chol_fixtures[f].seed + 200);
    auto x = b;
    tf::solve_cholesky(pattern, factor, x, r, scratch);
    const double agreement = 10.0 * chol_condition(matrix, pattern) *
                             std::numeric_limits<double>::epsilon();
    for (std::size_t c = 0; c < r; ++c) {
      std::vector<double> y(n);
      for (std::size_t k = 0; k < n; ++k)
        y[k] = b[std::size_t(pattern.order[k]) * r + c];
      for (std::size_t j = 0; j < n; ++j) {
        y[j] /= l[j * n + j];
        for (std::size_t i = j + 1; i < n; ++i)
          y[i] -= l[i * n + j] * y[j];
      }
      for (std::size_t j = n; j-- > 0;) {
        for (std::size_t i = j + 1; i < n; ++i)
          y[j] -= l[i * n + j] * y[i];
        y[j] /= l[j * n + j];
      }
      double largest = 0.0, apart = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        const double got = x[std::size_t(pattern.order[k]) * r + c];
        largest = std::max(largest, std::abs(y[k]));
        apart = std::max(apart, std::abs(got - y[k]));
      }
      REQUIRE(apart <= agreement * largest);
    }
  }
}

TEST_CASE("cholesky: every ordering answers alike over nine decades of weight",
          "[linalg][cholesky]") {
  for (const auto &spec : chol_fixtures) {
    const auto form = chol_form_of(spec);
    const auto &matrix = form.free;
    const std::size_t n = matrix.size();
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> ordered;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, ordered));
    const double kappa = chol_condition(matrix, ordered);
    const auto b = chol_rhs(n, 2, spec.seed);
    const auto x = chol_solve(matrix, ordered, b, 2);

    // The componentwise backward error: a residual measured against `|b|`
    // alone would grow with the condition however stable the solve.
    tf::buffer<double> residual, scale;
    tf::linalg::symmetric_lower_residual(matrix, ordered, b, x, 2, residual,
                                         scale);
    for (std::size_t k = 0; k < residual.size(); ++k)
      REQUIRE(std::abs(residual[k]) <= 1e-12 * scale[k]);

    const double agreement =
        10.0 * kappa * std::numeric_limits<double>::epsilon();
    for (const auto &order : {chol_natural(n), chol_shuffled(n, spec.seed),
                              std::vector<chol_index>(ordered.order.begin(),
                                                      ordered.order.end())}) {
      const auto other =
          chol_solve(matrix, chol_pattern_under(matrix, order), b, 2);
      double apart = 0.0;
      for (std::size_t k = 0; k < x.size(); ++k)
        apart = std::max(apart, std::abs(other[k] - x[k]));
      REQUIRE(apart <= agreement * chol_max_abs(x));
    }
  }
}

TEST_CASE("cholesky: a pattern states the laws its factor stands on",
          "[linalg][cholesky]") {
  for (const auto &spec : chol_fixtures) {
    const auto form = chol_form_of(spec);
    const auto &matrix = form.free;
    const std::size_t n = matrix.size();
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> pattern;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, pattern));
    for (std::size_t k = 0; k < n; ++k)
      REQUIRE(std::size_t(pattern.rank[std::size_t(pattern.order[k])]) == k);

    const std::size_t supernodes = pattern.number_of_supernodes();
    REQUIRE(pattern.supernode_offsets[0] == 0);
    REQUIRE(std::size_t(pattern.supernode_offsets[supernodes]) == n);
    REQUIRE(supernodes < n);
    std::vector<chol_index> size(supernodes, 1);
    std::size_t stored = 0;
    for (std::size_t s = 0; s < supernodes; ++s) {
      const auto first = pattern.supernode_offsets[s];
      const auto k = std::size_t(pattern.supernode_offsets[s + 1] - first);
      REQUIRE(k > 0);
      const auto *rows =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[s]);
      const auto m =
          std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
      REQUIRE(std::size_t(pattern.value_offsets[s]) == stored);
      stored += m * k;
      for (std::size_t t = 0; t < k; ++t)
        REQUIRE(rows[t] == chol_index(first + chol_index(t)));
      for (std::size_t p = 1; p < m; ++p)
        REQUIRE(rows[p - 1] < rows[p]);
      const auto up = pattern.supernode_parent[s];
      REQUIRE((up == -1) == (m == k));
      if (up == -1)
        continue;
      REQUIRE(std::size_t(up) > s);
      size[std::size_t(up)] += size[s];
      // The update a supernode passes up lands among its parent's rows.
      const auto *above =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[up]);
      const auto *above_end =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[up + 1]);
      for (std::size_t p = k; p < m; ++p) {
        above = std::lower_bound(above, above_end, rows[p]);
        REQUIRE((above != above_end && *above == rows[p]));
      }
    }
    REQUIRE(std::size_t(pattern.value_offsets[supernodes]) == stored);
    for (std::size_t s = 0; s < supernodes; ++s) {
      const auto up = pattern.supernode_parent[s];
      if (up != -1)
        REQUIRE(chol_index(s) > up - size[std::size_t(up)]);
    }

    tf::cholesky_factor factor;
    REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));
    REQUIRE(factor.values.size() == stored);
    // A minimum degree order beats the natural one only where the ordering
    // has room to decide; on a few dozen rows a mesh's own numbering can
    // fill less.
    tf::cholesky_scratch<chol_index> natural;
    chol_pattern_under(matrix, chol_natural(n), natural);
    std::size_t fill = 0, natural_fill = 0;
    for (std::size_t k = 0; k < n; ++k) {
      fill += std::size_t(scratch.counts[k]);
      natural_fill += std::size_t(natural.counts[k]);
    }
    if (n >= 100)
      REQUIRE(fill <= natural_fill);
  }
}

TEST_CASE("cholesky: one pattern refactors and one factor batches, bitwise",
          "[linalg][cholesky]") {
  for (const auto &spec : chol_fixtures) {
    const auto form = chol_form_of(spec);
    auto matrix = form.free;
    const std::size_t n = matrix.size();
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> pattern;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, pattern));
    tf::cholesky_factor factor;
    REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));

    for (std::size_t i = 0; i < n; ++i)
      matrix.values[std::size_t(matrix.row_offsets[i + 1]) - 1] *= 1.5;
    REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, factor));
    tf::cholesky_scratch<chol_index> fresh_scratch;
    tf::cholesky_pattern<chol_index> fresh_pattern;
    tf::cholesky_factor fresh;
    REQUIRE(tf::analyze_cholesky(matrix, fresh_scratch, fresh_pattern));
    REQUIRE(tf::factor_cholesky(fresh_pattern, matrix.values, fresh_scratch,
                                fresh));
    REQUIRE(chol_bitwise(fresh.values, factor.values));

    const std::size_t r = 3;
    auto batch = chol_rhs(n, r, spec.seed + 100);
    const auto stated = batch;
    tf::solve_cholesky(pattern, factor, batch, r, scratch);
    for (std::size_t c = 0; c < r; ++c) {
      tf::buffer<double> single;
      for (std::size_t k = 0; k < n; ++k)
        single.push_back(stated[k * r + c]);
      tf::solve_cholesky(pattern, factor, single, 1, scratch);
      for (std::size_t k = 0; k < n; ++k)
        REQUIRE(single[k] == batch[k * r + c]);
    }

    const auto again = chol_solve(matrix, fresh_pattern, stated, r);
    const auto once = chol_solve(matrix, pattern, stated, r);
    for (std::size_t k = 0; k < once.size(); ++k)
      REQUIRE(again[k] == once[k]);
  }
}

TEST_CASE("cholesky: a matrix that is not definite refuses",
          "[linalg][cholesky]") {
  const auto form = chol_form_of(chol_fixtures[4]);
  tf::cholesky_scratch<chol_index> scratch;
  tf::cholesky_pattern<chol_index> pattern;
  REQUIRE(tf::analyze_cholesky(form.free, scratch, pattern));
  tf::cholesky_factor factor;
  REQUIRE(tf::factor_cholesky(pattern, form.free.values, scratch, factor));

  auto indefinite = form.free.values;
  const std::size_t middle = form.free.size() / 2;
  indefinite[std::size_t(form.free.row_offsets[middle + 1]) - 1] *= -1.0;
  REQUIRE_FALSE(tf::factor_cholesky(pattern, indefinite, scratch, factor));

  auto negated = form.free.values;
  for (auto &v : negated)
    v = -v;
  REQUIRE_FALSE(tf::factor_cholesky(pattern, negated, scratch, factor));

  auto unstated = form.free.values;
  unstated[std::size_t(form.free.row_offsets[middle])] =
      std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(tf::factor_cholesky(pattern, unstated, scratch, factor));

  REQUIRE(tf::factor_cholesky(pattern, form.free.values, scratch, factor));
}

TEST_CASE("cholesky: every partition at every thread count factors and "
          "solves to the same bits",
          "[linalg][cholesky]") {
  // The grid's top fronts span several tiles and pass the parallel
  // assembly's cutoff, so every parallel path of a front runs.
  const std::vector<chol_fixture_spec> specs{{false, 80, 2, 0.0, 9.0, 21},
                                             {true, 30, 1, 0.0, 9.0, 22}};
  for (const auto &spec : specs) {
    const auto form = chol_form_of(spec);
    const auto &matrix = form.free;
    const std::size_t n = matrix.size();
    tf::cholesky_scratch<chol_index> scratch;
    tf::cholesky_pattern<chol_index> pattern;
    REQUIRE(tf::analyze_cholesky(matrix, scratch, pattern));
    std::size_t widest = 0;
    for (std::size_t s = 0; s < pattern.number_of_supernodes(); ++s)
      widest = std::max(widest, std::size_t(pattern.row_offsets[s + 1] -
                                            pattern.row_offsets[s]));
    if (spec.order == 2)
      REQUIRE(widest > 2 * tf::linalg::cholesky_tile_size);

    chol_partition(pattern, 1);
    tf::cholesky_factor reference;
    tf::buffer<double> reference_x, reference_x1;
    const std::size_t r = 3;
    const auto b = chol_rhs(n, r, spec.seed);
    tf::buffer<double> b1;
    for (std::size_t k = 0; k < n; ++k)
      b1.push_back(b[k * r]);
    {
      tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                  1);
      REQUIRE(tf::factor_cholesky(pattern, matrix.values, scratch, reference));
      reference_x = b;
      tf::solve_cholesky(pattern, reference, reference_x, r, scratch);
      reference_x1 = b1;
      tf::solve_cholesky(pattern, reference, reference_x1, 1, scratch);
    }

    std::size_t frontiers = 0;
    for (std::size_t target : {1, 3, 8, 32, 200}) {
      chol_partition(pattern, target);
      frontiers += pattern.frontier_first() < pattern.number_of_supernodes();
      for (std::size_t threads :
           {std::size_t(1), std::size_t(2),
            std::size_t(tbb::this_task_arena::max_concurrency())}) {
        tbb::global_control control(
            tbb::global_control::max_allowed_parallelism, threads);
        for (int run = 0; run < 2; ++run) {
          tf::cholesky_scratch<chol_index> local;
          tf::cholesky_factor factor;
          REQUIRE(tf::factor_cholesky(pattern, matrix.values, local, factor));
          REQUIRE(chol_bitwise(factor.values, reference.values));
          auto x = b;
          tf::solve_cholesky(pattern, factor, x, r, local);
          REQUIRE(chol_bitwise(x, reference_x));
          auto x1 = b1;
          tf::solve_cholesky(pattern, factor, x1, 1, local);
          REQUIRE(chol_bitwise(x1, reference_x1));
        }
      }

      // A pivot that fails in a block or in the frontier refuses alike.
      for (std::size_t slot :
           {std::size_t(0), pattern.number_of_supernodes() - 1}) {
        const std::size_t row = std::size_t(pattern.order[std::size_t(
            pattern.supernode_offsets[std::size_t(pattern.schedule[slot])])]);
        auto broken = matrix.values;
        broken[std::size_t(matrix.row_offsets[row + 1]) - 1] *= -1.0;
        tf::cholesky_scratch<chol_index> local;
        tf::cholesky_factor factor;
        REQUIRE_FALSE(tf::factor_cholesky(pattern, broken, local, factor));
      }
    }
    REQUIRE(frontiers >= 3);
  }
}

TEST_CASE("cholesky: a dense front factors its columns and updates the rest",
          "[linalg][cholesky]") {
  constexpr std::size_t P = tf::linalg::cholesky_panel_width;
  constexpr std::size_t T = tf::linalg::cholesky_tile_size;
  // Fronts whose columns end inside, on and past panel and tile edges.
  const std::vector<std::array<std::size_t, 2>> shapes{{1, 1},
                                                       {9, 3},
                                                       {P, P},
                                                       {P + 1, P},
                                                       {2 * T + 5, 1},
                                                       {3 * P + 7, P + 3},
                                                       {2 * T + 1, 2 * P},
                                                       {150, 150}};
  std::mt19937 random(5);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  for (const auto &shape : shapes) {
    const std::size_t m = shape[0], k = shape[1];
    std::vector<double> a(m * m);
    for (std::size_t j = 0; j < m; ++j)
      for (std::size_t i = j; i < m; ++i)
        a[j * m + i] = a[i * m + j] = unit(random);
    for (std::size_t i = 0; i < m; ++i)
      a[i * m + i] += double(m);
    std::vector<double> head(m * k), tail((m - k) * (m - k));
    for (std::size_t j = 0; j < k; ++j)
      for (std::size_t i = 0; i < m; ++i)
        head[j * m + i] = a[j * m + i];
    for (std::size_t j = k; j < m; ++j)
      for (std::size_t i = k; i < m; ++i)
        tail[(j - k) * (m - k) + i - k] = a[j * m + i];
    REQUIRE(tf::linalg::factor_front_columns(head.data(), tail.data(), m, k));

    auto l = a;
    for (std::size_t j = 0; j < k; ++j) {
      l[j * m + j] = std::sqrt(l[j * m + j]);
      for (std::size_t i = j + 1; i < m; ++i)
        l[j * m + i] /= l[j * m + j];
      for (std::size_t c = j + 1; c < m; ++c)
        for (std::size_t i = c; i < m; ++i)
          l[c * m + i] -= l[j * m + i] * l[j * m + c];
    }
    for (std::size_t j = 0; j < m; ++j)
      for (std::size_t i = 0; i < m; ++i) {
        if (j < k && i < j) {
          // The triangle above the diagonal is never written.
          REQUIRE(head[j * m + i] == a[j * m + i]);
          continue;
        }
        if (i < j)
          continue;
        const double got =
            j < k ? head[j * m + i] : tail[(j - k) * (m - k) + i - k];
        REQUIRE(std::abs(got - l[j * m + i]) <= 1e-12 * double(m));
      }

    // A pivot that is not positive refuses, wherever its panel falls.
    for (std::size_t j = 0; j < k; ++j)
      for (std::size_t i = 0; i < m; ++i)
        head[j * m + i] = a[j * m + i];
    for (std::size_t j = k; j < m; ++j)
      for (std::size_t i = k; i < m; ++i)
        tail[(j - k) * (m - k) + i - k] = a[j * m + i];
    head[(k - 1) * m + (k - 1)] = -1.0;
    REQUIRE_FALSE(
        tf::linalg::factor_front_columns(head.data(), tail.data(), m, k));
  }
}
