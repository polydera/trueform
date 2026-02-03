/**
 * @file test_least_squares_parallel.cpp
 * @brief Tests for parallel TSQR least squares solver
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/linalg/least_squares.hpp>
#include <trueform/core/linalg/least_squares_parallel.hpp>
#include <trueform/core/buffer.hpp>
#include <cmath>
#include <random>

// =============================================================================
// Helper functions
// =============================================================================

template <typename T>
auto approx_equal(T a, T b, T tol = T(1e-5)) -> bool {
  return std::abs(a - b) < tol;
}

template <typename T>
auto relative_error(const T *x1, const T *x2, std::size_t n) -> T {
  T norm_diff = T(0);
  T norm_x1 = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    T d = x1[i] - x2[i];
    norm_diff += d * d;
    norm_x1 += x1[i] * x1[i];
  }
  if (norm_x1 < std::numeric_limits<T>::min())
    return std::sqrt(norm_diff);
  return std::sqrt(norm_diff / norm_x1);
}

// Generate random tall-skinny matrix and RHS
template <typename T>
void generate_random_system(T *A, T *b, std::size_t rows, std::size_t cols,
                            unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(T(-1), T(1));

  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < rows; ++i) {
      A[i + j * rows] = dist(gen);
    }
  }
  for (std::size_t i = 0; i < rows; ++i) {
    b[i] = dist(gen);
  }
}

// Generate system with known solution: A*x_true = b (overdetermined)
template <typename T>
void generate_system_with_solution(T *A, T *b, const T *x_true,
                                   std::size_t rows, std::size_t cols,
                                   unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(T(-1), T(1));

  // Generate random A
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < rows; ++i) {
      A[i + j * rows] = dist(gen);
    }
  }

  // Compute b = A * x_true
  for (std::size_t i = 0; i < rows; ++i) {
    b[i] = T(0);
    for (std::size_t j = 0; j < cols; ++j) {
      b[i] += A[i + j * rows] * x_true[j];
    }
  }
}

// =============================================================================
// Tests comparing parallel vs sequential solver
// =============================================================================

TEMPLATE_TEST_CASE("parallel_vs_sequential_random", "[core][linalg][parallel]",
                   float, double) {
  using T = TestType;
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-10);

  SECTION("small matrix (falls back to sequential)") {
    constexpr std::size_t rows = 100;
    constexpr std::size_t cols = 6;

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_random_system(A.data(), b.data(), rows, cols);

    // Sequential solve
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A.begin(), A.end(), A_copy.begin());
    tf::buffer<T> work;
    work.allocate(tf::linalg::least_squares_workspace_size<T>(rows, cols));
    std::array<T, cols> x_seq{};
    tf::linalg::solve_least_squares(A_copy.data(), b.data(), x_seq.data(), rows,
                                    cols, work.data());

    // Parallel solve
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    T err = relative_error(x_seq.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  }

  SECTION("medium matrix") {
    constexpr std::size_t rows = 2000;
    constexpr std::size_t cols = 6;

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_random_system(A.data(), b.data(), rows, cols);

    // Sequential solve
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A.begin(), A.end(), A_copy.begin());
    tf::buffer<T> work;
    work.allocate(tf::linalg::least_squares_workspace_size<T>(rows, cols));
    std::array<T, cols> x_seq{};
    tf::linalg::solve_least_squares(A_copy.data(), b.data(), x_seq.data(), rows,
                                    cols, work.data());

    // Parallel solve
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    T err = relative_error(x_seq.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  }

  SECTION("large matrix") {
    constexpr std::size_t rows = 50000;
    constexpr std::size_t cols = 6;

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_random_system(A.data(), b.data(), rows, cols);

    // Sequential solve
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A.begin(), A.end(), A_copy.begin());
    tf::buffer<T> work;
    work.allocate(tf::linalg::least_squares_workspace_size<T>(rows, cols));
    std::array<T, cols> x_seq{};
    tf::linalg::solve_least_squares(A_copy.data(), b.data(), x_seq.data(), rows,
                                    cols, work.data());

    // Parallel solve
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    T err = relative_error(x_seq.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  }
}

// =============================================================================
// Tests with known solution
// =============================================================================

TEMPLATE_TEST_CASE("parallel_known_solution", "[core][linalg][parallel]",
                   float, double) {
  using T = TestType;
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-10);

  SECTION("exact solution recovery") {
    constexpr std::size_t rows = 10000;
    constexpr std::size_t cols = 6;

    std::array<T, cols> x_true = {T(1), T(-2), T(3), T(-4), T(5), T(-6)};

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_system_with_solution(A.data(), b.data(), x_true.data(), rows, cols);

    // Parallel solve
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    T err = relative_error(x_true.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  }

  SECTION("noisy system") {
    constexpr std::size_t rows = 10000;
    constexpr std::size_t cols = 6;

    std::array<T, cols> x_true = {T(1), T(-2), T(3), T(-4), T(5), T(-6)};

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_system_with_solution(A.data(), b.data(), x_true.data(), rows, cols);

    // Add noise to b
    std::mt19937 gen(123);
    std::normal_distribution<T> noise(T(0), T(0.01));
    for (std::size_t i = 0; i < rows; ++i) {
      b[i] += noise(gen);
    }

    // Parallel solve
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    // With noise, we can't recover exactly but should be close
    T err = relative_error(x_true.data(), x_par.data(), cols);
    REQUIRE(err < T(0.1)); // Allow larger tolerance due to noise
  }
}

// =============================================================================
// Tests with different column counts
// =============================================================================

TEMPLATE_TEST_CASE("parallel_various_cols", "[core][linalg][parallel]",
                   float, double) {
  using T = TestType;
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-10);

  auto test_cols = [&](std::size_t cols) {
    constexpr std::size_t rows = 5000;

    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_random_system(A.data(), b.data(), rows, cols);

    // Sequential solve
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A.begin(), A.end(), A_copy.begin());
    tf::buffer<T> work;
    work.allocate(tf::linalg::least_squares_workspace_size<T>(rows, cols));
    tf::buffer<T> x_seq;
    x_seq.allocate(cols);
    tf::linalg::solve_least_squares(A_copy.data(), b.data(), x_seq.data(), rows,
                                    cols, work.data());

    // Parallel solve
    tf::buffer<T> x_par;
    x_par.allocate(cols);
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols);

    T err = relative_error(x_seq.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  };

  SECTION("3 columns") { test_cols(3); }
  SECTION("6 columns") { test_cols(6); }
  SECTION("10 columns") { test_cols(10); }
  SECTION("20 columns") { test_cols(20); }
}

// =============================================================================
// Test reusable state
// =============================================================================

TEMPLATE_TEST_CASE("parallel_reusable_state", "[core][linalg][parallel]",
                   float, double) {
  using T = TestType;
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-10);

  constexpr std::size_t rows = 5000;
  constexpr std::size_t cols = 6;

  tf::linalg::parallel_least_squares_state<T> state;

  // Solve multiple systems reusing state
  for (int iter = 0; iter < 5; ++iter) {
    tf::buffer<T> A, b;
    A.allocate(rows * cols);
    b.allocate(rows);
    generate_random_system(A.data(), b.data(), rows, cols, 42 + iter);

    // Sequential solve
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A.begin(), A.end(), A_copy.begin());
    tf::buffer<T> work;
    work.allocate(tf::linalg::least_squares_workspace_size<T>(rows, cols));
    std::array<T, cols> x_seq{};
    tf::linalg::solve_least_squares(A_copy.data(), b.data(), x_seq.data(), rows,
                                    cols, work.data());

    // Parallel solve with reused state
    std::array<T, cols> x_par{};
    tf::linalg::solve_least_squares_parallel(A.data(), b.data(), x_par.data(),
                                             rows, cols, state);

    T err = relative_error(x_seq.data(), x_par.data(), cols);
    REQUIRE(err < tol);
  }
}
