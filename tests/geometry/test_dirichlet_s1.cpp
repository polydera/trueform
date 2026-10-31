/**
 * @file test_dirichlet_s1.cpp
 * @brief The Dirichlet field on the S1 patch, against its references
 *
 * Tests for:
 * - tf::make_dirichlet_field on a system whose Jacobi-scaled condition is
 *   about 1e12 at order 2
 * - tf::analyze_cholesky's fill on it, within the bound a minimum degree
 *   order meets there (346,828 and 1,292,750 factor entries at orders 1 and
 *   2) and far below the natural order's 2.13M and 4.53M
 *
 * The fixture is a 33 mm geodesic patch of S1 under curvature-derived corner
 * weights: 715 rim points held at 0 and 870 seed points held at 1. The
 * references are a direct solve of the same system at both orders and an
 * unrefined direct solve at order 2. The field stands 8.8e-12 from the order-1
 * reference under its 1e-9 bound, 7.0e-6 from the order-2 one under 1e-5 and
 * 1.2e-5 from the unrefined one under 1e-4, every solve product fused.
 *
 * The patch is local data, read from TF_TEST_DATA_DIR under dirichlet/, so
 * the cases are hidden: run them by the [local] tag, and a configure that
 * names no directory holding the patch skips them.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/laplacian/assemble_laplacian_system.hpp>
#include <trueform/geometry/laplacian/make_laplacian_form_pattern.hpp>
#include <trueform/geometry/make_dirichlet_field.hpp>
#include <trueform/linalg/analyze_cholesky.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <vector>

namespace {

struct s1_fixture {
  tf::polygons_buffer<int, float, 3, 3> mesh;
  std::vector<double> weights;
  std::vector<int> fixed;
  std::vector<double> values;
};

auto s1_path(const char *name) -> std::string {
  return std::string(TF_TEST_DATA_DIR) + "/dirichlet/" + name;
}

template <typename T>
auto s1_read(std::FILE *file, std::size_t count) -> std::vector<T> {
  std::vector<T> out(count);
  REQUIRE(std::fread(out.data(), sizeof(T), count, file) == count);
  return out;
}

auto s1_fixture_of() -> s1_fixture {
  std::ifstream probe(s1_path("s1_patch.bin"));
  if (!probe)
    SKIP("dirichlet/s1_patch.bin is not in TF_TEST_DATA_DIR");
  std::FILE *file = std::fopen(s1_path("s1_patch.bin").c_str(), "rb");
  REQUIRE(file != nullptr);
  const auto header = s1_read<std::uint64_t>(file, 4);
  const auto points = s1_read<float>(file, 3 * header[0]);
  const auto faces = s1_read<std::int32_t>(file, 3 * header[1]);
  s1_fixture out;
  out.weights = s1_read<double>(file, 3 * header[1]);
  const auto fixed = s1_read<std::int32_t>(file, header[2]);
  std::fclose(file);
  out.mesh.points_buffer().allocate(header[0]);
  for (std::size_t k = 0; k < header[0]; ++k)
    for (std::size_t a = 0; a < 3; ++a)
      out.mesh.points()[k][a] = points[3 * k + a];
  out.mesh.faces_buffer().allocate(header[1]);
  for (std::size_t f = 0; f < header[1]; ++f)
    for (std::size_t c = 0; c < 3; ++c)
      out.mesh.faces()[f][c] = faces[3 * f + c];
  for (std::size_t k = 0; k < fixed.size(); ++k) {
    out.fixed.push_back(fixed[k]);
    out.values.push_back(k < header[3] ? 0.0 : 1.0);
  }
  return out;
}

auto s1_reference(const char *name, std::size_t n) -> std::vector<double> {
  std::FILE *file = std::fopen(s1_path(name).c_str(), "rb");
  REQUIRE(file != nullptr);
  auto out = s1_read<double>(file, n);
  std::fclose(file);
  return out;
}

/**
 * @brief The largest difference from a reference, and the points the
 *        reference's 0.5 threshold classifies otherwise once both are clamped
 *        to `[0.01, 0.99]`, the range the field is read in, a point within
 *        `1e-6` of 0.5 excepted.
 */
struct s1_agreement {
  double largest = 0.0;
  std::size_t disagreements = 0;
};

auto s1_agreement_of(const tf::buffer<double> &field,
                     const std::vector<double> &reference) -> s1_agreement {
  const auto clamp = [](double v) { return std::min(0.99, std::max(0.01, v)); };
  s1_agreement out;
  for (std::size_t k = 0; k < reference.size(); ++k) {
    out.largest = std::max(out.largest, std::abs(field[k] - reference[k]));
    if (std::abs(reference[k] - 0.5) > 1e-6 &&
        (clamp(field[k]) > 0.5) != (clamp(reference[k]) > 0.5))
      ++out.disagreements;
  }
  return out;
}

} // namespace

TEST_CASE("dirichlet_field: the S1 harmonic field answers the direct solve",
          "[geometry][dirichlet][s1][local][.]") {
  const auto s1 = s1_fixture_of();
  const std::size_t n = s1.mesh.points().size();
  REQUIRE(n == 18644);
  const auto result = tf::make_dirichlet_field<double>(
      s1.mesh.polygons(), s1.weights, s1.fixed, s1.values,
      tf::dirichlet_order::harmonic);
  REQUIRE(result.status == tf::dirichlet_field_status::solved);
  const auto direct =
      s1_agreement_of(result.field, s1_reference("s1_order1_reference.f64", n));
  REQUIRE(direct.largest <= 1e-9);
  REQUIRE(direct.disagreements == 0);
}

TEST_CASE("dirichlet_field: the S1 biharmonic field answers the direct solves",
          "[geometry][dirichlet][s1][local][.]") {
  const auto s1 = s1_fixture_of();
  const std::size_t n = s1.mesh.points().size();
  const auto result = tf::make_dirichlet_field<double>(
      s1.mesh.polygons(), s1.weights, s1.fixed, s1.values,
      tf::dirichlet_order::biharmonic);
  REQUIRE(result.status == tf::dirichlet_field_status::solved);
  for (std::size_t k = 0; k < s1.fixed.size(); ++k)
    REQUIRE(result.field[std::size_t(s1.fixed[k])] == s1.values[k]);
  const auto direct =
      s1_agreement_of(result.field, s1_reference("s1_order2_reference.f64", n));
  REQUIRE(direct.largest <= 1e-5);
  REQUIRE(direct.disagreements == 0);
  // The unrefined answer stands off the exact field by the system's own
  // kappa u, about 2e-4 here; the classification must not move.
  const auto unrefined = s1_agreement_of(
      result.field, s1_reference("s1_order2_unrefined_reference.f64", n));
  REQUIRE(unrefined.largest <= 1e-4);
  REQUIRE(unrefined.disagreements == 0);
}

TEST_CASE("dirichlet_field: the S1 factor fills within its minimum degree bound",
          "[geometry][dirichlet][s1][local][.]") {
  const auto s1 = s1_fixture_of();
  const std::size_t n = s1.mesh.points().size();
  tf::face_membership<int> membership;
  membership.build(s1.mesh.polygons());
  tf::geometry::laplacian_assembly_scratch assembly;
  tf::geometry::laplacian_system<int> system;
  REQUIRE(tf::geometry::assemble_laplacian_system(
      s1.mesh.polygons() | tf::tag(membership),
      [&s1](std::size_t t, std::size_t slot, const tf::point<double, 3>(&)[3],
            double) { return s1.weights[3 * t + (slot + 2) % 3]; },
      assembly, system));
  tf::buffer<int> free_of;
  free_of.allocate_and_initialize(n, 0);
  for (auto id : s1.fixed)
    free_of[std::size_t(id)] = -1;
  tf::buffer<int> free_ids;
  for (std::size_t k = 0; k < n; ++k)
    if (free_of[k] == 0) {
      free_of[k] = int(free_ids.size());
      free_ids.push_back(int(k));
    }
  for (int order : {1, 2}) {
    tf::geometry::laplacian_form<int> form;
    tf::geometry::make_laplacian_form_pattern(system, free_ids, free_of,
                                              order == 2, form);
    tf::cholesky_scratch<int> scratch;
    tf::cholesky_pattern<int> pattern;
    REQUIRE(tf::analyze_cholesky(form.free, scratch, pattern));
    std::size_t entries = 0;
    for (std::size_t k = 0; k < free_ids.size(); ++k)
      entries += std::size_t(scratch.counts[k]);
    REQUIRE(entries <=
            (order == 1 ? std::size_t(346828) : std::size_t(1292750)));
  }
}
