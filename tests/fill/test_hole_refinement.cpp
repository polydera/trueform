/**
 * @file test_hole_refinement.cpp
 * @brief Tests for the table-tier refinement driver
 *
 * Tests for:
 * - tf::fill::refine_hole_patch
 * - tf::fill::hole_patch_quality
 * - tf::fill::build_hole_patch / tf::fill::emit_hole_patch
 * - tf::fill::split_hole_patch_edge
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/vector.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/fill/holes/hole_patch.hpp>
#include <trueform/fill/holes/refine_hole_patch.hpp>
#include <trueform/fill/hole_split.hpp>
#include <trueform/topology/face_membership.hpp>
#include "fill_generators.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using refine_index = int;

/**
 * @brief A prepared cycle naming vertices 0..n-1 at the given positions, none
 *        of them a split, which is what a white-box patch is built over.
 */
template <typename Int, typename Real>
auto refine_rim(const std::vector<tf::point<Real, 3>> &positions)
    -> tf::fill::hole_rim<refine_index, Int, Real> {
  tf::fill::hole_rim<refine_index, Int, Real> rim;
  const refine_index n = refine_index(positions.size());
  for (refine_index k = 0; k < n; ++k) {
    rim.positions.push_back(positions[std::size_t(k)]);
    rim.corners.push_back(k);
    const refine_index other = refine_index(k + 1 == n ? 0 : k + 1);
    rim.edges.push_back({refine_index(-1), std::min(k, other),
                         std::max(k, other),
                         tf::vector<double, 3>{0.0, 0.0, 1.0}, std::uint8_t(0),
                         std::uint8_t(tf::hole_split_scale), true});
  }
  return rim;
}

/**
 * @brief What one white-box refinement produced.
 */
template <typename Real> struct refine_outcome {
  tf::hole_refine_status status;
  tf::buffer<std::array<refine_index, 3>> triangles;
  tf::buffer<tf::point<Real, 3>> minted;
};

/**
 * @brief Refine the stated patch of the stated rim, against a cone mesh whose
 *        own faces carry the rim edges and nothing else.
 */
template <typename Int, typename Real>
auto refine_case(const std::vector<tf::point<Real, 3>> &positions,
                 const tf::point<Real, 3> &apex,
                 const std::vector<std::array<refine_index, 3>> &patch,
                 double floor, std::size_t mints, std::size_t flips)
    -> refine_outcome<Real> {
  const auto mesh = hole_cone_mesh<refine_index, Real>(positions, apex);
  tf::face_membership<refine_index> membership;
  membership.build(mesh.polygons());
  const auto rim = refine_rim<Int, Real>(positions);

  refine_outcome<Real> outcome{tf::hole_refine_status::not_attempted, {}, {}};
  for (const auto &triangle : patch)
    outcome.triangles.push_back(triangle);

  tf::fill::hole_patch_scratch<refine_index> shape;
  tf::fill::hole_patch<refine_index, Real> shaped;
  tf::fill::build_hole_patch(rim, outcome.triangles, shape, shaped);
  tf::fill::hole_refine_scratch<refine_index> scratch;
  outcome.status = tf::fill::refine_hole_patch(
      mesh.polygons(), membership, refine_index(mesh.points().size()), floor,
      mints, flips, outcome.minted, scratch, shaped);
  tf::fill::emit_hole_patch(shaped, shape, outcome.triangles);
  return outcome;
}

auto refine_holds_chord(const tf::buffer<std::array<refine_index, 3>> &patch,
                        refine_index a, refine_index b) -> bool {
  for (const auto &triangle : patch)
    for (int e = 0; e < 3; ++e) {
      const refine_index x = triangle[std::size_t(e)];
      const refine_index y = triangle[std::size_t((e + 1) % 3)];
      if ((x == a && y == b) || (x == b && y == a))
        return true;
    }
  return false;
}

/// The rhombus whose one chord is the longest edge of both its triangles, so
/// halving it lifts them both.
const std::vector<tf::point<float, 3>> refine_rhombus{{-4.0f, 0.0f, 0.0f},
                                                      {0.0f, -1.0f, 0.0f},
                                                      {4.0f, 0.0f, 0.0f},
                                                      {0.0f, 1.0f, 0.0f}};
const std::vector<std::array<refine_index, 3>> refine_rhombus_patch{{0, 1, 2},
                                                                    {0, 2, 3}};

/// Two congruent slivers meeting at one vertex: the split of either one's
/// chord leaves the other standing in its own support, tied at the worst
/// quality the patch holds.
const std::vector<tf::point<float, 3>> refine_twins{
    {-8.0f, 0.0f, 0.0f}, {-4.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
    {4.0f, -1.0f, 0.0f}, {8.0f, 0.0f, 0.0f},   {0.0f, 3.0f, 0.0f}};
const std::vector<std::array<refine_index, 3>> refine_twins_patch{
    {0, 1, 2}, {0, 2, 5}, {2, 3, 4}, {2, 4, 5}};

} // namespace

TEST_CASE("fill: a patch below the floor refines to it on midpoint splits",
          "[fill][refine]") {
  const auto outcome = refine_case<tf::exact::int32, float>(
      refine_rhombus, {0.0f, 0.0f, -4.0f}, refine_rhombus_patch, 0.25, 8, 16);
  REQUIRE(outcome.status == tf::hole_refine_status::floor_met);
  REQUIRE(outcome.minted.size() == 1);
  REQUIRE(outcome.triangles.size() == 4);
  REQUIRE(outcome.minted[0][0] == 0.0f);
  REQUIRE(outcome.minted[0][1] == 0.0f);
  REQUIRE(outcome.minted[0][2] == 0.0f);
}

TEST_CASE("fill: a patch the rim itself holds below the floor stalls",
          "[fill][refine]") {
  const std::vector<tf::point<float, 3>> sliver{
      {0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}, {5.0f, 0.125f, 0.0f}};
  const auto outcome = refine_case<tf::exact::int32, float>(
      sliver, {5.0f, 0.0f, -4.0f}, {{0, 1, 2}}, 0.5, 4, 8);
  REQUIRE(outcome.status == tf::hole_refine_status::stalled);
  REQUIRE(outcome.triangles.size() == 1);
  REQUIRE(outcome.minted.size() == 0);
}

TEST_CASE("fill: a budget that removes every improving move states itself",
          "[fill][refine]") {
  const auto outcome = refine_case<tf::exact::int32, float>(
      refine_rhombus, {0.0f, 0.0f, -4.0f}, refine_rhombus_patch, 1.0, 0, 0);
  REQUIRE(outcome.status == tf::hole_refine_status::budget_exhausted);
  REQUIRE(outcome.triangles.size() == 2);
  REQUIRE(outcome.minted.size() == 0);
}

TEST_CASE("fill: one family's exhausted budget still offers the other",
          "[fill][refine]") {
  const auto outcome = refine_case<tf::exact::int32, float>(
      refine_rhombus, {0.0f, 0.0f, -4.0f}, refine_rhombus_patch, 1.0, 0, 16);
  REQUIRE(outcome.minted.size() == 0);
  REQUIRE(outcome.triangles.size() == 2);
  REQUIRE(refine_holds_chord(outcome.triangles, 1, 3));
  REQUIRE_FALSE(refine_holds_chord(outcome.triangles, 0, 2));
  REQUIRE(outcome.status == tf::hole_refine_status::stalled);
}

TEST_CASE("fill: a triangle the split does not replace blocks it from its own "
          "support",
          "[fill][refine]") {
  const auto outcome = refine_case<tf::exact::int32, float>(
      refine_twins, {0.0f, 0.0f, -6.0f}, refine_twins_patch, 0.5, 16, 32);
  REQUIRE(outcome.status == tf::hole_refine_status::stalled);
  REQUIRE(outcome.triangles.size() == 4);
  REQUIRE(outcome.minted.size() == 0);
}

TEST_CASE("fill: a triangle that states no quality reads the worst one",
          "[fill][refine]") {
  double stated = -1.0;
  REQUIRE_FALSE(tf::fill::hole_patch_quality({0.0, 0.0, 0.0},
                                             {1e200, 0.0, 0.0},
                                             {0.0, 1e200, 0.0}, stated));
  REQUIRE(tf::fill::hole_patch_quality({0.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
                                       {0.0, 4.0, 0.0}, stated));
  REQUIRE(stated > 0.0);

  std::vector<tf::point<double, 3>> wrecked;
  for (const auto &position : refine_twins)
    wrecked.push_back({double(position[0]), double(position[1]),
                       double(position[2])});
  wrecked[3] = tf::point<double, 3>{4.0, -1e200, 0.0};
  const auto outcome = refine_case<tf::exact::int64, double>(
      wrecked, {0.0, 0.0, -6.0}, refine_twins_patch, 0.1, 16, 32);
  REQUIRE(outcome.status == tf::hole_refine_status::stalled);
  REQUIRE(outcome.triangles.size() == 4);
  REQUIRE(outcome.minted.size() == 0);
}
