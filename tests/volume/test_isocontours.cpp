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
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/volume.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

// The shared sphere-SDF fixture: radius r, centered at the local origin, on a
// grid spanning [-2.5, 2.5]^3 with spacing h.
constexpr float kR = 1.5f;
constexpr float kH = 0.1f;

auto make_test_sphere() {
  return tf::make_sphere_sdf<float>({51, 51, 51},
                                    tf::point<float, 3>{kH, kH, kH},
                                    tf::point<float, 3>{-2.5f, -2.5f, -2.5f},
                                    tf::point<float, 3>{0.0f, 0.0f, 0.0f}, kR);
}

} // namespace

// A 2D volume is a volume, and its isocontours are the level set of its own
// bilinear field: the same operation the mesh entry names, one dimension down.
TEST_CASE("isocontours of a 2D volume close around its disc",
          "[volume][isocontours]") {
  const float r = 1.0f;
  const float h = 0.05f;
  const int n = 81;
  auto disc = tf::make_sphere_sdf<float>(
      std::array<int, 2>{n, n}, tf::point<float, 2>{h, h},
      tf::point<float, 2>{-2.0f, -2.0f}, tf::point<float, 2>{0.0f, 0.0f}, r);

  auto contours = tf::make_isocontours(disc.volume(), 0.0f);

  // one closed polyline, its first and last point the same crossing
  REQUIRE(contours.size() == 1u);
  const auto path = contours.paths()[0];
  REQUIRE(path.size() > 3u);
  REQUIRE(path[0] == path[path.size() - 1]);

  // Every point lies on the circle. The field is an exact SDF, so the
  // linear-interpolation error along an edge is O(h^2 / r), far below h.
  const auto &pts = contours.points_buffer();
  float max_dev = 0.0f;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto p = pts[i];
    const float d = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    max_dev = std::max(max_dev, std::abs(d - r));
  }
  REQUIRE(max_dev < 0.1f * h);

  // The polyline tiles the full circle: its length is the circumference
  // (chordal, so slightly below, never above).
  float total_len = 0.0f;
  for (std::size_t k = 1; k < path.size(); ++k) {
    const auto a = pts[std::size_t(path[k - 1])];
    const auto b = pts[std::size_t(path[k])];
    const float dx = b[0] - a[0];
    const float dy = b[1] - a[1];
    total_len += std::sqrt(dx * dx + dy * dy);
  }
  const float circumference = 2.0f * 3.14159265358979f * r;
  REQUIRE(total_len <= circumference * 1.001f);
  REQUIRE(total_len > circumference * 0.99f);
}

// The resampler behind the plane-contour entry, called on its own: the slice of
// a volume is a volume one dimension down. The plane is a geometric form, so it
// is spelled as one; the grid and its step are numbers and are written as the
// braced literals the entry takes.
TEST_CASE("a volume slice is a 2D volume of the plane it is given",
          "[volume][isocontours]") {
  auto vol = make_test_sphere();

  // The z = 0 plane through the sphere centre, on the volume's own grid: node
  // (i, j) lands exactly on sample (i, j, 25), so the resample is a read.
  auto slice = tf::make_volume_slice(
      vol.volume(), tf::point<float, 3>{-2.5f, -2.5f, 0.0f},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {51, 51}, {kH, kH});
  static_assert(tf::coordinate_dims_v<decltype(slice.volume())> == 2,
                "the slice of a volume is a volume one dimension down");
  REQUIRE(slice.dims() == std::array<int, 2>{51, 51});
  REQUIRE(slice.voxel_count() == 51u * 51u);
  for (int j = 0; j < 51; ++j)
    for (int i = 0; i < 51; ++i) {
      INFO("node " << i << " " << j);
      REQUIRE_THAT(slice(i, j), WithinAbs(vol(i, j, 25), 1e-5f));
    }

  // A plane wholly outside the volume's box takes the sentinel, which sits
  // above every field value so no isovalue the field carries is crossed there.
  auto outside = tf::make_volume_slice(
      vol.volume(), tf::point<float, 3>{10.0f, 10.0f, 10.0f},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {8, 8}, {kH, kH});
  REQUIRE(outside.voxel_count() == 64u);
  float field_max = vol.samples_buffer()[0];
  for (std::size_t i = 1; i < vol.samples_buffer().size(); ++i)
    field_max = std::max(field_max, vol.samples_buffer()[i]);
  for (std::size_t i = 0; i < outside.voxel_count(); ++i)
    REQUIRE(outside.samples_buffer()[i] > field_max);
}

TEST_CASE("isocontours of a sphere through the center", "[volume][isocontours]") {
  auto vol = make_test_sphere();

  // Slice plane z = 0 through the sphere center; the slice grid coincides
  // with the volume's z = 25 sample plane.
  const std::array<float, 1> isovalues{0.0f};
  auto contours = tf::make_isocontours(
      vol.volume(), tf::point<float, 3>{-2.5f, -2.5f, 0.0f},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {51, 51}, {kH, kH},
      tf::make_range(isovalues.data(), isovalues.data() + isovalues.size()));

  // the great circle is one closed polyline, lifted into the volume's frame
  REQUIRE(contours.size() == 1u);
  const auto path = contours.paths()[0];
  REQUIRE(path[0] == path[path.size() - 1]);

  const auto &pts = contours.points_buffer();
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto p = pts[i];
    REQUIRE_THAT(p[2], WithinAbs(0.0f, 1e-6));
    const float d = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    REQUIRE_THAT(d, WithinAbs(kR, 0.1f * kH));
  }
}

TEST_CASE("isocontours of an off-center slice", "[volume][isocontours]") {
  auto vol = make_test_sphere();

  // Slice at z = 0.55 (between grid planes, exercising the multilinear
  // resample): the section is a circle of radius sqrt(r^2 - z^2).
  const float zh = 0.55f;
  const std::array<float, 1> isovalues{0.0f};
  auto contours = tf::make_isocontours(
      vol.volume(), tf::point<float, 3>{-2.5f, -2.5f, zh},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {51, 51}, {kH, kH},
      tf::make_range(isovalues.data(), isovalues.data() + isovalues.size()));

  REQUIRE(contours.size() > 0u);

  const float expected = std::sqrt(kR * kR - zh * zh);
  const auto &pts = contours.points_buffer();
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto p = pts[i];
    REQUIRE_THAT(p[2], WithinAbs(zh, 1e-6));
    const float d = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    REQUIRE_THAT(d, WithinAbs(expected, 0.5f * kH));
  }
}

TEST_CASE("isocontours at several isovalues", "[volume][isocontours]") {
  auto vol = make_test_sphere();

  // SDF isovalue s contours the circle of radius r + s on the center plane:
  // three concentric rings at r/2, r and 3r/2 (= 2.25, inside the 2.5 domain).
  const std::array<float, 3> isovalues{-kR / 2.0f, 0.0f, kR / 2.0f};
  const std::array<float, 3> expected{kR / 2.0f, kR, 3.0f * kR / 2.0f};

  auto contours = tf::make_isocontours(
      vol.volume(), tf::point<float, 3>{-2.5f, -2.5f, 0.0f},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {51, 51}, {kH, kH},
      tf::make_range(isovalues.data(), isovalues.data() + isovalues.size()));

  // one closed ring per isovalue
  REQUIRE(contours.size() == 3u);

  // Every point sits on one of the three rings, and all three rings appear.
  std::array<std::size_t, 3> ring_points{0, 0, 0};
  const auto &pts = contours.points_buffer();
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto p = pts[i];
    const float d = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    std::size_t best = 0;
    float best_dev = std::abs(d - expected[0]);
    for (std::size_t k = 1; k < expected.size(); ++k) {
      const float dev = std::abs(d - expected[k]);
      if (dev < best_dev) {
        best_dev = dev;
        best = k;
      }
    }
    REQUIRE(best_dev < 0.1f * kH);
    ++ring_points[best];
  }
  for (std::size_t k = 0; k < ring_points.size(); ++k) {
    INFO("ring " << k << " (radius " << expected[k] << ")");
    REQUIRE(ring_points[k] > 0u);
  }
}

TEST_CASE("isocontours above the field maximum are empty",
          "[volume][isocontours]") {
  auto vol = make_test_sphere();

  // Far above every field value (max SDF sample is the corner, ~2.83).
  const std::array<float, 1> isovalues{10.0f};
  auto contours = tf::make_isocontours(
      vol.volume(), tf::point<float, 3>{-2.5f, -2.5f, 0.0f},
      tf::point<float, 3>{1.0f, 0.0f, 0.0f},
      tf::point<float, 3>{0.0f, 1.0f, 0.0f}, {51, 51}, {kH, kH},
      tf::make_range(isovalues.data(), isovalues.data() + isovalues.size()));

  REQUIRE(contours.size() == 0u);
  REQUIRE(contours.points_buffer().size() == 0u);
}
