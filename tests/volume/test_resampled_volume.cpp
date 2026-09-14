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
#include <trueform/trueform.hpp>
#include <trueform/volume.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

using Catch::Matchers::WithinAbs;

namespace {

auto resample_sphere_field(tf::point<float, 3> center, float radius) {
  return tf::make_sphere_sdf<float>(
      {24, 24, 24}, tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-5.75f, -5.75f, -5.75f}, center, radius);
}

auto resample_sphere_at(const tf::point<float, 3> &p,
                        const tf::point<float, 3> &center, float radius)
    -> float {
  return (p - center).length() - radius;
}

// The sentinel a posed resample writes past a field's own domain: strictly
// above every sample, by a margin of one (largest) grid step.
auto resample_far_outside(const tf::volume_buffer<float> &field) -> float {
  float max_sample = field.samples_buffer()[0];
  for (const auto &s : field.samples_buffer())
    max_sample = std::max(max_sample, s);
  float max_spacing = field.spacing()[0];
  for (std::size_t d = 1; d < 3; ++d)
    max_spacing = std::max(max_spacing, field.spacing()[d]);
  return max_sample + std::max(max_spacing, 1.f);
}

// Is a local-space point inside the sphere-field box, by `margin`? A positive
// margin asks for strictly inside, a negative one for strictly outside.
auto resample_box_holds(const tf::point<float, 3> &local, float margin)
    -> bool {
  for (int d = 0; d < 3; ++d)
    if (local[d] < -5.75f + margin || local[d] > 5.75f - margin)
      return false;
  return true;
}

} // namespace

TEST_CASE("an identity regrid reproduces the field exactly",
          "[volume][resample]") {
  auto field = resample_sphere_field({0.f, 0.f, 0.f}, 3.f);
  auto same = tf::make_resampled_volume(field.volume(), field.dims(),
                                        field.spacing(), field.origin());
  REQUIRE(same.voxel_count() == field.voxel_count());
  for (std::size_t i = 0; i < field.voxel_count(); ++i)
    REQUIRE(same.samples_buffer()[i] == field.samples_buffer()[i]);
}

TEST_CASE("a downsampled field tracks the analytic one", "[volume][resample]") {
  const auto center = tf::point<float, 3>{0.f, 0.f, 0.f};
  auto field = resample_sphere_field(center, 3.f);
  auto coarse = tf::make_resampled_volume(
      field.volume(), {12, 12, 12}, tf::point<float, 3>{0.9f, 0.9f, 0.9f},
      tf::point<float, 3>{-4.95f, -4.95f, -4.95f});
  auto vol = coarse.volume();
  for (int z = 0; z < 12; ++z)
    for (int y = 0; y < 12; ++y)
      for (int x = 0; x < 12; ++x) {
        const auto p = vol.point_at(x, y, z);
        REQUIRE_THAT(vol(x, y, z),
                     WithinAbs(resample_sphere_at(p, center, 3.f), 0.08f));
      }
}

TEST_CASE("a posed field resamples through its pose", "[volume][resample]") {
  const auto center = tf::point<float, 3>{1.f, 0.f, 0.f};
  auto field = resample_sphere_field(center, 2.5f);
  auto pose = tf::make_rotation(tf::deg(30.f), tf::axis<2>);
  pose(1, 3) = 4.f;
  auto frame = tf::make_frame(pose);
  auto posed = field.volume() | tf::tag(frame);

  auto world = tf::make_resampled_volume(
      posed, {20, 20, 20}, tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-4.75f, -0.75f, -4.75f});
  auto vol = world.volume();
  const auto inverse = frame.inverse_transformation();
  for (int z = 0; z < 20; ++z)
    for (int y = 0; y < 20; ++y)
      for (int x = 0; x < 20; ++x) {
        const auto local = tf::transformed(vol.point_at(x, y, z), inverse);
        bool inside = true;
        for (int d = 0; d < 3; ++d)
          inside = inside && local[d] > -5.75f && local[d] < 5.75f;
        // The SDF is a cone at its own center, where linear interpolation
        // legitimately undershoots; the smooth field is the assertion.
        if (!inside || (local - center).length() < 1.f)
          continue;
        REQUIRE_THAT(vol(x, y, z),
                     WithinAbs(resample_sphere_at(local, center, 2.5f), 0.08f));
      }
}

TEST_CASE("a reflecting pose resamples through its inverse too",
          "[volume][resample]") {
  const auto center = tf::point<float, 3>{1.5f, 0.f, 0.f};
  auto field = resample_sphere_field(center, 2.f);
  auto mirror = tf::make_identity_transformation<float, 3>();
  mirror(0, 0) = -1.f;
  auto frame = tf::make_frame(mirror);

  auto world = tf::make_resampled_volume(
      field.volume() | tf::tag(frame), {16, 16, 16},
      tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-5.25f, -3.75f, -3.75f});
  auto vol = world.volume();
  // The mirrored sphere sits at world x = -1.5.
  const auto p = vol.point_at(8, 8, 8);
  const auto expected = resample_sphere_at(
      tf::point<float, 3>{-p[0], p[1], p[2]}, center, 2.f);
  REQUIRE_THAT(vol(8, 8, 8), WithinAbs(expected, 0.08f));
}

TEST_CASE("a node outside the posed domain takes the far-outside sentinel",
          "[volume][resample]") {
  auto field = resample_sphere_field({0.f, 0.f, 0.f}, 2.f);
  float field_max = field.samples_buffer()[0];
  for (const auto &s : field.samples_buffer())
    field_max = std::max(field_max, s);
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{100.f, 0.f, 0.f}));

  auto world = tf::make_resampled_volume(
      field.volume() | tf::tag(frame), {4, 4, 4},
      tf::point<float, 3>{1.f, 1.f, 1.f}, tf::point<float, 3>{0.f, 0.f, 0.f});
  for (const auto &s : world.samples_buffer())
    REQUIRE(s > field_max);
}

TEST_CASE("posed operands boolean through the shared world grid",
          "[volume][resample]") {
  const auto ca = tf::point<float, 3>{0.f, 0.f, 0.f};
  const auto cb = tf::point<float, 3>{1.f, 0.f, 0.f};
  auto a = resample_sphere_field(ca, 2.5f);
  auto b = resample_sphere_field(cb, 2.5f);
  auto frame_a = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{-1.f, 0.f, 0.f}));
  auto frame_b = tf::make_frame(tf::make_rotation(tf::deg(45.f), tf::axis<2>));

  auto [merged, pose] =
      tf::make_boolean(a.volume() | tf::tag(frame_a),
                       b.volume() | tf::tag(frame_b),
                       tf::volume_boolean_op::union_);
  const auto identity = tf::make_identity_transformation<float, 3>();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      REQUIRE(pose(i, j) == identity(i, j));

  const auto ia = frame_a.inverse_transformation();
  const auto ib = frame_b.inverse_transformation();
  const float sentinel =
      std::min(resample_far_outside(a), resample_far_outside(b));
  auto vol = merged.volume();
  const auto d = vol.dims();
  std::size_t n_inside = 0, n_outside = 0;
  for (int z = 0; z < d[2]; ++z)
    for (int y = 0; y < d[1]; ++y)
      for (int x = 0; x < d[0]; ++x) {
        const auto p = vol.point_at(x, y, z);
        const auto la = tf::transformed(p, ia);
        const auto lb = tf::transformed(p, ib);
        // Past both posed domains — the region the general path exists for —
        // both operands read their own sentinel and the union takes the
        // smaller, verbatim.
        if (!resample_box_holds(la, -0.01f) && !resample_box_holds(lb, -0.01f)) {
          ++n_outside;
          REQUIRE(vol(x, y, z) == sentinel);
          continue;
        }
        if (!resample_box_holds(la, 0.75f) || !resample_box_holds(lb, 0.75f))
          continue;
        // The SDF is a cone at its own center, where linear interpolation
        // legitimately undershoots; the smooth field is the assertion.
        if ((la - ca).length() < 1.f || (lb - cb).length() < 1.f)
          continue;
        ++n_inside;
        const float expected = std::min(resample_sphere_at(la, ca, 2.5f),
                                        resample_sphere_at(lb, cb, 2.5f));
        REQUIRE_THAT(vol(x, y, z), WithinAbs(expected, 0.1f));
      }
  REQUIRE(n_inside > 0);
  REQUIRE(n_outside > 0);
}

TEST_CASE("a posed operand booleans against an unposed one",
          "[volume][resample]") {
  const auto ca = tf::point<float, 3>{0.f, 0.f, 0.f};
  const auto cb = tf::point<float, 3>{1.f, 0.f, 0.f};
  auto a = resample_sphere_field(ca, 2.5f);
  auto b = resample_sphere_field(cb, 2.f);
  auto frame_a = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{2.f, 0.f, 0.f}));

  auto [merged, pose] =
      tf::make_boolean(a.volume() | tf::tag(frame_a), b.volume(),
                       tf::volume_boolean_op::union_);
  const auto identity = tf::make_identity_transformation<float, 3>();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      REQUIRE(pose(i, j) == identity(i, j));

  // Inside both boxes the two operands answer with their own fields, which
  // is where the analytic union is the expectation; past either box the
  // sentinel answers and there is nothing analytic to compare against.
  const auto ia = frame_a.inverse_transformation();
  auto vol = merged.volume();
  const auto d = vol.dims();
  std::size_t n_asserted = 0;
  for (int z = 0; z < d[2]; ++z)
    for (int y = 0; y < d[1]; ++y)
      for (int x = 0; x < d[0]; ++x) {
        const auto p = vol.point_at(x, y, z);
        const auto la = tf::transformed(p, ia);
        if (!resample_box_holds(la, 0.75f) || !resample_box_holds(p, 0.75f))
          continue;
        if ((la - ca).length() < 1.f || (p - cb).length() < 1.f)
          continue;
        ++n_asserted;
        const float expected = std::min(resample_sphere_at(la, ca, 2.5f),
                                        resample_sphere_at(p, cb, 2.f));
        REQUIRE_THAT(vol(x, y, z), WithinAbs(expected, 0.1f));
      }
  REQUIRE(n_asserted > 0);
}

TEST_CASE("a solid reaching its own box does not continue past it",
          "[volume][resample]") {
  // The half-space x < 0, sampled on a box its solid reaches: continuing
  // this field outward would grow the solid to wherever the shared grid
  // reaches. Out of the box is outside the solid, so it does not.
  tf::volume_buffer<float> slab({12, 12, 12},
                                tf::point<float, 3>{0.5f, 0.5f, 0.5f},
                                tf::point<float, 3>{-2.75f, -2.75f, -2.75f});
  {
    auto field = slab.volume();
    for (int z = 0; z < 12; ++z)
      for (int y = 0; y < 12; ++y)
        for (int x = 0; x < 12; ++x)
          field(x, y, z) = field.point_at(x, y, z)[0];
  }
  const auto cb = tf::point<float, 3>{-6.f, 0.f, 0.f};
  auto ball = tf::make_sphere_sdf<float>(
      {12, 12, 12}, tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-8.75f, -2.75f, -2.75f}, cb, 1.f);

  auto merged = tf::make_boolean(slab.volume(), ball.volume(),
                                 tf::volume_boolean_op::union_);
  auto vol = merged.volume();
  const auto d = vol.dims();
  std::size_t n_asserted = 0;
  for (int z = 0; z < d[2]; ++z)
    for (int y = 0; y < d[1]; ++y)
      for (int x = 0; x < d[0]; ++x) {
        const auto p = vol.point_at(x, y, z);
        // Past the slab's own box, and outside the ball's solid: the union
        // holds nothing here.
        if (p[0] > -3.f || (p - cb).length() < 1.5f)
          continue;
        ++n_asserted;
        REQUIRE(vol(x, y, z) > 0.f);
      }
  REQUIRE(n_asserted > 0);
}

TEST_CASE("an empty operand answers by the algebra", "[volume][resample]") {
  auto a = resample_sphere_field({0.f, 0.f, 0.f}, 2.f);
  tf::volume_buffer<float> empty({0, 0, 0},
                                 tf::point<float, 3>{0.5f, 0.5f, 0.5f},
                                 tf::point<float, 3>{0.f, 0.f, 0.f});

  auto a_union = tf::make_boolean(a.volume(), empty.volume(),
                                  tf::volume_boolean_op::union_);
  REQUIRE(a_union.voxel_count() == a.voxel_count());
  for (std::size_t i = 0; i < a.voxel_count(); ++i)
    REQUIRE(a_union.samples_buffer()[i] == a.samples_buffer()[i]);

  auto a_difference = tf::make_boolean(a.volume(), empty.volume(),
                                       tf::volume_boolean_op::difference);
  REQUIRE(a_difference.voxel_count() == a.voxel_count());

  REQUIRE(tf::make_boolean(a.volume(), empty.volume(),
                           tf::volume_boolean_op::intersection)
              .voxel_count() == 0);

  auto b_union = tf::make_boolean(empty.volume(), a.volume(),
                                  tf::volume_boolean_op::union_);
  REQUIRE(b_union.voxel_count() == a.voxel_count());
  REQUIRE(tf::make_boolean(empty.volume(), a.volume(),
                           tf::volume_boolean_op::intersection)
              .voxel_count() == 0);
  REQUIRE(tf::make_boolean(empty.volume(), a.volume(),
                           tf::volume_boolean_op::difference)
              .voxel_count() == 0);
}

TEST_CASE("an empty posed operand answers standing where the other stood",
          "[volume][resample]") {
  auto b = resample_sphere_field({0.f, 0.f, 0.f}, 2.f);
  tf::volume_buffer<float> empty({0, 0, 0},
                                 tf::point<float, 3>{0.5f, 0.5f, 0.5f},
                                 tf::point<float, 3>{0.f, 0.f, 0.f});
  auto frame_a = tf::make_frame(tf::make_rotation(tf::deg(20.f), tf::axis<2>));
  auto frame_b = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{0.f, 3.f, 0.f}));

  auto [merged, pose] =
      tf::make_boolean(empty.volume() | tf::tag(frame_a),
                       b.volume() | tf::tag(frame_b),
                       tf::volume_boolean_op::union_);
  REQUIRE(merged.voxel_count() == b.voxel_count());
  const auto &t = frame_b.transformation();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      REQUIRE(pose(i, j) == t(i, j));

  auto [carved, carved_pose] =
      tf::make_boolean(empty.volume() | tf::tag(frame_a),
                       b.volume() | tf::tag(frame_b),
                       tf::volume_boolean_op::difference);
  REQUIRE(carved.voxel_count() == 0);
  const auto &ta = frame_a.transformation();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      REQUIRE(carved_pose(i, j) == ta(i, j));
}

TEST_CASE("a posed 2D boolean shares the world grid too",
          "[volume][resample]") {
  const auto ca = tf::point<float, 2>{0.f, 0.f};
  const auto cb = tf::point<float, 2>{1.f, 0.f};
  auto a = tf::make_sphere_sdf<float>(
      std::array<int, 2>{24, 24}, tf::point<float, 2>{0.5f, 0.5f},
      tf::point<float, 2>{-5.75f, -5.75f}, ca, 2.5f);
  auto b = tf::make_sphere_sdf<float>(
      std::array<int, 2>{24, 24}, tf::point<float, 2>{0.5f, 0.5f},
      tf::point<float, 2>{-5.75f, -5.75f}, cb, 2.f);
  auto rotation = tf::make_identity_transformation<float, 2>();
  const float c = std::cos(0.5f), s = std::sin(0.5f);
  rotation(0, 0) = c;
  rotation(0, 1) = -s;
  rotation(1, 0) = s;
  rotation(1, 1) = c;
  auto frame_a = tf::make_frame(rotation);

  auto [merged, pose] =
      tf::make_boolean(a.volume() | tf::tag(frame_a), b.volume(),
                       tf::volume_boolean_op::union_);
  const auto identity = tf::make_identity_transformation<float, 2>();
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      REQUIRE(pose(i, j) == identity(i, j));

  const auto ia = frame_a.inverse_transformation();
  auto vol = merged.volume();
  const auto d = vol.dims();
  std::size_t n_asserted = 0;
  for (int y = 0; y < d[1]; ++y)
    for (int x = 0; x < d[0]; ++x) {
      const auto p = vol.point_at(x, y);
      const auto la = tf::transformed(p, ia);
      bool inside = true;
      for (int k = 0; k < 2; ++k)
        inside = inside && la[k] > -5.f && la[k] < 5.f && p[k] > -5.f &&
                 p[k] < 5.f;
      if (!inside || (la - ca).length() < 1.f || (p - cb).length() < 1.f)
        continue;
      ++n_asserted;
      const float expected = std::min((la - ca).length() - 2.5f,
                                      (p - cb).length() - 2.f);
      REQUIRE_THAT(vol(x, y), WithinAbs(expected, 0.1f));
    }
  REQUIRE(n_asserted > 0);
}

TEST_CASE("matched poses combine samplewise and keep their pose",
          "[volume][resample]") {
  auto a = resample_sphere_field({0.f, 0.f, 0.f}, 2.5f);
  auto b = resample_sphere_field({1.f, 0.f, 0.f}, 2.f);
  auto frame = tf::make_frame(tf::make_rotation(tf::deg(30.f), tf::axis<2>));

  auto flat = tf::make_boolean(a.volume(), b.volume(),
                               tf::volume_boolean_op::intersection);
  auto [posed, pose] =
      tf::make_boolean(a.volume() | tf::tag(frame),
                       b.volume() | tf::tag(frame),
                       tf::volume_boolean_op::intersection);
  REQUIRE(posed.voxel_count() == flat.voxel_count());
  for (std::size_t i = 0; i < flat.voxel_count(); ++i)
    REQUIRE(posed.samples_buffer()[i] == flat.samples_buffer()[i]);
  const auto &t = frame.transformation();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      REQUIRE(pose(i, j) == t(i, j));
}

TEST_CASE("the unposed combine keeps its shape", "[volume][resample]") {
  auto a = resample_sphere_field({0.f, 0.f, 0.f}, 2.f);
  auto out = tf::make_boolean(a.volume(), a.volume(),
                              tf::volume_boolean_op::union_);
  static_assert(std::is_same_v<decltype(out), tf::volume_buffer<float>>);
  REQUIRE(out.voxel_count() == a.voxel_count());
}
