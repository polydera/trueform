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
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

/// A sphere SDF rasterized as a CT would store it: int16 samples in
/// hundredths of a millimetre, on an anisotropic millimetre grid.
auto carrier_ct_sphere(std::array<int, 3> dims, tf::point<float, 3> spacing,
                       tf::point<float, 3> center, float radius)
    -> std::vector<std::int16_t> {
  std::vector<std::int16_t> samples(
      static_cast<std::size_t>(dims[0]) * static_cast<std::size_t>(dims[1]) *
      static_cast<std::size_t>(dims[2]));
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const float dx = static_cast<float>(x) * spacing[0] - center[0];
        const float dy = static_cast<float>(y) * spacing[1] - center[1];
        const float dz = static_cast<float>(z) * spacing[2] - center[2];
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
        samples[static_cast<std::size_t>(x) +
                static_cast<std::size_t>(dims[0]) *
                    (static_cast<std::size_t>(y) +
                     static_cast<std::size_t>(dims[1]) *
                         static_cast<std::size_t>(z))] =
            static_cast<std::int16_t>(std::lround(d * 100.0f));
      }
  return samples;
}

/// The same sphere in an UNSIGNED sample type, biased to the type's midpoint
/// because an unsigned field has no negative side to put the inside on.
template <typename T>
auto carrier_unsigned_sphere(std::array<int, 3> dims,
                             tf::point<float, 3> spacing,
                             tf::point<float, 3> center, float radius)
    -> std::vector<T> {
  const auto bias =
      static_cast<float>(std::numeric_limits<T>::max() / 2);
  std::vector<T> samples(static_cast<std::size_t>(dims[0]) *
                         static_cast<std::size_t>(dims[1]) *
                         static_cast<std::size_t>(dims[2]));
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const float dx = static_cast<float>(x) * spacing[0] - center[0];
        const float dy = static_cast<float>(y) * spacing[1] - center[1];
        const float dz = static_cast<float>(z) * spacing[2] - center[2];
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
        const float v = std::max(
            0.f, std::min(static_cast<float>(std::numeric_limits<T>::max()),
                          bias + d * 4.0f));
        samples[static_cast<std::size_t>(x) +
                static_cast<std::size_t>(dims[0]) *
                    (static_cast<std::size_t>(y) +
                     static_cast<std::size_t>(dims[1]) *
                         static_cast<std::size_t>(z))] =
            static_cast<T>(std::lround(v));
      }
  return samples;
}

/// An unsigned field consumes exactly like a signed one: two types on the
/// carrier, the level set emitted in the float coordinate type.
template <typename T>
auto check_unsigned_carrier_consumes(float radius) -> void {
  const std::array<int, 3> dims{16, 16, 16};
  const tf::point<float, 3> spacing{0.7f, 0.7f, 1.5f};
  const tf::point<float, 3> center{7.5f * 0.7f, 7.5f * 0.7f, 7.5f * 1.5f};
  auto samples = carrier_unsigned_sphere<T>(dims, spacing, center, radius);

  auto vol = tf::make_volume(samples.data(), dims, spacing,
                             tf::point<float, 3>{0.f, 0.f, 0.f});
  static_assert(std::is_same_v<typename decltype(vol)::sample_type, T>);
  static_assert(std::is_same_v<tf::coordinate_type<decltype(vol)>, float>);

  auto mesh = tf::make_isosurface(
      vol, static_cast<float>(std::numeric_limits<T>::max() / 2));
  REQUIRE(mesh.faces().size() > 0);
  static_assert(
      std::is_same_v<tf::coordinate_type<decltype(mesh.points())>, float>);

  for (auto q : mesh.points()) {
    const float dx = q[0] - center[0];
    const float dy = q[1] - center[1];
    const float dz = q[2] - center[2];
    const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
    REQUIRE(r > radius - 0.6f);
    REQUIRE(r < radius + 0.6f);
  }
}

} // namespace

TEST_CASE("an unsigned field stands on a float grid too",
          "[volume][carrier]") {
  check_unsigned_carrier_consumes<std::uint16_t>(4.0f);
  check_unsigned_carrier_consumes<std::uint8_t>(4.0f);
}

TEST_CASE("an int16 field stands on a float grid", "[volume][carrier]") {
  const std::array<int, 3> dims{16, 16, 16};
  const tf::point<float, 3> spacing{0.7f, 0.7f, 1.5f};
  const tf::point<float, 3> center{7.5f * 0.7f, 7.5f * 0.7f, 7.5f * 1.5f};
  const float radius = 4.0f;
  auto samples = carrier_ct_sphere(dims, spacing, center, radius);

  auto vol = tf::make_volume(samples.data(), dims, spacing,
                             tf::point<float, 3>{0.f, 0.f, 0.f});
  using volume_t = decltype(vol);
  static_assert(std::is_same_v<volume_t::sample_type, std::int16_t>);
  static_assert(std::is_same_v<tf::coordinate_type<volume_t>, float>);

  // The grid answers in millimetres: fractional anisotropic spacing intact.
  auto p = vol.point_at(1, 1, 1);
  REQUIRE_THAT(p[0], WithinAbs(0.7f, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(1.5f, 1e-6));

  // The unstated request resolves to the coordinate type, so an int16 field
  // legally emits a float mesh; the samples are in hundredths, so the level
  // set of 0 is the sphere.
  auto mesh = tf::make_isosurface(vol, 0.0f);
  REQUIRE(mesh.faces().size() > 0);
  static_assert(
      std::is_same_v<tf::coordinate_type<decltype(mesh.points())>, float>);

  float rmin = 1e9f, rmax = -1e9f;
  for (auto q : mesh.points()) {
    const float dx = q[0] - center[0];
    const float dy = q[1] - center[1];
    const float dz = q[2] - center[2];
    const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
    rmin = std::min(rmin, r);
    rmax = std::max(rmax, r);
  }
  REQUIRE(rmin > radius - 0.15f);
  REQUIRE(rmax < radius + 0.15f);
}

TEST_CASE("the factory deduces the two types from its arguments",
          "[volume][carrier]") {
  std::vector<std::int16_t> samples(8, std::int16_t{0});
  const std::array<int, 3> dims{2, 2, 2};

  auto mixed = tf::make_volume(samples.data(), dims,
                               tf::point<float, 3>{1.f, 1.f, 1.f},
                               tf::point<float, 3>{0.f, 0.f, 0.f});
  static_assert(std::is_same_v<decltype(mixed)::sample_type, std::int16_t>);
  static_assert(std::is_same_v<tf::coordinate_type<decltype(mixed)>, float>);

  std::vector<float> fsamples(8, 0.f);
  auto same = tf::make_volume(fsamples.data(), dims,
                              tf::point<float, 3>{1.f, 1.f, 1.f},
                              tf::point<float, 3>{0.f, 0.f, 0.f});
  static_assert(std::is_same_v<decltype(same)::sample_type, float>);
  static_assert(std::is_same_v<tf::coordinate_type<decltype(same)>, float>);

  auto unit = tf::make_volume(samples.data(), dims);
  static_assert(
      std::is_same_v<tf::coordinate_type<decltype(unit)>, std::int16_t>);
}

TEST_CASE("the same-by-default carrier is unchanged", "[volume][carrier]") {
  tf::volume_buffer<float> vb({2, 2, 2});
  static_assert(std::is_same_v<decltype(vb)::sample_type, float>);
  static_assert(std::is_same_v<decltype(vb)::coordinate_type, float>);
  static_assert(
      std::is_same_v<decltype(vb.spacing()), const tf::point<float, 3> &>);

  tf::volume_buffer<std::int16_t, float> ct({2, 2, 2});
  static_assert(std::is_same_v<decltype(ct)::sample_type, std::int16_t>);
  static_assert(std::is_same_v<decltype(ct)::coordinate_type, float>);
  ct.set_spacing(tf::point<float, 3>{0.7f, 0.7f, 1.5f});
  auto view = ct.volume();
  static_assert(std::is_same_v<decltype(view)::sample_type, std::int16_t>);
  static_assert(std::is_same_v<tf::coordinate_type<decltype(view)>, float>);

  tf::volume_buffer<float, float, 2> planar({3, 3});
  static_assert(decltype(planar)::grid_dims::value == 2);
}

TEST_CASE("a copied field keeps its two types unless one is asked for",
          "[volume][carrier]") {
  tf::volume_buffer<std::int16_t, float> ct({2, 2, 2});
  ct.set_spacing(tf::point<float, 3>{0.5f, 0.5f, 0.5f});
  for (auto &s : ct.samples_buffer())
    s = std::int16_t{7};

  auto preserved = tf::make_volume_buffer(ct.volume());
  static_assert(
      std::is_same_v<decltype(preserved)::sample_type, std::int16_t>);
  static_assert(std::is_same_v<decltype(preserved)::coordinate_type, float>);
  REQUIRE(preserved.samples_buffer()[0] == std::int16_t{7});
  REQUIRE_THAT(preserved.spacing()[0], WithinAbs(0.5f, 1e-6));

  auto stated = tf::make_volume_buffer<float>(ct.volume());
  static_assert(std::is_same_v<decltype(stated)::sample_type, float>);
  static_assert(std::is_same_v<decltype(stated)::coordinate_type, float>);
  REQUIRE_THAT(stated.samples_buffer()[0], WithinAbs(7.f, 1e-6));
}

TEST_CASE("a signed integer field booleans in its float coordinate type",
          "[volume][carrier]") {
  const std::array<int, 3> dims{8, 8, 8};
  const tf::point<float, 3> spacing{1.f, 1.f, 1.f};
  auto a_samples = carrier_ct_sphere(dims, spacing,
                                     tf::point<float, 3>{3.5f, 3.5f, 3.5f},
                                     2.5f);
  auto b_samples = carrier_ct_sphere(dims, spacing,
                                     tf::point<float, 3>{4.5f, 3.5f, 3.5f},
                                     2.5f);
  auto a = tf::make_volume(a_samples.data(), dims, spacing,
                           tf::point<float, 3>{0.f, 0.f, 0.f});
  auto b = tf::make_volume(b_samples.data(), dims, spacing,
                           tf::point<float, 3>{0.f, 0.f, 0.f});

  auto merged = tf::make_boolean(a, b, tf::volume_boolean_op::union_);
  static_assert(std::is_same_v<decltype(merged)::sample_type, float>);
  const auto i = merged.linear_index(4, 4, 4);
  REQUIRE_THAT(merged.samples_buffer()[i],
               WithinAbs(std::min(static_cast<float>(a[i]),
                                  static_cast<float>(b[i])),
                         1e-6));
}

TEST_CASE("the isosurface emits in the frame the volume states",
          "[volume][carrier][frame]") {
  auto field = tf::make_sphere_sdf<float>(
      {12, 12, 12}, tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-2.75f, -2.75f, -2.75f},
      tf::point<float, 3>{0.f, 0.f, 0.f}, 2.0f);

  auto local = tf::make_isosurface(field.volume(), 0.0f);
  REQUIRE(local.faces().size() > 0);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{4.f, 0.f, 0.f}));
  auto posed = tf::make_isosurface(field.volume() | tf::tag(frame), 0.0f);

  REQUIRE(posed.points().size() == local.points().size());
  for (std::size_t i = 0; i < local.points().size(); ++i) {
    REQUIRE_THAT(posed.points()[i][0],
                 WithinAbs(local.points()[i][0] + 4.f, 1e-5));
    REQUIRE_THAT(posed.points()[i][1], WithinAbs(local.points()[i][1], 1e-5));
    REQUIRE_THAT(posed.points()[i][2], WithinAbs(local.points()[i][2], 1e-5));
  }
}

namespace {

auto carrier_sphere_field() {
  return tf::make_sphere_sdf<float>(
      {12, 12, 12}, tf::point<float, 3>{0.5f, 0.5f, 0.5f},
      tf::point<float, 3>{-2.75f, -2.75f, -2.75f},
      tf::point<float, 3>{0.f, 0.f, 0.f}, 2.0f);
}

} // namespace

TEST_CASE("a rotated frame turns the emitted surface, not the field",
          "[volume][carrier][frame]") {
  auto field = carrier_sphere_field();
  auto frame = tf::make_frame(tf::make_rotation(tf::deg(30.f), tf::axis<2>));

  auto local = tf::make_isosurface(field.volume(), 0.0f);
  auto posed = tf::make_isosurface(field.volume() | tf::tag(frame), 0.0f);
  REQUIRE(posed.points().size() == local.points().size());
  for (std::size_t i = 0; i < local.points().size(); ++i) {
    const auto expected = tf::transformed(local.points()[i], frame);
    for (int d = 0; d < 3; ++d)
      REQUIRE_THAT(posed.points()[i][d], WithinAbs(expected[d], 1e-5));
  }
  REQUIRE_THAT(tf::signed_volume(posed.polygons()),
               WithinAbs(tf::signed_volume(local.polygons()), 1e-3));

  auto local_dc = tf::make_isosurface(field.volume(), 0.0f,
                                      tf::isosurface_method::dual_contouring);
  auto posed_dc =
      tf::make_isosurface(field.volume() | tf::tag(frame), 0.0f,
                          tf::isosurface_method::dual_contouring);
  REQUIRE(posed_dc.points().size() == local_dc.points().size());
  for (std::size_t i = 0; i < local_dc.points().size(); ++i) {
    const auto expected = tf::transformed(local_dc.points()[i], frame);
    for (int d = 0; d < 3; ++d)
      REQUIRE_THAT(posed_dc.points()[i][d], WithinAbs(expected[d], 1e-5));
  }
  REQUIRE_THAT(tf::signed_volume(posed_dc.polygons()),
               WithinAbs(tf::signed_volume(local_dc.polygons()), 1e-3));
}

TEST_CASE("a reflecting frame keeps the surface wound outward",
          "[volume][carrier][frame]") {
  auto field = carrier_sphere_field();
  auto mirror = tf::make_identity_transformation<float, 3>();
  mirror(0, 0) = -1.f;
  auto frame = tf::make_frame(mirror);

  auto local = tf::make_isosurface(field.volume(), 0.0f);
  auto posed = tf::make_isosurface(field.volume() | tf::tag(frame), 0.0f);
  REQUIRE(tf::signed_volume(local.polygons()) > 0.f);
  REQUIRE(tf::signed_volume(posed.polygons()) > 0.f);
  REQUIRE_THAT(tf::signed_volume(posed.polygons()),
               WithinAbs(tf::signed_volume(local.polygons()), 1e-3));
}

TEST_CASE("the 2D isocontours emit in the frame the grid states",
          "[volume][carrier][frame]") {
  tf::volume_buffer<float, float, 2> disc({16, 16});
  disc.set_spacing(tf::point<float, 2>{0.5f, 0.5f});
  disc.set_origin(tf::point<float, 2>{-3.75f, -3.75f});
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x) {
      const auto p = disc.point_at(x, y);
      disc(x, y) = std::sqrt(p[0] * p[0] + p[1] * p[1]) - 2.0f;
    }

  auto local = tf::make_isocontours(disc.volume(), 0.0f);
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 2>{3.f, -1.f}));
  auto posed = tf::make_isocontours(disc.volume() | tf::tag(frame), 0.0f);
  REQUIRE(posed.points().size() == local.points().size());
  REQUIRE(posed.points().size() > 0);
  for (std::size_t i = 0; i < local.points().size(); ++i) {
    REQUIRE_THAT(posed.points()[i][0],
                 WithinAbs(local.points()[i][0] + 3.f, 1e-5));
    REQUIRE_THAT(posed.points()[i][1],
                 WithinAbs(local.points()[i][1] - 1.f, 1e-5));
  }
}

TEST_CASE("the lifted slice contours emit in the frame the volume states",
          "[volume][carrier][frame]") {
  auto field = carrier_sphere_field();
  const auto plane_origin = tf::point<float, 3>{-2.75f, -2.75f, 0.f};
  const auto u = tf::point<float, 3>{1.f, 0.f, 0.f};
  const auto v = tf::point<float, 3>{0.f, 1.f, 0.f};
  const std::array<int, 2> dims2{16, 16};
  const std::array<float, 2> spacing2{0.4f, 0.4f};
  const auto one_iso = std::array<float, 1>{0.f};
  const auto isovalues = tf::make_range(one_iso.data(), one_iso.data() + 1);

  auto local = tf::make_isocontours(field.volume(), plane_origin, u, v, dims2,
                                    spacing2, isovalues);
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{0.f, 0.f, 5.f}));
  auto posed = tf::make_isocontours(field.volume() | tf::tag(frame),
                                    plane_origin, u, v, dims2, spacing2,
                                    isovalues);
  REQUIRE(posed.points().size() == local.points().size());
  REQUIRE(posed.points().size() > 0);
  for (std::size_t i = 0; i < local.points().size(); ++i) {
    REQUIRE_THAT(posed.points()[i][2],
                 WithinAbs(local.points()[i][2] + 5.f, 1e-5));
    REQUIRE_THAT(posed.points()[i][0], WithinAbs(local.points()[i][0], 1e-5));
  }
}

TEST_CASE("a stated frame does not move the slice plane",
          "[volume][carrier][frame]") {
  auto field = carrier_sphere_field();
  const auto plane_origin = tf::point<float, 3>{-2.75f, -2.75f, 0.25f};
  const auto u = tf::point<float, 3>{1.f, 0.f, 0.f};
  const auto v = tf::point<float, 3>{0.f, 1.f, 0.f};
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{7.f, 0.f, 0.f}));

  auto local =
      tf::make_volume_slice(field.volume(), plane_origin, u, v, {12, 12},
                            std::array<float, 2>{0.5f, 0.5f});
  auto posed =
      tf::make_volume_slice(field.volume() | tf::tag(frame), plane_origin, u,
                            v, {12, 12}, std::array<float, 2>{0.5f, 0.5f});
  REQUIRE(posed.voxel_count() == local.voxel_count());
  for (std::size_t i = 0; i < local.voxel_count(); ++i)
    REQUIRE(posed.samples_buffer()[i] == local.samples_buffer()[i]);
}

TEST_CASE("an identity frame collapses to the untagged volume",
          "[volume][carrier][frame]") {
  auto field = carrier_sphere_field();
  auto vol = field.volume();
  static_assert(std::is_same_v<
                std::decay_t<decltype(vol |
                                      tf::tag(tf::identity_frame<float, 3>{}))>,
                decltype(vol)>);
  SUCCEED();
}
