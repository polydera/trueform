/**
 * @file test_canonical_plane.cpp
 * @brief tf::exact::canonical_plane — one name per plane, and only one.
 *
 * The key is what the same-tag pool sorts by, so its equality has to BE
 * coplanarity up to scale and orientation: the scenes here state one plane
 * from triples that differ in winding, in spacing and in the lattice
 * magnitude of the cross they generate, and the plane beside it that must
 * not share the name.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/canonical_plane.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/plane_support.hpp>
#include <trueform/exact/vertex.hpp>

#include <array>

namespace {

using canonical_plane_int_t = tf::exact::int32;
using canonical_plane_pt3_t = tf::exact::pt3<canonical_plane_int_t>;
using canonical_plane_key = tf::exact::canonical_plane<canonical_plane_int_t>;
using canonical_plane_wide_t = tf::exact::meta<canonical_plane_int_t>::T2;

auto key_of(const canonical_plane_pt3_t &p0, const canonical_plane_pt3_t &p1,
            const canonical_plane_pt3_t &p2) -> canonical_plane_key {
  tf::exact::plane_support<canonical_plane_int_t> support;
  support.offer(p0);
  support.offer(p1);
  support.offer(p2);
  return tf::exact::make_canonical_plane<canonical_plane_int_t>(support);
}

} // namespace

TEST_CASE("canonical_plane: the name is reduced and sign-fixed",
          "[exact][canonical_plane]") {
  const auto key =
      key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{4, 0, 0},
             canonical_plane_pt3_t{0, 4, 0});
  CHECK(key == canonical_plane_key{
                   canonical_plane_wide_t(0), canonical_plane_wide_t(0),
                   canonical_plane_wide_t(1), canonical_plane_wide_t(0)});
}

TEST_CASE("canonical_plane: one plane keeps one name",
          "[exact][canonical_plane]") {
  const auto base =
      key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{4, 0, 0},
             canonical_plane_pt3_t{0, 4, 0});

  SECTION("the opposite winding") {
    CHECK(key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{0, 4, 0},
                 canonical_plane_pt3_t{4, 0, 0}) == base);
  }

  SECTION("another triple of the same plane, generating a longer cross") {
    CHECK(key_of(canonical_plane_pt3_t{7, 3, 0},
                 canonical_plane_pt3_t{-500, 3, 0},
                 canonical_plane_pt3_t{7, 900, 0}) == base);
  }

  SECTION("a parallel plane one unit away") {
    CHECK(key_of(canonical_plane_pt3_t{0, 0, 1}, canonical_plane_pt3_t{4, 0, 1},
                 canonical_plane_pt3_t{0, 4, 1}) != base);
  }
}

TEST_CASE("canonical_plane: a tilted plane reduces to its primitive normal",
          "[exact][canonical_plane]") {
  const auto unit =
      key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{1, -1, 0},
             canonical_plane_pt3_t{0, 1, -1});
  CHECK(unit == canonical_plane_key{
                    canonical_plane_wide_t(1), canonical_plane_wide_t(1),
                    canonical_plane_wide_t(1), canonical_plane_wide_t(0)});
  // the same plane, stated by a triple whose cross is six times as long
  CHECK(key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{3, -3, 0},
               canonical_plane_pt3_t{0, 2, -2}) == unit);
  // and the same plane translated off the origin along itself
  CHECK(key_of(canonical_plane_pt3_t{5, -5, 0},
               canonical_plane_pt3_t{5, -4, -1},
               canonical_plane_pt3_t{2, -2, 0}) == unit);
}

TEST_CASE("canonical_plane: a support with no plane names no plane",
          "[exact][canonical_plane]") {
  const auto collinear =
      key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{6, 0, 0},
             canonical_plane_pt3_t{9, 0, 0});
  CHECK(collinear == canonical_plane_key{
                         canonical_plane_wide_t(0), canonical_plane_wide_t(0),
                         canonical_plane_wide_t(0), canonical_plane_wide_t(0)});
}

TEST_CASE("canonical_plane: the name answers where a point lies",
          "[exact][canonical_plane]") {
  const auto tilted =
      key_of(canonical_plane_pt3_t{0, 0, 0}, canonical_plane_pt3_t{1, -1, 0},
             canonical_plane_pt3_t{0, 1, -1});

  SECTION("a point of the plane reads zero") {
    CHECK(tf::exact::orient3d_plane_value<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{0, 0, 0}) ==
          canonical_plane_wide_t(0));
    CHECK(tf::exact::orient3d_plane_sign<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{3, -3, 0}) == 0);
    CHECK(tf::exact::orient3d_plane_sign<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{5, -4, -1}) == 0);
  }

  SECTION("the two sides read opposite") {
    CHECK(tf::exact::orient3d_plane_sign<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{1, 1, 1}) == 1);
    CHECK(tf::exact::orient3d_plane_sign<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{-1, -1, -1}) == -1);
  }

  SECTION("the value is the reduced normal against the point") {
    // the name is (1, 1, 1, 0), so the value IS the coordinate sum
    CHECK(tf::exact::orient3d_plane_value<canonical_plane_int_t>(
              tilted, canonical_plane_pt3_t{7, 2, -1}) ==
          canonical_plane_wide_t(8));
  }

  SECTION("an offset plane carries its offset in the name") {
    const auto shifted =
        key_of(canonical_plane_pt3_t{0, 0, 4}, canonical_plane_pt3_t{4, 0, 4},
               canonical_plane_pt3_t{0, 4, 4});
    CHECK(shifted == canonical_plane_key{
                         canonical_plane_wide_t(0), canonical_plane_wide_t(0),
                         canonical_plane_wide_t(1), canonical_plane_wide_t(4)});
    CHECK(tf::exact::orient3d_plane_sign<canonical_plane_int_t>(
              shifted, canonical_plane_pt3_t{9, 9, 4}) == 0);
    CHECK(tf::exact::orient3d_plane_value<canonical_plane_int_t>(
              shifted, canonical_plane_pt3_t{0, 0, 7}) ==
          canonical_plane_wide_t(3));
  }
}
