/**
 * @file test_winding_number.cpp
 * @brief Tests for generalized winding numbers on spatial forms
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <trueform/trueform.hpp>

namespace {

template <typename Real> auto winding_tetra_mesh() {
  tf::polygons_buffer<int, Real, 3, 3> mesh;
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{1, 0, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 1, 0});
  mesh.points_buffer().push_back(tf::point<Real, 3>{0, 0, 1});
  mesh.faces_buffer().push_back(std::array<int, 3>{0, 2, 1});
  mesh.faces_buffer().push_back(std::array<int, 3>{0, 1, 3});
  mesh.faces_buffer().push_back(std::array<int, 3>{0, 3, 2});
  mesh.faces_buffer().push_back(std::array<int, 3>{1, 2, 3});
  return mesh;
}

constexpr double winding_exact_beta = 1e9;

} // namespace

TEMPLATE_TEST_CASE("winding number: a closed sphere answers one inside and "
                   "zero outside",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 24, 24);
  tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sphere.polygons());
  auto form = sphere.polygons() | tf::tag(tree) | tf::tag(moments);

  for (int x = -3; x <= 3; ++x)
    for (int y = -3; y <= 3; ++y)
      for (int z = -3; z <= 3; ++z) {
        const tf::point<real_t, 3> q{real_t(x) * real_t(0.5),
                                     real_t(y) * real_t(0.5),
                                     real_t(z) * real_t(0.5)};
        const auto r = std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] +
                                 double(q[2]) * q[2]);
        if (std::abs(r - 1.) < 0.15)
          continue; // the near-surface band is a float test by contract
        const bool inside = r < 1.;
        const auto w = tf::winding_number(form, q);
        CHECK((w > 0.5) == inside);
        const auto exact = tf::winding_number(form, q, {winding_exact_beta});
        CHECK(std::abs(exact - (inside ? 1. : 0.)) < 1e-6);
      }
}

TEMPLATE_TEST_CASE("winding number: the exact limit is the solid-angle sum",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto mesh = winding_tetra_mesh<real_t>();
  tf::aabb_tree<int, real_t, 3> tree(mesh.polygons(), tf::config_tree(2, 1));
  auto moments = tf::make_winding_moments(tree, mesh.polygons());
  auto form = mesh.polygons() | tf::tag(tree) | tf::tag(moments);

  const tf::point<real_t, 3> centroid{real_t(0.25), real_t(0.25), real_t(0.25)};
  const tf::point<real_t, 3> outside{real_t(2), real_t(2), real_t(2)};
  CHECK(tf::winding_number(form, centroid, {winding_exact_beta}) ==
        Catch::Approx(1.).margin(1e-12));
  CHECK(tf::winding_number(form, outside, {winding_exact_beta}) ==
        Catch::Approx(0.).margin(1e-12));
  // the default beta stays well clear of the threshold on both sides
  CHECK(tf::winding_number(form, centroid) > 0.9);
  CHECK(tf::winding_number(form, outside) < 0.1);
}

TEMPLATE_TEST_CASE("winding number: the batch is the single query at every "
                   "position",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 16, 16);
  tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sphere.polygons());
  auto form = sphere.polygons() | tf::tag(tree) | tf::tag(moments);

  tf::points_buffer<real_t, 3> queries;
  for (int i = 0; i < 64; ++i)
    queries.push_back(
        tf::point<real_t, 3>{real_t(i % 4) * real_t(0.61) - real_t(1),
                             real_t((i / 4) % 4) * real_t(0.53) - real_t(1),
                             real_t(i / 16) * real_t(0.47) - real_t(1)});
  tf::buffer<double> batch;
  batch.allocate(queries.points().size());
  tf::winding_number(form, queries.points(), batch);
  for (std::size_t i = 0; i < batch.size(); ++i)
    CHECK(batch[i] == tf::winding_number(form, queries.points()[i]));
}

TEMPLATE_TEST_CASE("winding number: an open sheet degrades gracefully",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sheet = tf::make_plane_mesh<int, real_t>(2, 2, 8, 8);
  tf::aabb_tree<int, real_t, 3> tree(sheet.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sheet.polygons());
  auto form = sheet.polygons() | tf::tag(tree) | tf::tag(moments);

  const auto near_above = tf::winding_number(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(0.01)});
  const auto near_below = tf::winding_number(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(-0.01)});
  const auto far_away = tf::winding_number(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(10)});
  CHECK(std::isfinite(near_above));
  CHECK(std::isfinite(near_below));
  CHECK(std::abs(near_above) > 0.3);
  CHECK(std::abs(near_above) < 0.7);
  CHECK(std::abs(near_below) > 0.3);
  CHECK(std::abs(near_below) < 0.7);
  // the sheet separates its two sides by a full winding unit
  CHECK(std::abs(near_above - near_below) == Catch::Approx(1.).margin(0.05));
  CHECK(std::abs(far_away) < 0.05);
}

TEMPLATE_TEST_CASE("winding number: the moments build is deterministic",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 16, 16);
  tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto m0 = tf::make_winding_moments(tree, sphere.polygons());
  auto m1 = tf::make_winding_moments(tree, sphere.polygons());
  auto v0 = m0.moments();
  auto v1 = m1.moments();
  REQUIRE(v0.tickets.size() == v1.tickets.size());
  REQUIRE(v0.blocks.size() == v1.blocks.size());
  CHECK(std::memcmp(v0.tickets.begin(), v1.tickets.begin(),
                    v0.tickets.size() * sizeof(*v0.tickets.begin())) == 0);
  CHECK(std::memcmp(v0.blocks.begin(), v1.blocks.begin(),
                    v0.blocks.size() * sizeof(*v0.blocks.begin())) == 0);
}

TEMPLATE_TEST_CASE("signed distance: winding moments sign like pseudonormals "
                   "on a closed mesh",
                   "[spatial][winding][signed_distance]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 24, 24);
  tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sphere.polygons());
  auto fm = tf::make_face_membership(sphere.polygons());
  auto mel = tf::make_manifold_edge_link(sphere.polygons());
  auto wform = sphere.polygons() | tf::tag(tree) | tf::tag(moments);
  auto pform = sphere.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  for (int i = 0; i < 32; ++i) {
    const tf::point<real_t, 3> q{real_t(i % 4) * real_t(0.6) - real_t(0.9),
                                 real_t((i / 4) % 4) * real_t(0.6) -
                                     real_t(0.9),
                                 real_t(i / 16) * real_t(0.6) - real_t(0.3)};
    const auto r = std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] +
                             double(q[2]) * q[2]);
    if (std::abs(r - 1.) < 0.15)
      continue;
    const auto sd_w = tf::signed_distance(wform, q);
    const auto sd_p = tf::signed_distance(pform, q);
    CHECK(sd_w == Catch::Approx(sd_p).margin(1e-9));
  }
}

TEMPLATE_TEST_CASE("signed distance: a winding-tagged open sheet still "
                   "answers",
                   "[spatial][winding][signed_distance]", float, double) {
  using real_t = TestType;
  auto sheet = tf::make_plane_mesh<int, real_t>(2, 2, 8, 8);
  tf::aabb_tree<int, real_t, 3> tree(sheet.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sheet.polygons());
  auto form = sheet.polygons() | tf::tag(tree) | tf::tag(moments);

  const auto far_above = tf::signed_distance(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(2)});
  CHECK(far_above == Catch::Approx(2.).epsilon(0.01));
  const auto far_below = tf::signed_distance(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(-2)});
  CHECK(far_below == Catch::Approx(2.).epsilon(0.01));
  CHECK(std::isfinite(tf::signed_distance(
      form, tf::point<real_t, 3>{real_t(0), real_t(0), real_t(0.01)})));
}

TEMPLATE_TEST_CASE("winding number: the default beta bounds its error over "
                   "a sample battery",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 32, 32);
  tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto moments = tf::make_winding_moments(tree, sphere.polygons());
  auto form = sphere.polygons() | tf::tag(tree) | tf::tag(moments);

  // deterministic battery off the near-surface band, both sides
  double max_error = 0;
  int sampled = 0;
  for (int i = 0; i < 4096; ++i) {
    const double a = double(i % 64) / 64. * 6.283185307179586;
    const double c = double(i / 64) / 64. * 2. - 1.;
    const double s = std::sqrt(std::max(0., 1. - c * c));
    const double r = 0.05 + 1.9 * double((i * 2654435761u) % 4096u) / 4096.;
    if (std::abs(r - 1.) < 0.15)
      continue;
    const tf::point<real_t, 3> q{real_t(r * s * std::cos(a)),
                                 real_t(r * s * std::sin(a)), real_t(r * c)};
    const double truth = r < 1. ? 1. : 0.;
    max_error =
        std::max(max_error, std::abs(tf::winding_number(form, q) - truth));
    ++sampled;
  }
  REQUIRE(sampled > 3000);
  CHECK(max_error < 5e-3);
}

TEMPLATE_TEST_CASE("winding number: wide inner nodes answer whole",
                   "[spatial][winding]", float, double) {
  using real_t = TestType;
  auto sphere = tf::make_sphere_mesh<int>(real_t(1), 24, 24);
  for (auto cfg : {tf::config_tree(5, 4), tf::config_tree(8, 8),
                   tf::config_tree(16, 16)}) {
    tf::aabb_tree<int, real_t, 3> tree(sphere.polygons(), cfg);
    auto moments = tf::make_winding_moments(tree, sphere.polygons());
    auto form = sphere.polygons() | tf::tag(tree) | tf::tag(moments);
    for (int i = 0; i < 64; ++i) {
      const double a = double(i % 8) / 8. * 6.283185307179586;
      const double c = double(i / 8) / 8. * 1.8 - 0.9;
      const double s = std::sqrt(std::max(0., 1. - c * c));
      const double r = (i % 2) ? 0.55 : 1.45;
      const tf::point<real_t, 3> q{real_t(r * s * std::cos(a)),
                                   real_t(r * s * std::sin(a)), real_t(r * c)};
      const double truth = r < 1. ? 1. : 0.;
      CHECK(std::abs(tf::winding_number(form, q) - truth) < 5e-2);
      CHECK(std::abs(tf::winding_number(form, q, {1e9}) - truth) < 1e-6);
    }
  }
}
