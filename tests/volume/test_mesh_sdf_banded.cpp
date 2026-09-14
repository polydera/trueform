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
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <trueform/core/frame.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/core/vector.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>
#include <trueform/volume/make_mesh_sdf.hpp>
#include <vector>

namespace {

struct sdfband_report {
  std::size_t band_wrong = 0;
  std::size_t sign_wrong = 0;
  float far_max = 0;
  float far_p99 = 0;
};

/// Banded against exact: the near set (within `near_h` voxels of the
/// surface, surely in the band) must match to float rounding, the sign must
/// match everywhere clear of zero, and the far field is judged by its
/// percentile and maximum.
template <typename Vol>
auto sdfband_compare(const Vol &exact, const Vol &banded, float h,
                     float near_h) -> sdfband_report {
  sdfband_report r;
  std::vector<float> far_errors;
  auto fe = exact.volume();
  auto fb = banded.volume();
  const auto dims = fe.dims();
  for (int z = 0; z < dims[2]; ++z)
    for (int y = 0; y < dims[1]; ++y)
      for (int x = 0; x < dims[0]; ++x) {
        const float a = fe(x, y, z);
        const float b = fb(x, y, z);
        if (std::abs(a) > 1e-6f && ((a < 0) != (b < 0)))
          ++r.sign_wrong;
        const float err = std::abs(a - b);
        if (std::abs(a) <= near_h * h) {
          if (err > 1e-5f)
            ++r.band_wrong;
        } else {
          far_errors.push_back(err);
        }
      }
  std::sort(far_errors.begin(), far_errors.end());
  if (!far_errors.empty()) {
    r.far_max = far_errors.back();
    r.far_p99 = far_errors[std::size_t(double(far_errors.size()) * 0.99)];
  }
  return r;
}

} // namespace

TEST_CASE("the default mesh sdf is the exact mode", "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 16, 16);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{21, 21, 21};
  const tf::point<float, 3> spacing{0.13f, 0.13f, 0.13f};
  const tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};

  auto unstated = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto braced = tf::make_mesh_sdf<float>(form, dims, spacing, origin, {});
  auto named = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                        tf::mesh_sdf_mode::exact);

  const auto bytes = unstated.samples_buffer().size() * sizeof(float);
  REQUIRE(std::memcmp(unstated.samples_buffer().data(),
                      braced.samples_buffer().data(), bytes) == 0);
  REQUIRE(std::memcmp(unstated.samples_buffer().data(),
                      named.samples_buffer().data(), bytes) == 0);
}

TEST_CASE("the banded field is exact in the band and near beyond it",
          "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  for (const int g : {41, 81}) {
    const std::array<int, 3> dims{g, g, g};
    const float h = 2.6f / float(g - 1);
    const tf::point<float, 3> spacing{h, h, h};
    const tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};

    auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
    auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                           tf::mesh_sdf_mode::banded);
    const auto r = sdfband_compare(exact, banded, h, 1.5f);
    REQUIRE(r.band_wrong == 0u);
    REQUIRE(r.sign_wrong == 0u);
    REQUIRE(r.far_p99 <= 1.2f * h);
    REQUIRE(r.far_max <= 8.0f * h);
  }
}

TEST_CASE("a wider band is exact wider", "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  const int g = 41;
  const std::array<int, 3> dims{g, g, g};
  const float h = 2.6f / float(g - 1);
  const tf::point<float, 3> spacing{h, h, h};
  const tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};

  auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto banded = tf::make_mesh_sdf<float>(
      form, dims, spacing, origin, {tf::mesh_sdf_mode::banded, 4});
  const auto r = sdfband_compare(exact, banded, h, 3.5f);
  REQUIRE(r.band_wrong == 0u);
  REQUIRE(r.sign_wrong == 0u);
}

TEST_CASE("the banded field holds on a reflex prism", "[volume][sdf][banded]") {
  tf::polygons_buffer<int, float, 3, 3> prism;
  const std::array<std::array<float, 2>, 6> ring{
      {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}}};
  for (int level = 0; level < 2; ++level)
    for (const auto &q : ring)
      prism.points_buffer().push_back({q[0], q[1], float(level)});
  const auto cap = std::array<std::array<int, 3>, 4>{
      {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}}};
  for (const auto &t : cap) {
    prism.faces_buffer().push_back({t[0], t[2], t[1]});
    prism.faces_buffer().push_back({t[0] + 6, t[1] + 6, t[2] + 6});
  }
  for (int i = 0; i < 6; ++i) {
    const int j = (i + 1) % 6;
    prism.faces_buffer().push_back({i, j, j + 6});
    prism.faces_buffer().push_back({i, j + 6, i + 6});
  }
  tf::aabb_tree<int, float, 3> tree(prism.polygons(), tf::config_tree(4, 4));
  auto form = prism.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{45, 45, 25};
  const tf::point<float, 3> spacing{0.061f, 0.061f, 0.063f};
  const tf::point<float, 3> origin{-0.35f, -0.35f, -0.27f};

  auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                         tf::mesh_sdf_mode::banded);
  const auto r = sdfband_compare(exact, banded, 0.063f, 1.5f);
  REQUIRE(r.band_wrong == 0u);
  REQUIRE(r.sign_wrong == 0u);
  REQUIRE(r.far_p99 <= 1.2f * 0.063f);
  REQUIRE(r.far_max <= 8.0f * 0.063f);
}

TEST_CASE("the banded field is direction-free", "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);
  const int g = 41;
  const float h = 2.6f / float(g - 1);
  tf::point<float, 3> spacing{h, -h, h};
  tf::point<float, 3> origin{-1.3f, 1.3f, -1.3f};
  auto exact = tf::make_mesh_sdf<float>(form, {g, g, g}, spacing, origin);
  auto banded = tf::make_mesh_sdf<float>(form, {g, g, g}, spacing, origin,
                                         tf::mesh_sdf_mode::banded);
  const auto r = sdfband_compare(exact, banded, h, 1.5f);
  REQUIRE(r.band_wrong == 0u);
  REQUIRE(r.sign_wrong == 0u);
  REQUIRE(r.far_p99 <= 1.2f * h);
}

TEST_CASE("a grid inside the mesh is measured whole", "[volume][sdf][banded]") {
  // No crossing reaches the grid: the banded mode measures every sample,
  // so the field is the exact one to the byte, finite and negative.
  auto sphere = tf::make_sphere_mesh<int>(2.0f, 24, 24);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);
  const std::array<int, 3> dims{16, 16, 16};
  const tf::point<float, 3> spacing{0.05f, 0.05f, 0.05f};
  const tf::point<float, 3> origin{-0.4f, -0.4f, -0.4f};
  auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                         tf::mesh_sdf_mode::banded);
  REQUIRE(std::memcmp(exact.samples_buffer().data(),
                      banded.samples_buffer().data(),
                      exact.samples_buffer().size() * sizeof(float)) == 0);
  for (int z = 0; z < 16; ++z)
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) {
        REQUIRE(std::isfinite(banded.volume()(x, y, z)));
        REQUIRE(banded.volume()(x, y, z) < 0.0f);
      }
}

TEST_CASE("the banded field holds where the surface leaves the grid",
          "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 32, 32);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  { // the octant: three faces of the grid cut through the solid
    const int g = 33;
    const float h = 1.25f / float(g - 1);
    const tf::point<float, 3> spacing{h, h, h};
    const tf::point<float, 3> origin{0.05f, 0.05f, 0.05f};
    auto exact = tf::make_mesh_sdf<float>(form, {g, g, g}, spacing, origin);
    auto banded = tf::make_mesh_sdf<float>(form, {g, g, g}, spacing, origin,
                                           tf::mesh_sdf_mode::banded);
    const auto r = sdfband_compare(exact, banded, h, 1.5f);
    REQUIRE(r.band_wrong == 0u);
    REQUIRE(r.sign_wrong == 0u);
    REQUIRE(r.far_p99 <= 1.2f * h);
    REQUIRE(r.far_max <= 4.0f * h);
  }
  { // the slab: a thin z-window near the top of the sphere
    const std::array<int, 3> dims{41, 41, 9};
    const float h = 2.6f / 40.0f;
    const float hz = 0.2f / 8.0f;
    const tf::point<float, 3> spacing{h, h, hz};
    const tf::point<float, 3> origin{-1.3f, -1.3f, 0.78f};
    auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
    auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                           tf::mesh_sdf_mode::banded);
    // The surface above the slab is off-grid: no crossing, no band claim —
    // the near set is proxied by the z step, and the off-grid neighbourhood
    // answers under the far-field promise.
    const auto r = sdfband_compare(exact, banded, h, 1.5f * hz / h);
    REQUIRE(r.band_wrong == 0u);
    REQUIRE(r.sign_wrong == 0u);
    // the slab leaves the surface: outside the coverage contract the far
    // field holds within the shell walk's answer, its p99 under a voxel
    REQUIRE(r.far_p99 <= 1.2f * h);
    REQUIRE(r.far_max <= 2.5f * h);
  }
}

namespace {

auto sdfband_merged(tf::polygons_buffer<int, float, 3, 3> a,
                    const tf::polygons_buffer<int, float, 3, 3> &b)
    -> tf::polygons_buffer<int, float, 3, 3> {
  const int base = int(a.points_buffer().size());
  for (std::size_t i = 0; i < b.points_buffer().size(); ++i)
    a.points_buffer().push_back(b.points_buffer()[i]);
  for (std::size_t f = 0; f < b.size(); ++f) {
    auto face = b.faces_buffer()[f];
    a.faces_buffer().push_back(
        {face[0] + base, face[1] + base, face[2] + base});
  }
  return a;
}

auto sdfband_translated(tf::polygons_buffer<int, float, 3, 3> mesh, float dx,
                        float dy, float dz)
    -> tf::polygons_buffer<int, float, 3, 3> {
  for (auto p : mesh.points_buffer().points()) {
    p[0] = p[0] + dx;
    p[1] = p[1] + dy;
    p[2] = p[2] + dz;
  }
  return mesh;
}

} // namespace

TEST_CASE("the banded field measures what no line informs",
          "[volume][sdf][banded]") {
  // An in-grid feature keeps the sweep alive while a second surface sits
  // past the grid: the samples only that surface could answer are blind to
  // every line and join the measured set.
  const int g = 33;
  const float h = 2.0f / float(g - 1);
  const std::array<int, 3> dims{g, g, g};
  const tf::point<float, 3> spacing{h, h, h};
  const tf::point<float, 3> origin{-1.0f, -1.0f, -1.0f};
  auto feature = tf::make_sphere_mesh<int>(0.25f, 16, 16);

  for (const float k : {2.0f, 6.0f, 12.0f}) {
    auto wall = sdfband_translated(tf::make_box_mesh<int>(4.0f, 8.0f, 8.0f),
                                   1.0f + k * h + 2.0f, 0.0f, 0.0f);
    auto mesh = sdfband_merged(feature, wall);
    tf::aabb_tree<int, float, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto form = mesh.polygons() | tf::tag(tree);
    auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
    auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                           tf::mesh_sdf_mode::banded);
    const auto r = sdfband_compare(exact, banded, h, 1.5f);
    INFO("wall at " << k << "h: p99 " << r.far_p99 / h << "h max "
                    << r.far_max / h << "h");
    REQUIRE(r.band_wrong == 0u);
    REQUIRE(r.sign_wrong == 0u);
    REQUIRE(r.far_p99 <= 1.2f * h);
    REQUIRE(r.far_max <= 2.5f * h);
  }

  { // a blob just past the grid's corner, diagonal to every axis
    auto blob = sdfband_translated(tf::make_sphere_mesh<int>(3.0f * h, 12, 12),
                                   1.0f + 2.0f * h, 1.0f + 2.0f * h,
                                   1.0f + 2.0f * h);
    auto mesh = sdfband_merged(feature, blob);
    tf::aabb_tree<int, float, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto form = mesh.polygons() | tf::tag(tree);
    auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
    auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                           tf::mesh_sdf_mode::banded);
    const auto r = sdfband_compare(exact, banded, h, 1.5f);
    INFO("off-corner blob: p99 " << r.far_p99 / h << "h max "
                                 << r.far_max / h << "h");
    REQUIRE(r.band_wrong == 0u);
    REQUIRE(r.sign_wrong == 0u);
    REQUIRE(r.far_p99 <= 1.2f * h);
    REQUIRE(r.far_max <= 2.5f * h);
  }
}

TEST_CASE("a feature no line meets is measured whole",
          "[volume][sdf][banded]") {
  // A blob smaller than the line spacing, centered between the lines: the
  // discovery states nothing, so the banded mode measures the whole grid
  // and equals the exact one to the byte.
  const int g = 17;
  const float h = 1.0f / float(g - 1);
  auto blob = sdfband_translated(tf::make_sphere_mesh<int>(0.4f * h, 12, 12),
                                 0.5f * h, 0.5f * h, 0.5f * h);
  tf::aabb_tree<int, float, 3> tree(blob.polygons(), tf::config_tree(4, 4));
  auto form = blob.polygons() | tf::tag(tree);
  const std::array<int, 3> dims{g, g, g};
  const tf::point<float, 3> spacing{h, h, h};
  const tf::point<float, 3> origin{-0.5f, -0.5f, -0.5f};
  auto exact = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto banded = tf::make_mesh_sdf<float>(form, dims, spacing, origin,
                                         tf::mesh_sdf_mode::banded);
  REQUIRE(std::memcmp(exact.samples_buffer().data(),
                      banded.samples_buffer().data(),
                      exact.samples_buffer().size() * sizeof(float)) == 0);
}

TEST_CASE("an untagged mesh gets the tree the call needs",
          "[volume][sdf][banded]") {
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 16, 16);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree);

  const std::array<int, 3> dims{21, 21, 21};
  const tf::point<float, 3> spacing{0.13f, 0.13f, 0.13f};
  const tf::point<float, 3> origin{-1.3f, -1.3f, -1.3f};
  const tf::mesh_sdf_config configs[]{tf::mesh_sdf_mode::exact,
                                      tf::mesh_sdf_mode::banded};

  for (const auto &config : configs) {
    auto tagged = tf::make_mesh_sdf<float>(form, dims, spacing, origin, config);
    auto untagged = tf::make_mesh_sdf<float>(sphere.polygons(), dims, spacing,
                                             origin, config);
    REQUIRE(untagged.samples_buffer().size() ==
            tagged.samples_buffer().size());
    REQUIRE(std::memcmp(untagged.samples_buffer().data(),
                        tagged.samples_buffer().data(),
                        tagged.samples_buffer().size() * sizeof(float)) == 0);
  }

  // the frame rides on the form either way: completion wraps the tree
  // around a framed mesh without displacing it
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{0.37f, -0.11f, 0.05f}));
  auto framed_tagged =
      tf::make_mesh_sdf<float>(form | tf::tag(frame), dims, spacing, origin);
  auto framed_untagged = tf::make_mesh_sdf<float>(
      sphere.polygons() | tf::tag(frame), dims, spacing, origin);
  REQUIRE(std::memcmp(framed_untagged.samples_buffer().data(),
                      framed_tagged.samples_buffer().data(),
                      framed_tagged.samples_buffer().size() * sizeof(float)) ==
          0);

  // the completion names its index type off the faces it was handed
  auto wide = tf::make_sphere_mesh<std::int64_t>(1.0f, 16, 16);
  auto wide_untagged =
      tf::make_mesh_sdf<float>(wide.polygons(), dims, spacing, origin);
  auto narrow = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  REQUIRE(std::memcmp(wide_untagged.samples_buffer().data(),
                      narrow.samples_buffer().data(),
                      narrow.samples_buffer().size() * sizeof(float)) == 0);
}
