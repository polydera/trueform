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
#include <trueform/volume/impl/flying_edges.hpp>
#include <trueform/volume/impl/marching_cubes.hpp>
#include <trueform/volume/impl/marching_cubes_tables.hpp>
#include <trueform/volume/make_mesh_sdf.hpp>
#include <chrono>
#include <cmath>
#include <utility>

using Catch::Matchers::WithinAbs;

TEST_CASE("volume addressing and geometry", "[volume]") {
  tf::volume_buffer<float> vol({4, 3, 2});
  vol.set_spacing(tf::point<float, 3>{0.5f, 0.5f, 0.5f});
  vol.set_origin(tf::point<float, 3>{1.0f, 2.0f, 3.0f});

  REQUIRE(vol.dims() == std::array<int, 3>{4, 3, 2});
  REQUIRE(vol.voxel_count() == 24u);
  REQUIRE(vol.samples_buffer().size() == 24u);

  // x-fastest layout
  REQUIRE(vol.linear_index(0, 0, 0) == 0u);
  REQUIRE(vol.linear_index(1, 0, 0) == 1u);
  REQUIRE(vol.linear_index(0, 1, 0) == 4u);
  REQUIRE(vol.linear_index(0, 0, 1) == 12u);

  // index -> local point: origin + index * spacing
  auto p = vol.point_at(2, 1, 1);
  REQUIRE_THAT(p[0], WithinAbs(1.0f + 2 * 0.5f, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(2.0f + 1 * 0.5f, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(3.0f + 1 * 0.5f, 1e-6));

  // round-trip a write through operator()
  vol(2, 1, 1) = 42.0f;
  REQUIRE(vol.samples_buffer()[vol.linear_index(2, 1, 1)] == 42.0f);
}

TEST_CASE("sphere SDF has correct sign and zero-crossing", "[volume][sdf]") {
  const float r = 2.0f;
  auto vol = tf::make_sphere_sdf<float>(
      {21, 21, 21}, tf::point<float, 3>{0.2f, 0.2f, 0.2f},
      tf::point<float, 3>{-2.0f, -2.0f, -2.0f},
      tf::point<float, 3>{0.0f, 0.0f, 0.0f}, r);

  // Center voxel (index 10) sits at local (0,0,0) == sphere center: SDF == -r.
  REQUIRE_THAT(vol(10, 10, 10), WithinAbs(-r, 1e-5));

  // A sample on the +x axis: distance from center minus radius.
  auto p = vol.point_at(15, 10, 10);
  const float expected = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - r;
  REQUIRE_THAT(vol(15, 10, 10), WithinAbs(expected, 1e-5));

  // Corner is outside (positive), center is inside (negative).
  REQUIRE(vol(0, 0, 0) > 0.0f);
  REQUIRE(vol(10, 10, 10) < 0.0f);
}

TEST_CASE("mesh SDF samples the frame the form states", "[volume][sdf]") {
  // A unit sphere mesh, once in its local frame and once carried +4x by a
  // frame tag. Both fields sample the same world-space grid, so the zero
  // level set must sit at the center the form states: local at 0, tagged at
  // +4x — and the tagged field is the local field translated exactly.
  auto sphere = tf::make_sphere_mesh<int>(1.0f, 24, 24);
  tf::face_membership<int> fm;
  fm.build(sphere.polygons());
  tf::manifold_edge_link<int, 3> mel;
  mel.build(sphere.faces(), fm);
  tf::aabb_tree<int, float, 3> tree(sphere.polygons(), tf::config_tree(4, 4));
  auto form = sphere.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  // x spans [-2, 6] (41 samples), y and z span [-2, 2]: both centers covered.
  const std::array<int, 3> dims{41, 21, 21};
  const tf::point<float, 3> spacing{0.2f, 0.2f, 0.2f};
  const tf::point<float, 3> origin{-2.0f, -2.0f, -2.0f};

  auto local = tf::make_mesh_sdf<float>(form, dims, spacing, origin);
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<float, 3>{4.0f, 0.0f, 0.0f}));
  auto world =
      tf::make_mesh_sdf<float>(form | tf::tag(frame), dims, spacing, origin);

  // Sample (10,10,10) is world (0,0,0); sample (30,10,10) is world (4,0,0).
  REQUIRE(local(10, 10, 10) < 0.0f);
  REQUIRE(local(30, 10, 10) > 0.0f);
  REQUIRE(world(30, 10, 10) < 0.0f);
  REQUIRE(world(10, 10, 10) > 0.0f);

  // The tagged field is the local field translated by the frame.
  REQUIRE_THAT(world(30, 10, 10), WithinAbs(local(10, 10, 10), 1e-4f));
  REQUIRE_THAT(world(35, 12, 11), WithinAbs(local(15, 12, 11), 1e-4f));
}

TEST_CASE("isosurface of a sphere SDF is a valid sphere mesh",
          "[volume][isosurface]") {
  const float r = 1.5f;
  const float h = 0.1f;
  // Grid spans [-2.5, 2.5]^3; the r=1.5 surface sits well inside with margin,
  // so the extracted mesh must be closed (not clipped by the grid boundary).
  auto vol = tf::make_sphere_sdf<float>(
      {51, 51, 51}, tf::point<float, 3>{h, h, h},
      tf::point<float, 3>{-2.5f, -2.5f, -2.5f},
      tf::point<float, 3>{0.0f, 0.0f, 0.0f}, r);

  // Default isovalue is 0 (the SDF zero level set).
  auto mesh = tf::make_isosurface(vol.volume());

  REQUIRE(mesh.points_buffer().size() > 0u);
  REQUIRE(mesh.size() > 0u);

  // Watertight, manifold surface.
  REQUIRE(tf::is_closed(mesh.polygons()));
  REQUIRE(tf::is_manifold(mesh.polygons()));

  // Every vertex must sit on the sphere of radius r. Because the field is an
  // SDF (near-linear along each edge), the linear-interpolation error is
  // O(h^2)*curvature, not O(h): measured max deviation is ~8e-4 here — roughly
  // 1% of a voxel. We assert well under 2% of a voxel to pin that precision.
  const auto &pts = mesh.points_buffer();
  float max_radius_dev = 0.0f;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto p = pts[i];
    const float d = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    max_radius_dev = std::max(max_radius_dev, std::abs(d - r));
  }
  REQUIRE(max_radius_dev < 0.02f * h);

  // Outward-oriented (positive signed volume) and close to the analytic volume.
  const float analytic = 4.0f / 3.0f * 3.14159265358979f * r * r * r;
  const float vol_signed = tf::signed_volume(mesh.polygons());
  REQUIRE(vol_signed > 0.0f);
  REQUIRE_THAT(vol_signed, WithinAbs(analytic, 0.03f * analytic));
}


TEST_CASE("flying edges matches marching cubes", "[volume][isosurface][fe]") {
  // A sphere SDF has iso surfaces at every level in [-r, boundary]; sweeping the
  // isovalue exercises interior cells, the grid boundary faces, and the empty
  // extremes. Flying Edges (behind tf::make_isosurface) must produce the same welded
  // surface as the marching-cubes baseline: same triangle/vertex counts, both
  // watertight and manifold, and the same enclosed volume.
  const float r = 1.5f;
  const float h = 0.1f;
  auto vol = tf::make_sphere_sdf<float>(
      {51, 51, 51}, tf::point<float, 3>{h, h, h},
      tf::point<float, 3>{-2.5f, -2.5f, -2.5f},
      tf::point<float, 3>{0.0f, 0.0f, 0.0f}, r);

  // Isovalues split into two groups. `aligned` levels are placed exactly on
  // grid samples (an SDF sphere hits e.g. iso 0 at distance r along the axes),
  // so there are zero-area crossings: Flying Edges keeps those degenerate
  // triangles where marching cubes' soup-weld collapses coincident vertices —
  // the surface is identical (same enclosed volume) but the triangle/vertex
  // counts differ. `generic` levels avoid grid-aligned crossings, so the two
  // extractors must agree down to the exact triangle and vertex count.
  const float aligned[] = {-1.0f, -0.5f, 0.0f};
  const float generic[] = {0.37f, 0.8f};

  for (float iso : aligned) {
    INFO("aligned iso=" << iso);
    auto fe = tf::make_isosurface(vol.volume(), iso);
    auto mc = tf::volume_detail::marching_cubes<int>(vol.volume(), iso);
    REQUIRE(fe.size() > 0u);
    // Watertight by construction; FE never has fewer triangles than the weld.
    REQUIRE(tf::is_closed(fe.polygons()));
    REQUIRE(tf::is_manifold(fe.polygons()));
    REQUIRE(fe.size() >= mc.size());
    // Same oriented surface => same enclosed volume, near-exact.
    const float fe_vol = tf::signed_volume(fe.polygons());
    const float mc_vol = tf::signed_volume(mc.polygons());
    REQUIRE_THAT(fe_vol, WithinAbs(mc_vol, 1e-4f * std::abs(mc_vol) + 1e-6f));
  }

  for (float iso : generic) {
    INFO("generic iso=" << iso);
    auto fe = tf::make_isosurface(vol.volume(), iso);
    auto mc = tf::volume_detail::marching_cubes<int>(vol.volume(), iso);
    REQUIRE(fe.size() > 0u);
    REQUIRE(tf::is_closed(fe.polygons()));
    REQUIRE(tf::is_manifold(fe.polygons()));
    // No grid-aligned crossings => exact agreement with marching cubes.
    REQUIRE(fe.size() == mc.size());
    REQUIRE(fe.points_buffer().size() == mc.points_buffer().size());
    const float fe_vol = tf::signed_volume(fe.polygons());
    const float mc_vol = tf::signed_volume(mc.polygons());
    REQUIRE_THAT(fe_vol, WithinAbs(mc_vol, 1e-4f * std::abs(mc_vol) + 1e-6f));
  }
}

TEST_CASE("flying edges vs marching cubes throughput", "[.][bench]") {
  const int n = 201;
  const float r = 6.0f;
  const float half = 10.0f;
  const float h = 2.0f * half / (n - 1);
  auto vol = tf::make_sphere_sdf<float>(
      {n, n, n}, tf::point<float, 3>{h, h, h},
      tf::point<float, 3>{-half, -half, -half},
      tf::point<float, 3>{0.0f, 0.0f, 0.0f}, r);

  const int reps = 5;
  auto bench = [&](auto &&fn) {
    auto m = fn(); // warm up / keep result alive
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
      m = fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::make_pair(
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps,
        m.size());
  };

  auto [fe_ms, fe_tris] = bench([&] { return tf::make_isosurface(vol.volume(), 0.0f); });
  auto [mc_ms, mc_tris] =
      bench([&] { return tf::volume_detail::marching_cubes<int>(vol.volume(), 0.0f); });

  WARN("grid=" << n << "^3  tris(fe)=" << fe_tris << " tris(mc)=" << mc_tris
               << "  flying_edges=" << fe_ms << "ms  marching_cubes=" << mc_ms
               << "ms  speedup=" << (mc_ms / fe_ms) << "x");
}

TEST_CASE("marching cubes table is internally consistent", "[volume][mc]") {
  using namespace tf::volume_detail;

  for (int cube_index = 0; cube_index < 256; ++cube_index) {
    const int *row = mc_tri_table[cube_index];

    int entries = 0;
    for (int e = 0; row[e] != -1; ++e) {
      ++entries;
      const int edge = row[e];
      REQUIRE(edge >= 0);
      REQUIRE(edge < 12);

      // Every edge a triangle uses must be an actual crossing: its two corners
      // must sit on opposite sides of the surface (one inside bit set, one not).
      const int a = mc_edge_corners[edge][0];
      const int b = mc_edge_corners[edge][1];
      const bool inside_a = (cube_index >> a) & 1;
      const bool inside_b = (cube_index >> b) & 1;
      INFO("cube_index=" << cube_index << " slot=" << e << " edge=" << edge);
      REQUIRE(inside_a != inside_b);
    }

    // Entries come in whole triangles.
    INFO("cube_index=" << cube_index);
    REQUIRE(entries % 3 == 0);
  }
}

TEST_CASE("volume boolean on a shared grid", "[volume][boolean]") {
  // Two concentric sphere SDFs on the SAME grid (fast, exact voxel-wise path).
  // union = larger sphere, intersection = smaller sphere, difference = the
  // spherical shell between them.
  const float h = 0.1f;
  const std::array<int, 3> dims{51, 51, 51};
  const tf::point<float, 3> sp{h, h, h};
  const tf::point<float, 3> og{-2.5f, -2.5f, -2.5f};
  const tf::point<float, 3> c{0.0f, 0.0f, 0.0f};
  const float rA = 1.5f; // large
  const float rB = 0.8f; // small
  auto A = tf::make_sphere_sdf<float>(dims, sp, og, c, rA);
  auto B = tf::make_sphere_sdf<float>(dims, sp, og, c, rB);

  const float pi = 3.14159265358979f;
  const float volA = 4.0f / 3.0f * pi * rA * rA * rA;
  const float volB = 4.0f / 3.0f * pi * rB * rB * rB;

  SECTION("union is the larger sphere") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::union_);
    REQUIRE(u.dims() == dims); // fast path keeps the shared grid
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(tf::signed_volume(m.polygons()), WithinAbs(volA, 0.03f * volA));
  }

  SECTION("intersection is the smaller sphere") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::intersection);
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(tf::signed_volume(m.polygons()), WithinAbs(volB, 0.03f * volB));
  }

  SECTION("difference is the shell between them") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::difference);
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    // Outer sphere (outward) + inner sphere (inward): closed, manifold, and the
    // enclosed volume is the difference of the two ball volumes.
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(tf::signed_volume(m.polygons()),
                 WithinAbs(volA - volB, 0.03f * volA));
  }
}

TEST_CASE("volume boolean resamples mismatched grids", "[volume][boolean]") {
  // Two equal spheres on DIFFERENT grids (offset origins) → the general
  // resample-onto-a-common-grid path. Centers are 1.2 apart with r=1, so the
  // spheres overlap: union, intersection and difference are all non-empty.
  const float h = 0.1f;
  const float r = 1.0f;
  auto sphere_at = [&](float cx) {
    const tf::point<float, 3> center{cx, 0.0f, 0.0f};
    const tf::point<float, 3> origin{cx - 2.0f, -2.0f, -2.0f};
    return tf::make_sphere_sdf<float>({41, 41, 41},
                                      tf::point<float, 3>{h, h, h}, origin,
                                      center, r);
  };
  auto A = sphere_at(-0.6f);
  auto B = sphere_at(0.6f);
  REQUIRE(A.origin()[0] != B.origin()[0]); // grids genuinely differ

  const float pi = 3.14159265358979f;
  const float volOne = 4.0f / 3.0f * pi * r * r * r;

  SECTION("union encloses more than one sphere, less than two") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::union_);
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    const float v = tf::signed_volume(m.polygons());
    REQUIRE(v > volOne);
    REQUIRE(v < 2.0f * volOne);
  }

  SECTION("intersection is smaller than a single sphere and non-empty") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::intersection);
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    const float v = tf::signed_volume(m.polygons());
    REQUIRE(v > 0.0f);
    REQUIRE(v < volOne);
  }

  SECTION("difference carves B out of A") {
    auto u = tf::make_boolean(A.volume(), B.volume(), tf::volume_boolean_op::difference);
    auto m = tf::make_isosurface(u.volume());
    REQUIRE(m.size() > 0u);
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    const float v = tf::signed_volume(m.polygons());
    REQUIRE(v > 0.0f);
    REQUIRE(v < volOne); // A minus the overlap lens
  }
}
