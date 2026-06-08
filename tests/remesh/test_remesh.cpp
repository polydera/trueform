/**
 * @file test_remesh.cpp
 * @brief Tests for decimation and isotropic remeshing
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>

#include <cmath>

TEST_CASE("decimated (box to 50%)", "[remesh][decimate]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f);
  auto orig_faces = box.faces().size();

  auto [dec, he] = tf::decimated(box.polygons(), 0.5f);
  REQUIRE(dec.faces().size() > 0);
  REQUIRE(dec.faces().size() <= orig_faces);
  REQUIRE(dec.points().size() > 0);
  REQUIRE(dec.points().size() <= box.points().size());
}

TEST_CASE("decimated (sphere -preserves rough shape)", "[remesh][decimate]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto orig_faces = sphere.faces().size();

  auto [dec, he] = tf::decimated(sphere.polygons(), 0.3f);
  REQUIRE(dec.faces().size() < orig_faces);
  REQUIRE(dec.faces().size() > 0);

  auto orig_vol = std::abs(tf::signed_volume(sphere.polygons()));
  auto dec_vol = std::abs(tf::signed_volume(dec.polygons()));
  auto ratio = dec_vol / orig_vol;
  REQUIRE(ratio > 0.5f);
  REQUIRE(ratio < 1.5f);
}

TEST_CASE("decimated with options", "[remesh][decimate]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 15, 15);

  tf::decimate_config<float> config(
      /*min_quality=*/0.3f,
      /*preserve_boundary=*/false,
      /*parallel=*/true,
      /*feature_angle=*/tf::rad<float>(-1),
      /*feature_weight=*/100.0f,
      /*stabilizer=*/1e-3);
  auto [dec, he] = tf::decimated(sphere.polygons(), 0.5f, config);
  REQUIRE(dec.faces().size() > 0);
  REQUIRE(dec.faces().size() <= sphere.faces().size());
}

TEST_CASE("simplified (error budget collapses flat faces)", "[remesh][simplify]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f, 4, 4, 4);
  auto orig = box.faces().size();
  auto [s, he] = tf::simplified(box.polygons());
  REQUIRE(s.faces().size() > 0);
  REQUIRE(s.faces().size() < orig); // flat box faces collapse for ~0 error
}

TEST_CASE("simplified (larger budget is coarser)", "[remesh][simplify]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto fine = tf::simplified(sphere.polygons(), tf::simplify_config<float>{0.002f}).first;
  auto coarse = tf::simplified(sphere.polygons(), tf::simplify_config<float>{0.02f}).first;
  REQUIRE(coarse.faces().size() > 0);
  REQUIRE(coarse.faces().size() <= fine.faces().size());
}

TEST_CASE("simplified (centering: large coords do not stall)",
          "[remesh][simplify]") {
  // The bbox-min centering must keep the quadric error meaningful at large
  // (e.g. UTM) coordinates; without it a translated mesh barely simplifies.
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto orig = sphere.faces().size();
  tf::simplify_config<float> cfg{0.01f};
  auto base = tf::simplified(sphere.polygons(), cfg).first;
  REQUIRE(base.faces().size() < orig);

  auto moved = sphere;
  for (std::size_t i = 0; i < moved.points().size(); ++i)
    for (int d = 0; d < 3; ++d)
      moved.points()[i][d] += 500000.0f;
  auto moveds = tf::simplified(moved.polygons(), cfg).first;
  REQUIRE(moveds.faces().size() < orig);                    // simplifies at scale
  REQUIRE(moveds.faces().size() < base.faces().size() * 2); // not stalled
}

namespace {
// Worst minimum interior angle over all faces (radians).
template <typename HE, typename Pts>
float mesh_min_angle(const HE &he, const Pts &points) {
  auto faces = tf::make_faces_buffer(he);
  float worst = 4.0f;
  for (decltype(faces.size()) f = 0; f < faces.size(); ++f) {
    auto t = faces[f];
    auto A = points[t[0]], B = points[t[1]], C = points[t[2]];
    auto L = [](auto p, auto q) {
      float dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
      return std::sqrt(dx*dx + dy*dy + dz*dz);
    };
    float a = L(B,C), b = L(A,C), c = L(A,B);
    auto ang = [&](float o, float s1, float s2) {
      if (s1 <= 0 || s2 <= 0) return 0.0f;
      float co = (s1*s1 + s2*s2 - o*o) / (2*s1*s2);
      co = std::max(-1.0f, std::min(1.0f, co));
      return std::acos(co);
    };
    worst = std::min({worst, ang(a,b,c), ang(b,a,c), ang(c,a,b)});
  }
  return worst;
}
} // namespace

TEST_CASE("improve_triangulation (fixed count)", "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  auto nf = he.number_of_faces();
  auto nv = points.size();
  tf::improve_triangulation(he, points.points()); // default: valence, 3 rounds
  REQUIRE(he.number_of_faces() == nf); // flips + relax never change the count
  REQUIRE(points.size() == nv);
}

TEST_CASE("improve_triangulation (min_angle objective, no folds)",
          "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  tf::improve_config<float> cfg;
  cfg.flip = tf::flip_objective::min_angle;
  cfg.check_normals = true;
  tf::improve_triangulation(he, points.points(), cfg);
  // the fold guard must keep every triangle non-degenerate / non-inverted
  REQUIRE(mesh_min_angle(he, points.points()) > 0.01f); // > ~0.6 degrees
}

TEST_CASE("improve_triangulation (reused workspace)", "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 15, 15);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  auto nf = he.number_of_faces();
  tf::points_buffer<double, 3> old_pos;
  tf::improve_config<float> cfg{1, 2, 0.5f, true, tf::flip_objective::valence};
  tf::improve_triangulation(he, points.points(), old_pos, cfg);
  tf::improve_triangulation(he, points.points(), old_pos, cfg); // reuse buffer
  REQUIRE(he.number_of_faces() == nf);
}

TEST_CASE("isotropicRemeshed (box -basic)", "[remesh][isotropic]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f, 3, 3, 3);
  auto mel = tf::mean_edge_length(box.polygons());

  auto [rem, he] = tf::isotropic_remeshed(box.polygons(), mel * 2.0f);
  REQUIRE(rem.faces().size() > 0);
  REQUIRE(rem.points().size() > 0);
}

TEST_CASE("isotropicRemeshed (sphere -edge lengths converge)",
          "[remesh][isotropic]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 10, 10);
  auto target = tf::mean_edge_length(sphere.polygons());

  tf::isotropic_remesh_config<float> config(target, /*iterations=*/5);
  auto [rem, he] = tf::isotropic_remeshed(sphere.polygons(), config);
  REQUIRE(rem.faces().size() > 0);

  auto min_el = tf::min_edge_length(rem.polygons());
  auto max_el = tf::max_edge_length(rem.polygons());
  auto ratio = max_el / min_el;
  REQUIRE(ratio < 10.0f);
}

TEST_CASE("isotropicRemeshed with quadric", "[remesh][isotropic]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 10, 10);
  auto target = tf::mean_edge_length(sphere.polygons());

  tf::isotropic_remesh_config<float> config(target, /*iterations=*/3);
  config.use_quadric = true;
  auto [rem, he] = tf::isotropic_remeshed(sphere.polygons(), config);
  REQUIRE(rem.faces().size() > 0);
}

TEST_CASE("pipeline: decimate then isotropic remesh",
          "[remesh][pipeline]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto orig_faces = sphere.faces().size();

  auto [dec, he1] = tf::decimated(sphere.polygons(), 0.3f);
  REQUIRE(dec.faces().size() < orig_faces);

  auto mel = tf::mean_edge_length(dec.polygons());
  tf::isotropic_remesh_config<float> config(mel, /*iterations=*/3);
  config.use_quadric = true;
  auto [rem, he2] = tf::isotropic_remeshed(dec.polygons(), config);
  REQUIRE(rem.faces().size() > 0);

  auto orig_vol = std::abs(tf::signed_volume(sphere.polygons()));
  auto rem_vol = std::abs(tf::signed_volume(rem.polygons()));
  auto ratio = rem_vol / orig_vol;
  REQUIRE(ratio > 0.3f);
  REQUIRE(ratio < 2.0f);
}

TEST_CASE("isotropicRemeshed with preserveBoundary",
          "[remesh][isotropic]") {
  auto plane = tf::make_plane_mesh(10.0f, 10.0f, 5, 5);
  auto mel = tf::mean_edge_length(plane.polygons());

  tf::isotropic_remesh_config<float> config(mel, /*iterations=*/2);
  config.preserve_boundary = true;
  auto [rem, he] = tf::isotropic_remeshed(plane.polygons(), config);
  REQUIRE(rem.faces().size() > 0);
}
