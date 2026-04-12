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
      /*max_aspect_ratio=*/20.0f,
      /*preserve_boundary=*/false,
      /*parallel=*/true,
      /*feature_angle=*/tf::rad<float>(-1),
      /*feature_weight=*/100.0f,
      /*stabilizer=*/1e-3);
  auto [dec, he] = tf::decimated(sphere.polygons(), 0.5f, config);
  REQUIRE(dec.faces().size() > 0);
  REQUIRE(dec.faces().size() <= sphere.faces().size());
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

  tf::remesh_config<float> config(target, /*iterations=*/5);
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

  tf::remesh_config<float> config(target, /*iterations=*/3);
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
  tf::remesh_config<float> config(mel, /*iterations=*/3);
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

  tf::remesh_config<float> config(mel, /*iterations=*/2);
  config.preserve_boundary = true;
  auto [rem, he] = tf::isotropic_remeshed(plane.polygons(), config);
  REQUIRE(rem.faces().size() > 0);
}
