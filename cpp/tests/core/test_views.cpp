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
#include "carriers.hpp"

#include "trueform/core/faces.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/range.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using view_index = tf::cpp::default_index_t;
using view_owned = tf::cpp::test::owned_mesh<view_index, float>;
using view_mixed_owned =
    tf::cpp::test::owned_mesh<view_index, float, 3, tf::dynamic_size>;

auto view_triangles() -> view_owned {
  return {tf::cpp::test::polygons_of<view_index, float>(
      {0, 1, 2, 0, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0})};
}

auto view_dynamic_faces() -> view_mixed_owned {
  return {tf::cpp::test::polygons_of<view_index, float>(
      {0, 3, 7}, {0, 1, 2, 0, 1, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0})};
}

auto view_translation(float x) -> std::array<float, 16> {
  return {1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

} // namespace

TEST_CASE("a mesh is the geometry, the frame and the structures behind it",
          "[cpp][core][view]") {
  const auto owned = view_triangles();
  const auto mesh = owned.mesh();

  STATIC_REQUIRE(std::is_same_v<decltype(mesh),
                                const tf::cpp::mesh<view_index, float, 3, 3>>);
  STATIC_REQUIRE(
      !std::is_default_constructible_v<tf::cpp::mesh<view_index, float, 3, 3>>);
  CHECK(mesh.number_of_faces() == 2);
  CHECK(mesh.number_of_points() == 4);
  CHECK(mesh.faces()[1][2] == view_index{3});
  CHECK(mesh.polygons()[0][1][0] == 1.0F);
  // a mesh borrows, so its points are read-only
  STATIC_REQUIRE(
      std::is_const_v<std::remove_reference_t<decltype(mesh.points()[0][0])>>);
}

TEST_CASE("a mesh is read at the arity its geometry is stored in",
          "[cpp][core][view]") {
  const auto triangles = view_triangles();
  const auto mixed = view_dynamic_faces();

  // the layout is the storage's own type, so a mesh has one arity and no
  // other, and asking for the wrong one is not a question a caller can ask
  STATIC_REQUIRE(std::is_same_v<decltype(triangles.mesh()),
                                tf::cpp::mesh<view_index, float, 3, 3>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(mixed.mesh()),
                     tf::cpp::mesh<view_index, float, 3, tf::dynamic_size>>);

  CHECK(triangles.mesh().faces()[0].size() == 3);
  CHECK(mixed.mesh().faces()[1].size() == 4);
  CHECK(mixed.mesh().face_membership().size() == 4);
}

TEST_CASE("a form always carries a frame, identity when none was given",
          "[cpp][core][view]") {
  // BIND THE MESH, THEN TAKE THE FORM: a form keeps no handle, so the mesh it
  // was taken from is what has to be alive while it is read
  auto owned = view_triangles();
  const auto unplaced = owned.mesh();
  const auto plain = unplaced.form();
  STATIC_REQUIRE(tf::has_frame_policy<decltype(plain)>);
  CHECK(plain.transformation()(0, 0) == 1.0F);
  CHECK(plain.transformation()(0, 3) == 0.0F);

  owned.place(view_translation(5));
  const auto moved = owned.mesh();
  const auto placed = moved.form();
  STATIC_REQUIRE(std::is_same_v<decltype(plain), decltype(placed)>);
  CHECK(placed.transformation()(0, 3) == 5.0F);
}

TEST_CASE("a mesh borrows, so what it reads outlives it", "[cpp][core][view]") {
  auto owned = std::make_shared<view_owned>(view_triangles());
  const auto &held = *owned;
  const auto mesh = held.mesh();
  const auto form = mesh.form();

  // a mesh reads the caller's own arrays, which is the law every view in
  // trueform obeys
  CHECK(&mesh.points()[0][0] ==
        held.polygons.points_buffer().data_buffer().data());
  CHECK(form[0][1][0] == 1.0F);
  CHECK(form.tree().bv().max[0] == 1.0F);
  CHECK(mesh.number_of_faces() == 2);

  // a reading that must outlive the caller's handle says so: the keepalive
  // slot retains the geometry AND the cache, because it retains what holds
  // both
  const auto retained = tf::cpp::mesh<view_index, float, 3, 3>(
      held.polygons.faces(), held.polygons.points(), held.cache, held.frame(),
      owned);
  owned.reset();
  CHECK(retained.number_of_faces() == 2);
  CHECK(retained.form()[0][1][0] == 1.0F);
}

TEST_CASE("a topology form publishes the structures a walk needs",
          "[cpp][core][view]") {
  const auto owned = view_triangles();
  const auto mesh = owned.mesh();
  const auto form = mesh.topology_form();
  CHECK(form.face_membership().size() == 4);
  CHECK(form.manifold_edge_link().size() == 2);
  CHECK(owned.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(owned.cache.is_manifold_edge_link_fresh(mesh.geometry()));
}

TEST_CASE("a cache answers for the reading it is given", "[cpp][core][cache]") {
  const auto owned = view_triangles();
  auto &cache = owned.cache;
  const auto &polygons = owned.polygons;

  const auto first = owned.mesh();
  CHECK(cache.face_membership_handle(first.geometry()).size() == 4);
  CHECK(cache.face_membership_build_count() == 1);
  CHECK(cache.is_face_membership_fresh(first.geometry()));

  // the same reading is answered from what is already there
  CHECK(cache.face_membership_handle(first.geometry()).size() == 4);
  CHECK(cache.face_membership_build_count() == 1);

  // the caller states that points moved: a different reading for the tree, and
  // the same one for a membership that only counts them
  cache.points_changed();
  const auto moved = owned.mesh();
  CHECK(cache.is_face_membership_fresh(moved.geometry()));
  CHECK_FALSE(cache.is_tree_fresh(moved.geometry()));

  // a point set of another size is a different reading for both, and whose
  // arrays it stands on is not the cache's question: it owns no geometry
  const auto smaller = tf::cpp::test::polygons_of<view_index, float>(
      {0, 1, 2}, {0, 0, 0, 1, 0, 0, 1, 1, 0});
  cache.faces_changed();
  cache.points_changed();
  const auto fewer = tf::cpp::mesh<view_index, float, 3, 3>(
      smaller.faces(), smaller.points(), cache);
  CHECK_FALSE(cache.is_face_membership_fresh(fewer.geometry()));
  CHECK(cache.face_membership_handle(fewer.geometry()).size() == 3);
  CHECK(cache.face_membership_build_count() == 2);

  cache.faces_changed();
  cache.points_changed();
  const auto again = tf::cpp::mesh<view_index, float, 3, 3>(
      polygons.faces(), polygons.points(), cache);
  CHECK(cache.face_membership_handle(again.geometry()).size() == 4);
  CHECK(cache.face_membership_build_count() == 3);
}

TEST_CASE("a mesh is assembled from memory this layer never allocated",
          "[cpp][core][view][foreign]") {
  // a source that hands out pointers and states its own changes, which is all
  // a foreign owner ever is
  struct foreign_source {
    std::vector<view_index> faces{0, 1, 2, 0, 2, 3};
    std::vector<float> points{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  };
  auto source = std::make_shared<foreign_source>();

  // THE ASSEMBLY: views into the caller's memory, the caller's cache, and the
  // handle that keeps the source alive for a reading that outlives it
  tf::cpp::cache<view_index, float, 3, 3> cache;
  const auto mesh = tf::cpp::mesh<view_index, float, 3, 3>(
      tf::make_faces<3>(
          tf::make_range(static_cast<const view_index *>(source->faces.data()),
                         source->faces.size())),
      tf::make_points<3>(
          tf::make_range(static_cast<const float *>(source->points.data()),
                         source->points.size())),
      cache, tf::cpp::identity_transformation_view<float, 3>(), source);

  // nothing was copied: the mesh reads the source's own arrays
  CHECK(&mesh.faces()[0][0] == source->faces.data());
  CHECK(&mesh.points()[0][0] == source->points.data());
  CHECK(mesh.number_of_faces() == 2);
  CHECK(mesh.topology_form().face_membership().size() == 4);
  CHECK(cache.face_membership_build_count() == 1);
  CHECK(mesh.tree().bv().max[0] == 1.0F);

  // and the source outlives the caller's own handle to it
  source.reset();
  CHECK(mesh.form()[0][1][0] == 1.0F);
}

TEST_CASE("N meshes over one geometry and one cache share every structure",
          "[cpp][core][view][assembly]") {
  // INSTANCING: a cache is built in local coordinates and the frame is a tag,
  // so instances of one geometry differ only in where they are placed
  const auto owned = view_triangles();
  const auto &polygons = owned.polygons;
  auto &cache = owned.cache;

  const auto placement = view_translation(5);
  const auto authored = tf::cpp::mesh<view_index, float, 3, 3>(
      polygons.faces(), polygons.points(), cache);
  const auto placed = tf::cpp::mesh<view_index, float, 3, 3>(
      polygons.faces(), polygons.points(), cache,
      tf::make_transformation_view<3>(placement.data()));

  CHECK(authored.frame()(0, 3) == 0.0F);
  CHECK(placed.frame()(0, 3) == 5.0F);
  CHECK(&placed.tree() == &authored.tree());
  CHECK(cache.tree_build_count() == 1);
  CHECK(placed.face_membership().size() == authored.face_membership().size());
  CHECK(cache.face_membership_build_count() == 1);
}

TEST_CASE("a mesh assembled before a stated change is a reading of its own",
          "[cpp][core][view][assembly]") {
  // the mesh snapshots the generations, so a caller that states a change
  // assembles again — and the cache answers each reading for what it is
  const auto owned = view_triangles();
  auto &cache = owned.cache;

  const auto before = owned.mesh();
  before.tree();
  CHECK(cache.tree_build_count() == 1);
  CHECK(cache.is_tree_fresh(before.geometry()));

  // the reading it was built for is still the reading it answers for; what
  // moved is what the cache is at now, which the next assembly snapshots
  cache.points_changed();
  const auto after = owned.mesh();
  CHECK_FALSE(cache.is_tree_fresh(after.geometry()));
  after.tree();
  CHECK(cache.tree_build_count() == 2);
  CHECK(cache.is_tree_fresh(after.geometry()));
}

TEST_CASE("an edge mesh and a point cloud publish the same shape",
          "[cpp][core][view]") {
  const tf::cpp::test::owned_edge_mesh<view_index, float> edges{
      tf::cpp::test::segments_of<view_index, float>(
          {0, 1, 1, 2}, {0, 0, 0, 1, 0, 0, 2, 0, 0})};
  const auto edge_mesh = edges.edge_mesh();
  CHECK(edge_mesh.number_of_edges() == 2);
  CHECK(edge_mesh.segments()[1][1][0] == 2.0F);
  STATIC_REQUIRE(tf::has_frame_policy<decltype(edge_mesh.form())>);
  CHECK(edge_mesh.topology_form().edge_membership().size() == 3);

  const tf::cpp::test::owned_point_cloud<float> points{
      tf::cpp::test::points_of<float>({0, 0, 0, 3, 0, 0})};
  const auto cloud = points.point_cloud();
  CHECK(cloud.number_of_points() == 2);
  STATIC_REQUIRE(tf::has_frame_policy<decltype(cloud.form())>);
  CHECK(cloud.form().tree().bv().max[0] == 3.0F);
}
