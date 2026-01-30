/**
 * @file test_mod_tree_queries.cpp
 * @brief Comprehensive tests for mod_tree spatial query operations
 *
 * Tests all spatial operations with mod_tree, verifying correct behavior when
 * query results come from:
 * - Dirty region (delta tree) - elements that were recently updated
 * - Non-dirty region (main tree) - elements that remain unchanged
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>
#include <mutex>
#include <set>
#include <vector>

namespace {

// =============================================================================
// Helper: Create a split mesh with clear left/right regions
// =============================================================================

/**
 * @brief Creates a grid mesh split into left and right halves
 *
 * The mesh is a grid in XY plane. Left half (x < midpoint) will remain
 * unchanged (main tree), right half (x >= midpoint) will be modified (delta
 * tree).
 *
 * @return Mesh with 8x4 grid = 56 triangles (28 left, 28 right)
 */
template <typename index_t, typename real_t>
auto create_split_mesh() -> tf::polygons_buffer<index_t, real_t, 3, 3> {
  tf::polygons_buffer<index_t, real_t, 3, 3> result;

  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 5;

  // Create vertices
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      result.points_buffer().emplace_back(static_cast<real_t>(i),
                                          static_cast<real_t>(j), real_t(0));
    }
  }

  // Create triangles (2 per grid cell)
  for (std::size_t j = 0; j < ny - 1; ++j) {
    for (std::size_t i = 0; i < nx - 1; ++i) {
      auto v00 = static_cast<index_t>(j * nx + i);
      auto v10 = static_cast<index_t>(j * nx + i + 1);
      auto v01 = static_cast<index_t>((j + 1) * nx + i);
      auto v11 = static_cast<index_t>((j + 1) * nx + i + 1);

      // Lower-left triangle
      result.faces_buffer().emplace_back(v00, v10, v01);
      // Upper-right triangle
      result.faces_buffer().emplace_back(v10, v11, v01);
    }
  }

  return result;
}

/**
 * @brief Get polygon IDs for the right half of the mesh (x >= midpoint)
 */
template <typename index_t, typename real_t, typename Mesh>
auto get_right_half_ids(const Mesh &mesh) -> std::vector<index_t> {
  std::vector<index_t> ids;
  real_t midpoint = real_t(4); // Half of 8-cell width

  for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
    auto poly = mesh.polygons()[i];
    auto centroid = tf::centroid(poly);
    if (centroid[0] >= midpoint) {
      ids.push_back(static_cast<index_t>(i));
    }
  }

  return ids;
}

/**
 * @brief Get polygon IDs for the left half of the mesh (x < midpoint)
 */
template <typename index_t, typename real_t, typename Mesh>
auto get_left_half_ids(const Mesh &mesh) -> std::vector<index_t> {
  std::vector<index_t> ids;
  real_t midpoint = real_t(4);

  for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
    auto poly = mesh.polygons()[i];
    auto centroid = tf::centroid(poly);
    if (centroid[0] < midpoint) {
      ids.push_back(static_cast<index_t>(i));
    }
  }

  return ids;
}

/**
 * @brief Slightly modify vertices of the right half polygons
 *
 * This perturbs the z-coordinate of vertices in the right half to create
 * actual geometry changes that necessitate tree updates.
 *
 * @return Set of modified vertex IDs
 */
template <typename index_t, typename real_t, typename Mesh>
auto modify_right_half(Mesh &mesh, const std::vector<index_t> &right_ids,
                       real_t perturbation = real_t(0.1)) -> std::set<index_t> {
  std::set<index_t> modified_verts;

  // Collect vertices from right half polygons
  for (auto poly_id : right_ids) {
    auto face = mesh.faces()[poly_id];
    for (auto vid : face) {
      modified_verts.insert(vid);
    }
  }

  // Modify z-coordinate of collected vertices
  for (auto vid : modified_verts) {
    mesh.points()[vid][2] += perturbation;
  }

  return modified_verts;
}

/**
 * @brief Get all polygon IDs that have at least one vertex in the modified set
 *
 * This is the correct way to determine dirty polygons - any polygon with a
 * modified vertex must be in the delta tree, not just polygons by centroid.
 */
template <typename index_t, typename Mesh>
auto get_dirty_polygon_ids(const Mesh &mesh, const std::set<index_t> &modified_verts)
    -> std::vector<index_t> {
  std::vector<index_t> dirty_ids;

  for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
    auto face = mesh.faces()[i];
    for (auto vid : face) {
      if (modified_verts.count(vid)) {
        dirty_ids.push_back(static_cast<index_t>(i));
        break;
      }
    }
  }

  return dirty_ids;
}

} // namespace

// =============================================================================
// Test 1: distance - result in dirty vs non-dirty region
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_distance_queries", "[mod_tree][distance]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // Create mesh split into left/right halves
  auto mesh = create_split_mesh<index_t, real_t>();

  // Build mod_tree
  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  // Get right half IDs and modify them
  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);

  // Get ALL polygons with at least one modified vertex (not just right half by centroid)
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  // Create keep_if predicate
  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  // Update tree with dirty IDs
  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  // Build fresh reference tree
  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("distance to point - result in dirty region") {
    // Query point in right half (dirty region)
    auto query_pt = tf::make_point(real_t(6), real_t(2), real_t(0.1));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto dist_mod = tf::distance(form_mod, query_pt);
    auto dist_ref = tf::distance(form_ref, query_pt);

    REQUIRE(std::abs(dist_mod - dist_ref) < tf::epsilon<real_t>);
  }

  SECTION("distance to point - result in non-dirty region") {
    // Query point in left half (non-dirty region)
    auto query_pt = tf::make_point(real_t(1), real_t(2), real_t(0.1));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto dist_mod = tf::distance(form_mod, query_pt);
    auto dist_ref = tf::distance(form_ref, query_pt);

    REQUIRE(std::abs(dist_mod - dist_ref) < tf::epsilon<real_t>);
  }

  SECTION("distance2 to point - result in dirty region") {
    auto query_pt = tf::make_point(real_t(6), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto dist2_mod = tf::distance2(form_mod, query_pt);
    auto dist2_ref = tf::distance2(form_ref, query_pt);

    REQUIRE(std::abs(dist2_mod - dist2_ref) < tf::epsilon<real_t>);
  }

  SECTION("distance2 to point - result in non-dirty region") {
    auto query_pt = tf::make_point(real_t(1), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto dist2_mod = tf::distance2(form_mod, query_pt);
    auto dist2_ref = tf::distance2(form_ref, query_pt);

    REQUIRE(std::abs(dist2_mod - dist2_ref) < tf::epsilon<real_t>);
  }
}

// =============================================================================
// Test 2: intersects - result in dirty vs non-dirty region
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_intersects_queries", "[mod_tree][intersects]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("intersects with AABB - in dirty region") {
    // AABB in right half
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(5), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(7), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    bool intersects_mod = tf::intersects(form_mod, query_aabb);
    bool intersects_ref = tf::intersects(form_ref, query_aabb);

    REQUIRE(intersects_mod == intersects_ref);
    REQUIRE(intersects_mod == true);
  }

  SECTION("intersects with AABB - in non-dirty region") {
    // AABB in left half
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(1), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(3), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    bool intersects_mod = tf::intersects(form_mod, query_aabb);
    bool intersects_ref = tf::intersects(form_ref, query_aabb);

    REQUIRE(intersects_mod == intersects_ref);
    REQUIRE(intersects_mod == true);
  }

  SECTION("intersects with point - in dirty region") {
    // Point on right half polygon (after modification, z is raised)
    auto query_pt = tf::make_point(real_t(6), real_t(2), real_t(0.1));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    bool intersects_mod = tf::intersects(form_mod, query_pt);
    bool intersects_ref = tf::intersects(form_ref, query_pt);

    REQUIRE(intersects_mod == intersects_ref);
  }

  SECTION("intersects with point - in non-dirty region") {
    // Point on left half polygon (z = 0)
    auto query_pt = tf::make_point(real_t(1.5), real_t(1.5), real_t(0));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    bool intersects_mod = tf::intersects(form_mod, query_pt);
    bool intersects_ref = tf::intersects(form_ref, query_pt);

    REQUIRE(intersects_mod == intersects_ref);
    REQUIRE(intersects_mod == true);
  }
}

// =============================================================================
// Test 3: neighbor_search - result in dirty vs non-dirty region
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_neighbor_search_queries",
                   "[mod_tree][neighbor_search]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("neighbor_search - result in dirty region") {
    // Query near right half
    auto query_pt = tf::make_point(real_t(6), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
    REQUIRE(nearest_mod.element == nearest_ref.element);
  }

  SECTION("neighbor_search - result in non-dirty region") {
    // Query near left half
    auto query_pt = tf::make_point(real_t(1), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
    REQUIRE(nearest_mod.element == nearest_ref.element);
  }

  SECTION("neighbor_search with radius - result in dirty region") {
    auto query_pt = tf::make_point(real_t(6), real_t(2), real_t(0.2));
    real_t max_radius = real_t(1.0);

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt, max_radius);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt, max_radius);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
  }

  SECTION("neighbor_search with radius - result in non-dirty region") {
    auto query_pt = tf::make_point(real_t(1), real_t(2), real_t(0.2));
    real_t max_radius = real_t(1.0);

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt, max_radius);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt, max_radius);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
  }
}

// =============================================================================
// Test 4: ray_cast - result in dirty vs non-dirty region
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_ray_cast_queries", "[mod_tree][ray_cast]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("ray_cast - hit in dirty region") {
    // Ray pointing down at right half
    auto ray = tf::make_ray(tf::make_point(real_t(6), real_t(2), real_t(1)),
                            tf::make_unit_vector(real_t(0), real_t(0), real_t(-1)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto hit_mod = tf::ray_cast(ray, form_mod);
    auto hit_ref = tf::ray_cast(ray, form_ref);

    REQUIRE(hit_mod);
    REQUIRE(hit_ref);
    REQUIRE(hit_mod.element == hit_ref.element);
    REQUIRE(std::abs(hit_mod.info.t - hit_ref.info.t) < tf::epsilon<real_t>);
  }

  SECTION("ray_cast - hit in non-dirty region") {
    // Ray pointing down at left half (z=0 plane)
    auto ray = tf::make_ray(tf::make_point(real_t(1), real_t(2), real_t(1)),
                            tf::make_unit_vector(real_t(0), real_t(0), real_t(-1)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto hit_mod = tf::ray_cast(ray, form_mod);
    auto hit_ref = tf::ray_cast(ray, form_ref);

    REQUIRE(hit_mod);
    REQUIRE(hit_ref);
    REQUIRE(hit_mod.element == hit_ref.element);
    REQUIRE(std::abs(hit_mod.info.t - hit_ref.info.t) < tf::epsilon<real_t>);
  }

  SECTION("ray_cast with config - hit in dirty region") {
    auto ray = tf::make_ray(tf::make_point(real_t(6), real_t(2), real_t(2)),
                            tf::make_unit_vector(real_t(0), real_t(0), real_t(-1)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto config = tf::make_ray_config(real_t(0), real_t(10));

    auto hit_mod = tf::ray_cast(ray, form_mod, config);
    auto hit_ref = tf::ray_cast(ray, form_ref, config);

    REQUIRE(hit_mod);
    REQUIRE(hit_ref);
    REQUIRE(hit_mod.element == hit_ref.element);
  }
}

// =============================================================================
// Test 5: gather_ids - results from both regions
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_gather_ids_queries", "[mod_tree][gather_ids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("gather_ids with AABB - dirty region only") {
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(5), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(7), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<index_t> ids_mod, ids_ref;
    tf::gather_ids(
        form_mod, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_mod));
    tf::gather_ids(
        form_ref, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_ref));

    std::sort(ids_mod.begin(), ids_mod.end());
    std::sort(ids_ref.begin(), ids_ref.end());

    REQUIRE(ids_mod.size() == ids_ref.size());
    REQUIRE(ids_mod == ids_ref);
  }

  SECTION("gather_ids with AABB - non-dirty region only") {
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(1), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(3), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<index_t> ids_mod, ids_ref;
    tf::gather_ids(
        form_mod, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_mod));
    tf::gather_ids(
        form_ref, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_ref));

    std::sort(ids_mod.begin(), ids_mod.end());
    std::sort(ids_ref.begin(), ids_ref.end());

    REQUIRE(ids_mod.size() == ids_ref.size());
    REQUIRE(ids_mod == ids_ref);
  }

  SECTION("gather_ids with AABB - spanning both regions") {
    // AABB spanning the center (both dirty and non-dirty)
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(3), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(5), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<index_t> ids_mod, ids_ref;
    tf::gather_ids(
        form_mod, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_mod));
    tf::gather_ids(
        form_ref, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
        std::back_inserter(ids_ref));

    std::sort(ids_mod.begin(), ids_mod.end());
    std::sort(ids_ref.begin(), ids_ref.end());

    REQUIRE(ids_mod.size() == ids_ref.size());
    REQUIRE(ids_mod == ids_ref);
  }

  SECTION("gather_ids within distance - spanning both regions") {
    auto query_pt = tf::make_point(real_t(4), real_t(2), real_t(0.5));
    auto dist2 = real_t(4.0);

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<index_t> ids_mod, ids_ref;
    tf::gather_ids(
        form_mod, [&](const auto &bv) { return tf::distance2(bv, query_pt) <= dist2; },
        [&](const auto &prim) { return tf::distance2(prim, query_pt) <= dist2; },
        std::back_inserter(ids_mod));
    tf::gather_ids(
        form_ref, [&](const auto &bv) { return tf::distance2(bv, query_pt) <= dist2; },
        [&](const auto &prim) { return tf::distance2(prim, query_pt) <= dist2; },
        std::back_inserter(ids_ref));

    std::sort(ids_mod.begin(), ids_mod.end());
    std::sort(ids_ref.begin(), ids_ref.end());

    REQUIRE(ids_mod.size() == ids_ref.size());
    REQUIRE(ids_mod == ids_ref);
  }
}

// =============================================================================
// Test 6: gather_self_ids
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_gather_self_ids_queries",
                   "[mod_tree][gather_self_ids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("gather_self_ids - intersecting pairs") {
    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<std::pair<index_t, index_t>> pairs_mod, pairs_ref;
    tf::gather_self_ids(form_mod, tf::intersects_f, std::back_inserter(pairs_mod));
    tf::gather_self_ids(form_ref, tf::intersects_f, std::back_inserter(pairs_ref));

    // Normalize pairs (smaller id first)
    auto normalize = [](std::vector<std::pair<index_t, index_t>> &pairs) {
      for (auto &[i, j] : pairs) {
        if (i > j)
          std::swap(i, j);
      }
      std::sort(pairs.begin(), pairs.end());
    };

    normalize(pairs_mod);
    normalize(pairs_ref);

    REQUIRE(pairs_mod.size() == pairs_ref.size());
    REQUIRE(pairs_mod == pairs_ref);
  }

  SECTION("gather_self_ids - close pairs within distance") {
    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    real_t threshold2 = real_t(0.5);
    auto check_bvs = [threshold2](const auto &bv0, const auto &bv1) {
      return tf::distance2(bv0, bv1) <= threshold2;
    };

    std::vector<std::pair<index_t, index_t>> pairs_mod, pairs_ref;
    tf::gather_self_ids(form_mod, check_bvs, std::back_inserter(pairs_mod));
    tf::gather_self_ids(form_ref, check_bvs, std::back_inserter(pairs_ref));

    // Normalize pairs (smaller id first)
    auto normalize = [](std::vector<std::pair<index_t, index_t>> &pairs) {
      for (auto &[i, j] : pairs) {
        if (i > j)
          std::swap(i, j);
      }
      std::sort(pairs.begin(), pairs.end());
    };

    normalize(pairs_mod);
    normalize(pairs_ref);

    REQUIRE(pairs_mod.size() == pairs_ref.size());
    REQUIRE(pairs_mod == pairs_ref);
  }
}

// =============================================================================
// Test 7: search - custom traversal
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_search_queries", "[mod_tree][search]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("search for first polygon in AABB - dirty region") {
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(5), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(7), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    index_t found_mod = -1, found_ref = -1;

    tf::search(
        form_mod, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) {
          if (tf::intersects(prim, query_aabb)) {
            found_mod = prim.id();
            return true; // Stop search
          }
          return false;
        });

    tf::search(
        form_ref, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) {
          if (tf::intersects(prim, query_aabb)) {
            found_ref = prim.id();
            return true;
          }
          return false;
        });

    REQUIRE(found_mod >= 0);
    REQUIRE(found_ref >= 0);
    // Both should find a valid polygon (may be different due to traversal order)
    REQUIRE(tf::intersects(mesh.polygons()[found_mod], query_aabb));
    REQUIRE(tf::intersects(mesh.polygons()[found_ref], query_aabb));
  }

  SECTION("search for first polygon in AABB - non-dirty region") {
    auto query_aabb =
        tf::make_aabb(tf::make_point(real_t(1), real_t(1), real_t(-0.5)),
                      tf::make_point(real_t(3), real_t(3), real_t(0.5)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    index_t found_mod = -1, found_ref = -1;

    tf::search(
        form_mod, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) {
          if (tf::intersects(prim, query_aabb)) {
            found_mod = prim.id();
            return true;
          }
          return false;
        });

    tf::search(
        form_ref, [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
        [&](const auto &prim) {
          if (tf::intersects(prim, query_aabb)) {
            found_ref = prim.id();
            return true;
          }
          return false;
        });

    REQUIRE(found_mod >= 0);
    REQUIRE(found_ref >= 0);
    REQUIRE(tf::intersects(mesh.polygons()[found_mod], query_aabb));
    REQUIRE(tf::intersects(mesh.polygons()[found_ref], query_aabb));
  }
}

// =============================================================================
// Test 8: search_self - self-intersection search
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_search_self_queries", "[mod_tree][search_self]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("search_self for intersecting pairs") {
    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<std::pair<index_t, index_t>> pairs_mod, pairs_ref;
    std::mutex mutex_mod, mutex_ref;

    tf::search_self(
        form_mod, tf::intersects_f,
        [&](const auto &prim0, const auto &prim1) {
          if (tf::intersects(prim0, prim1)) {
            std::lock_guard<std::mutex> lock(mutex_mod);
            pairs_mod.emplace_back(prim0.id(), prim1.id());
          }
          return false;
        });

    tf::search_self(
        form_ref, tf::intersects_f,
        [&](const auto &prim0, const auto &prim1) {
          if (tf::intersects(prim0, prim1)) {
            std::lock_guard<std::mutex> lock(mutex_ref);
            pairs_ref.emplace_back(prim0.id(), prim1.id());
          }
          return false;
        });

    // Normalize pairs (smaller id first)
    auto normalize = [](std::vector<std::pair<index_t, index_t>> &pairs) {
      for (auto &[i, j] : pairs) {
        if (i > j)
          std::swap(i, j);
      }
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    };

    normalize(pairs_mod);
    normalize(pairs_ref);

    REQUIRE(pairs_mod.size() == pairs_ref.size());
    REQUIRE(pairs_mod == pairs_ref);
  }

  SECTION("search_self for close pairs") {
    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    real_t threshold2 = real_t(0.5);
    std::vector<std::pair<index_t, index_t>> pairs_mod, pairs_ref;
    std::mutex mutex_mod, mutex_ref;

    auto check_bvs = [threshold2](const auto &bv0, const auto &bv1) {
      return tf::distance2(bv0, bv1) <= threshold2;
    };

    tf::search_self(
        form_mod, check_bvs,
        [&](const auto &prim0, const auto &prim1) {
          if (tf::distance2(prim0, prim1) <= threshold2) {
            std::lock_guard<std::mutex> lock(mutex_mod);
            pairs_mod.emplace_back(prim0.id(), prim1.id());
          }
          return false;
        });

    tf::search_self(
        form_ref, check_bvs,
        [&](const auto &prim0, const auto &prim1) {
          if (tf::distance2(prim0, prim1) <= threshold2) {
            std::lock_guard<std::mutex> lock(mutex_ref);
            pairs_ref.emplace_back(prim0.id(), prim1.id());
          }
          return false;
        });

    // Normalize pairs
    auto normalize = [](std::vector<std::pair<index_t, index_t>> &pairs) {
      for (auto &[i, j] : pairs) {
        if (i > j)
          std::swap(i, j);
      }
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    };

    normalize(pairs_mod);
    normalize(pairs_ref);

    REQUIRE(pairs_mod.size() == pairs_ref.size());
    REQUIRE(pairs_mod == pairs_ref);
  }
}

// =============================================================================
// Test 9: Comprehensive iteration over all dirty/non-dirty polygons
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_comprehensive_region_coverage",
                   "[mod_tree][comprehensive]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);

  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  // Compute non-dirty IDs (complement of dirty_ids)
  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  std::vector<index_t> non_dirty_ids;
  for (std::size_t i = 0; i < mesh.size(); ++i) {
    if (dirty_set.find(static_cast<index_t>(i)) == dirty_set.end()) {
      non_dirty_ids.push_back(static_cast<index_t>(i));
    }
  }

  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("ray_cast to every polygon in dirty region") {
    for (auto poly_id : dirty_ids) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);
      auto normal = tf::make_normal(poly);

      real_t offset = real_t(0.01);
      auto ray = tf::make_ray(centroid + offset * normal, -normal);

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto hit_mod = tf::ray_cast(ray, form_mod);
      auto hit_ref = tf::ray_cast(ray, form_ref);

      REQUIRE(hit_mod);
      REQUIRE(hit_ref);
      REQUIRE(hit_mod.element == hit_ref.element);
    }
  }

  SECTION("ray_cast to every polygon in non-dirty region") {
    for (auto poly_id : non_dirty_ids) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);
      auto normal = tf::make_normal(poly);

      real_t offset = real_t(0.01);
      auto ray = tf::make_ray(centroid + offset * normal, -normal);

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto hit_mod = tf::ray_cast(ray, form_mod);
      auto hit_ref = tf::ray_cast(ray, form_ref);

      REQUIRE(hit_mod);
      REQUIRE(hit_ref);
      REQUIRE(hit_mod.element == hit_ref.element);
    }
  }

  SECTION("neighbor_search from every polygon centroid") {
    for (std::size_t poly_id = 0; poly_id < mesh.size(); ++poly_id) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto nearest_mod = tf::neighbor_search(form_mod, centroid);
      auto nearest_ref = tf::neighbor_search(form_ref, centroid);

      REQUIRE(nearest_mod);
      REQUIRE(nearest_ref);
      REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
              tf::epsilon<real_t>);
    }
  }

  SECTION("distance to every polygon centroid") {
    for (std::size_t poly_id = 0; poly_id < mesh.size(); ++poly_id) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);
      // Offset the query point slightly above the mesh
      auto query_pt = centroid + tf::make_vector(real_t(0), real_t(0), real_t(0.5));

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto dist_mod = tf::distance(form_mod, query_pt);
      auto dist_ref = tf::distance(form_ref, query_pt);

      REQUIRE(std::abs(dist_mod - dist_ref) < tf::epsilon<real_t>);
    }
  }
}

// =============================================================================
// Test 10: Multiple updates - verify consistency through iterations
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_multiple_updates", "[mod_tree][multiple_updates]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  // Perform multiple update cycles
  constexpr int n_iterations = 5;

  for (int iter = 0; iter < n_iterations; ++iter) {
    // Alternate which half's vertices to modify
    std::vector<index_t> region_ids;
    if (iter % 2 == 0) {
      region_ids = get_right_half_ids<index_t, real_t>(mesh);
    } else {
      region_ids = get_left_half_ids<index_t, real_t>(mesh);
    }

    // Collect and modify the vertices
    std::set<index_t> modified_verts;
    for (auto poly_id : region_ids) {
      auto face = mesh.faces()[poly_id];
      for (auto vid : face) {
        modified_verts.insert(vid);
      }
    }

    real_t perturbation = real_t(0.02) * (iter + 1);
    for (auto vid : modified_verts) {
      mesh.points()[vid][2] += perturbation;
    }

    // Get ALL polygons with modified vertices (not just region_ids)
    auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

    // Update tree
    std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
    auto keep_if = [&dirty_set](index_t id) {
      return dirty_set.find(id) == dirty_set.end();
    };

    tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

    // Build fresh reference tree
    tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                               tf::config_tree(4, 4));

    // Verify some sample queries
    for (auto poly_id : dirty_ids) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);
      auto normal = tf::make_normal(poly);

      real_t offset = real_t(0.01);
      auto ray = tf::make_ray(centroid + offset * normal, -normal);

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto hit_mod = tf::ray_cast(ray, form_mod);
      auto hit_ref = tf::ray_cast(ray, form_ref);

      REQUIRE(hit_mod);
      REQUIRE(hit_ref);
      REQUIRE(hit_mod.element == hit_ref.element);
    }

    // Verify neighbor search
    auto query_pt = tf::make_point(real_t(4), real_t(2), real_t(1));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
  }
}

// =============================================================================
// Test 11: Verify main_tree and delta_tree contents
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_main_delta_contents",
                   "[mod_tree][main_delta]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);

  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  SECTION("queries hit correct tree region") {
    // Query point far in left region should be found via main tree
    auto left_query = tf::make_point(real_t(0.5), real_t(2), real_t(0.5));
    auto form = mesh.polygons() | tf::tag(tree);
    auto nearest_left = tf::neighbor_search(form, left_query);
    REQUIRE(nearest_left);
    // Result should be a left-half polygon
    auto result_centroid = tf::centroid(mesh.polygons()[nearest_left.element]);
    REQUIRE(result_centroid[0] < real_t(4));

    // Query point far in right region should be found via delta tree
    auto right_query = tf::make_point(real_t(7.5), real_t(2), real_t(0.5));
    auto nearest_right = tf::neighbor_search(form, right_query);
    REQUIRE(nearest_right);
    // Result should be a right-half polygon
    result_centroid = tf::centroid(mesh.polygons()[nearest_right.element]);
    REQUIRE(result_centroid[0] >= real_t(4));
  }
}

// =============================================================================
// Test 12: Empty delta tree (no update performed)
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_empty_delta", "[mod_tree][empty_delta]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  // Build mod_tree but don't update (empty delta tree)
  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  // Build reference tree
  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("neighbor_search with empty delta") {
    auto query_pt = tf::make_point(real_t(4), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
  }

  SECTION("ray_cast with empty delta") {
    auto ray = tf::make_ray(tf::make_point(real_t(4), real_t(2), real_t(1)),
                            tf::make_unit_vector(real_t(0), real_t(0), real_t(-1)));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto hit_mod = tf::ray_cast(ray, form_mod);
    auto hit_ref = tf::ray_cast(ray, form_ref);

    REQUIRE(hit_mod);
    REQUIRE(hit_ref);
    REQUIRE(hit_mod.element == hit_ref.element);
  }

  SECTION("gather_self_ids with empty delta") {
    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    std::vector<std::pair<index_t, index_t>> pairs_mod, pairs_ref;
    tf::gather_self_ids(form_mod, tf::intersects_f, std::back_inserter(pairs_mod));
    tf::gather_self_ids(form_ref, tf::intersects_f, std::back_inserter(pairs_ref));

    auto normalize = [](std::vector<std::pair<index_t, index_t>> &pairs) {
      for (auto &[i, j] : pairs) {
        if (i > j)
          std::swap(i, j);
      }
      std::sort(pairs.begin(), pairs.end());
    };

    normalize(pairs_mod);
    normalize(pairs_ref);

    REQUIRE(pairs_mod.size() == pairs_ref.size());
    REQUIRE(pairs_mod == pairs_ref);
  }
}

// =============================================================================
// Test 13: Boundary region polygons (centroid on left but dirty due to shared vertices)
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_boundary_region", "[mod_tree][boundary]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto mesh = create_split_mesh<index_t, real_t>();

  tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
  tree.build(mesh.polygons(), tf::config_tree(4, 4));

  auto right_ids = get_right_half_ids<index_t, real_t>(mesh);
  auto modified_verts = modify_right_half<index_t, real_t>(mesh, right_ids);
  auto dirty_ids = get_dirty_polygon_ids<index_t>(mesh, modified_verts);

  // Identify boundary polygons: centroid < 4 but in dirty_ids
  std::set<index_t> dirty_set(dirty_ids.begin(), dirty_ids.end());
  std::vector<index_t> boundary_ids;
  for (auto id : dirty_ids) {
    auto centroid = tf::centroid(mesh.polygons()[id]);
    if (centroid[0] < real_t(4)) {
      boundary_ids.push_back(id);
    }
  }

  auto keep_if = [&dirty_set](index_t id) {
    return dirty_set.find(id) == dirty_set.end();
  };

  tree.update(mesh.polygons(), dirty_ids, keep_if, tf::config_tree(4, 4));

  tf::aabb_tree<index_t, real_t, 3> ref_tree(mesh.polygons(),
                                             tf::config_tree(4, 4));

  SECTION("boundary polygons exist") {
    // There should be boundary polygons (polygons with centroid < 4 but dirty)
    REQUIRE(boundary_ids.size() > 0);
  }

  SECTION("ray_cast to boundary polygons") {
    for (auto poly_id : boundary_ids) {
      auto poly = mesh.polygons()[poly_id];
      auto centroid = tf::centroid(poly);
      auto normal = tf::make_normal(poly);

      real_t offset = real_t(0.01);
      auto ray = tf::make_ray(centroid + offset * normal, -normal);

      auto form_mod = mesh.polygons() | tf::tag(tree);
      auto form_ref = mesh.polygons() | tf::tag(ref_tree);

      auto hit_mod = tf::ray_cast(ray, form_mod);
      auto hit_ref = tf::ray_cast(ray, form_ref);

      REQUIRE(hit_mod);
      REQUIRE(hit_ref);
      REQUIRE(hit_mod.element == hit_ref.element);
    }
  }

  SECTION("neighbor_search near boundary") {
    // Query at the boundary x=4, should find results from both trees
    auto query_pt = tf::make_point(real_t(4), real_t(2), real_t(0.5));

    auto form_mod = mesh.polygons() | tf::tag(tree);
    auto form_ref = mesh.polygons() | tf::tag(ref_tree);

    auto nearest_mod = tf::neighbor_search(form_mod, query_pt);
    auto nearest_ref = tf::neighbor_search(form_ref, query_pt);

    REQUIRE(nearest_mod);
    REQUIRE(nearest_ref);
    REQUIRE(std::abs(nearest_mod.metric() - nearest_ref.metric()) <
            tf::epsilon<real_t>);
  }
}
