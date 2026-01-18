/*
 * mod_tree test - verifies raycast and neighbor_search work correctly
 * after boolean operations with stitched mod_tree
 */
#ifndef TRUEFORM_DATA_DIR
#define TRUEFORM_DATA_DIR "."
#endif

#include <iostream>
#include <trueform/trueform.hpp>

struct test_result {
  bool passed = true;
  std::string failures;

  auto fail(const std::string &msg) -> void {
    passed = false;
    failures += "  " + msg + "\n";
  }
};

// Test raycast: shoot ray at polygon centroid, verify we hit that polygon
template <typename Polygons, typename Tree>
auto test_raycast(const Polygons &polygons, const Tree &tree, int polygon_id,
                  const char *tree_name) -> test_result {
  test_result result;

  auto poly = polygons[polygon_id];
  auto centroid = tf::centroid(poly);
  auto normal = tf::make_normal(poly);

  float offset = 0.01f;
  auto ray = tf::make_ray(centroid + offset * normal, -normal);

  auto form = polygons | tf::tag(tree);
  auto info = tf::ray_cast(ray, form);

  if (!info) {
    result.fail(std::string(tree_name) + " raycast to polygon " +
                std::to_string(polygon_id) + ": MISS");
  } else if (info.element != polygon_id) {
    result.fail(std::string(tree_name) + " raycast to polygon " +
                std::to_string(polygon_id) + ": hit " +
                std::to_string(info.element) + " instead");
  }

  return result;
}

// Test neighbor_search: query at polygon centroid, verify polygon is in results
template <typename Polygons, typename Tree>
auto test_neighbor_search(const Polygons &polygons, const Tree &tree,
                          int polygon_id, const char *tree_name)
    -> test_result {
  test_result result;

  auto poly = polygons[polygon_id];
  auto centroid = tf::centroid(poly);

  auto form = polygons | tf::tag(tree);
  auto nearest = tf::neighbor_search(form, centroid);

  if (!nearest) {
    result.fail(std::string(tree_name) + " neighbor_search for polygon " +
                std::to_string(polygon_id) + ": no result");
  } else if (nearest.element != polygon_id) {
    // Check if we at least got a polygon whose centroid matches
    auto hit_centroid = tf::centroid(polygons[nearest.element]);
    auto dist_expected = tf::distance(centroid, tf::centroid(poly));
    auto dist_got = tf::distance(centroid, hit_centroid);
    // Allow if distance is essentially the same (coplanar faces)
    if (dist_got > dist_expected + tf::epsilon<float>) {
      result.fail(std::string(tree_name) + " neighbor_search for polygon " +
                  std::to_string(polygon_id) + ": got " +
                  std::to_string(nearest.element) + " instead");
    }
  }

  return result;
}

auto main() -> int {
  bool all_passed = true;

  // Load mesh
  auto mesh =
      tf::read_stl<int>(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");

  // Build topology for mesh
  tf::face_membership<int> fm;
  fm.build(mesh.polygons());
  tf::manifold_edge_link<int, 3> mel;
  mel.build(mesh.faces(), fm);

  // Start with a mod_aabb_tree
  tf::mod_tree<int, tf::aabb<float, 3>> mod_tree_input;
  mod_tree_input.build(mesh.polygons(), tf::config_tree(4, 4));

  // Create sphere for boolean
  auto sphere = tf::make_sphere_mesh(
      tf::aabb_from(mesh.points()).diagonal().length() / 10, 20, 20);

  // Build topology for sphere
  tf::face_membership<int> fm1;
  fm1.build(sphere.polygons());
  tf::manifold_edge_link<int, 3> mel1;
  mel1.build(sphere.faces(), fm1);
  tf::aabb_tree<int, float, 3> tree1(sphere.polygons(), tf::config_tree(4, 4));

  // Position sphere at last point of mesh
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      mesh.points().back().as_vector()));

  // Do the boolean
  auto [result, labels, index_maps] = tf::make_boolean(
      mesh.polygons() | tf::tag(fm) | tf::tag(mel) | tf::tag(mod_tree_input),
      sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) |
          tf::tag(frame),
      tf::boolean_op::left_difference, tf::return_index_map);

  // Stitch the mod_tree
  auto fm_res = tf::stitched_face_membership(
      result.faces(), int(result.points().size()), fm, fm1, index_maps);
  auto mel_res = tf::stitched_manifold_edge_link(result.faces(), mel, mel1,
                                                 fm_res, index_maps);
  tf::stitch_mod_tree(result.polygons(), mod_tree_input, tf::none, index_maps,
                      tf::config_tree(4, 4));

  // Get test polygon IDs from main and delta trees
  auto main_id = mod_tree_input.main_tree().ids()[0];
  auto delta_id = mod_tree_input.delta_tree().ids()[0];

  // Build regular tree for comparison
  tf::aabb_tree<int, float, 3> regular_tree(result.polygons(),
                                            tf::config_tree(4, 4));

  // Build fresh mod_tree for comparison
  tf::mod_tree<int, tf::aabb<float, 3>> fresh_mod_tree;
  fresh_mod_tree.build(result.polygons(), tf::config_tree(4, 4));

  // === Raycast tests ===
  auto check_raycast = [&, &result = result](int poly_id,
                                             const char *poly_name) -> void {
    auto r1 =
        test_raycast(result.polygons(), regular_tree, poly_id, "regular_tree");
    auto r2 = test_raycast(result.polygons(), mod_tree_input, poly_id,
                           "stitched_mod_tree");
    auto r3 = test_raycast(result.polygons(), fresh_mod_tree, poly_id,
                           "fresh_mod_tree");

    if (!r1.passed || !r2.passed || !r3.passed) {
      std::cout << "FAIL raycast " << poly_name << " (id=" << poly_id << "):\n";
      if (!r1.passed)
        std::cout << r1.failures;
      if (!r2.passed)
        std::cout << r2.failures;
      if (!r3.passed)
        std::cout << r3.failures;
      all_passed = false;
    }
  };

  check_raycast(main_id, "main_tree_polygon");
  check_raycast(delta_id, "delta_tree_polygon");

  // === Neighbor search tests ===
  auto check_neighbor = [&, &result=result](int poly_id, const char *poly_name) -> void {
    auto r1 = test_neighbor_search(result.polygons(), regular_tree, poly_id,
                                   "regular_tree");
    auto r2 = test_neighbor_search(result.polygons(), mod_tree_input, poly_id,
                                   "stitched_mod_tree");
    auto r3 = test_neighbor_search(result.polygons(), fresh_mod_tree, poly_id,
                                   "fresh_mod_tree");

    if (!r1.passed || !r2.passed || !r3.passed) {
      std::cout << "FAIL neighbor_search " << poly_name << " (id=" << poly_id
                << "):\n";
      if (!r1.passed)
        std::cout << r1.failures;
      if (!r2.passed)
        std::cout << r2.failures;
      if (!r3.passed)
        std::cout << r3.failures;
      all_passed = false;
    }
  };

  check_neighbor(main_id, "main_tree_polygon");
  check_neighbor(delta_id, "delta_tree_polygon");

  if (all_passed) {
    std::cout << "All mod_tree tests passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
