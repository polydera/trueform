/*
 * Point cloud k-NN verification test
 * Compares tree-based k-NN search against brute-force results
 */
#include <trueform/trueform.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef TRUEFORM_DATA_DIR
#define TRUEFORM_DATA_DIR "."
#endif

int main() {
  const std::string data_dir =
      std::string(TRUEFORM_DATA_DIR) + "/benchmarks/data/";
  std::string mesh_path = data_dir + "dragon-50k.stl";

  std::cout << "Loading mesh: " << mesh_path << std::endl;

  auto mesh = tf::read_stl<int>(mesh_path.c_str());
  if (mesh.polygons().size() == 0) {
    std::cerr << "Failed to load mesh" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << mesh.points().size() << " points" << std::endl;

  // Compute AABB
  auto aabb = tf::aabb_from(mesh.polygons());
  auto diagonal = tf::distance(aabb.min, aabb.max);
  auto center = aabb.center();

  std::cout << "AABB: [" << aabb.min[0] << ", " << aabb.min[1] << ", "
            << aabb.min[2] << "] - [" << aabb.max[0] << ", " << aabb.max[1]
            << ", " << aabb.max[2] << "]" << std::endl;
  std::cout << "Diagonal: " << diagonal << std::endl;

  // Pick a query point well outside the AABB
  tf::point<float, 3> query_point =
      center + tf::random_vector<float, 3>() * diagonal * 2;

  std::cout << "\nQuery point: [" << query_point[0] << ", " << query_point[1]
            << ", " << query_point[2] << "]" << std::endl;

  // Build tree
  tf::aabb_tree<int, float, 3> tree(mesh.points(), tf::config_tree(4, 4));
  auto points_with_tree = mesh.points() | tf::tag(tree);

  // k-NN parameters
  constexpr std::size_t k = 10;

  // Tree-based k-NN search
  std::array<tf::nearest_neighbor<int, float, 3>, k> knn_buffer;
  auto knn = tf::make_nearest_neighbors(knn_buffer.begin(), k);
  tf::neighbor_search(points_with_tree, query_point, knn);

  std::cout << "\n=== Tree-based k-NN results ===" << std::endl;
  std::vector<std::pair<float, int>> tree_results;
  for (const auto &neighbor : knn) {
    float dist2 = neighbor.metric();
    tree_results.push_back({dist2, neighbor.element});
    std::cout << "  id=" << neighbor.element << ", dist2=" << dist2
              << ", point=[" << neighbor.info.point[0] << ", "
              << neighbor.info.point[1] << ", " << neighbor.info.point[2] << "]"
              << std::endl;
  }

  // Brute-force k-NN
  std::cout << "\n=== Brute-force k-NN results ===" << std::endl;
  std::vector<std::pair<float, int>> brute_results;
  brute_results.reserve(mesh.points().size());

  for (std::size_t i = 0; i < mesh.points().size(); ++i) {
    float dist2 = tf::distance2(mesh.points()[i], query_point);
    brute_results.push_back({dist2, static_cast<int>(i)});
  }

  // Sort by distance
  std::sort(brute_results.begin(), brute_results.end());

  for (std::size_t i = 0; i < k && i < brute_results.size(); ++i) {
    int id = brute_results[i].second;
    float dist2 = brute_results[i].first;
    const auto &pt = mesh.points()[id];
    std::cout << "  id=" << id << ", dist2=" << dist2 << ", point=[" << pt[0]
              << ", " << pt[1] << ", " << pt[2] << "]" << std::endl;
  }

  // Compare results
  std::cout << "\n=== Comparison ===" << std::endl;
  bool all_match = true;
  for (std::size_t i = 0; i < k; ++i) {
    float tree_dist = tree_results[i].first;
    float brute_dist = brute_results[i].first;
    float diff = std::abs(tree_dist - brute_dist);

    bool match = diff < 1e-5f;
    if (!match) {
      all_match = false;
      std::cout << "  [" << i << "] MISMATCH: tree=" << tree_dist
                << ", brute=" << brute_dist << ", diff=" << diff << std::endl;
    }
  }

  if (all_match) {
    std::cout << "All " << k << " nearest neighbors match!" << std::endl;
    return 0;
  } else {
    std::cout << "FAILED: Some neighbors don't match" << std::endl;
    return 1;
  }
}
