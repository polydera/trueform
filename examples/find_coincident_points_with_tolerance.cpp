#include <algorithm>
#include <iostream>
#include <vector>

// Assuming a convenience header for the core library and spatial queries.
#include "trueform/trueform.hpp"

// Helper function to print the results in a standardized way.
void print_results(const std::string &method_name,
                   const std::vector<std::pair<int, int>> &pairs) {
  std::cout << "\n--- Results from " << method_name << " ---" << std::endl;
  std::cout << "Found " << pairs.size()
            << " coincident pairs:" << std::endl;
  for (const auto &pair : pairs) {
    std::cout << "  - Pair: (" << pair.first << ", " << pair.second << ")"
              << std::endl;
  }
}

int main() {
  // This example demonstrates a common geometry processing workflow:
  // identifying and merging nearly-coincident points in a point cloud. This is
  // a crucial step for cleaning up noisy data or preparing a mesh for
  // operations that require a "welded" manifold topology.

  // --- 1. Generate and Pre-process Data ---
  const size_t num_points = 1000;
  std::cout << "Generating " << num_points << " random points..." << std::endl;
  std::vector<tf::point<float, 3>> points;
  points.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points.push_back(tf::random_point<float, 3>());
  }

  // First, remove any exact duplicates.
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  std::cout << "Point count after removing exact duplicates: " << points.size()
            << std::endl;

  // --- 2. Artificially Create Near-Duplicates for Demonstration ---
  if (!points.empty()) {
    const float epsilon = 1e-5f;
    const int num_duplicates_to_create = 10;

    std::cout << "\nCreating " << num_duplicates_to_create
              << " near-duplicates with an epsilon of " << epsilon << "..."
              << std::endl;
    for (int i = 0; i < num_duplicates_to_create; ++i) {
      const size_t idx = tf::random(size_t{0}, points.size() - 1);
      auto new_pt = points[idx];
      // Shift the duplicated point slightly in a random direction.
      new_pt += tf::normalized(tf::random_vector<float, 3>()) * epsilon;
      points.push_back(new_pt);
    }
  }
  std::cout << "Final point count for searching: " << points.size()
            << std::endl;

  // --- 3. Find All Coincident Points Using trueform ---
  const float tolerance = 2e-5f;
  const float tolerance2 =
      tolerance * tolerance; // Use squared distance for performance

  // Build the acceleration structure and query form once.
  auto final_points_view = tf::make_points(points);
  tf::aabb_tree<int, float, 3> point_tree(final_points_view, tf::config_tree(4, 4));
  auto form = tf::make_form(point_tree, final_points_view);

  // --- Approach 4.1: The High-Level Approach with `gather_self_ids` ---
  std::vector<std::pair<int, int>> coincident_pairs_high_level;
  tf::gather_self_ids(
      form,
      // Broad-phase: check if AABBs are close enough.
      [tolerance](const auto &aabb1, const auto &aabb2) {
        return tf::distance(aabb1, aabb2) < tolerance;
      },
      // Narrow-phase: check the precise distance.
      [tolerance2](const auto &p1, const auto &p2) {
        return tf::distance2(p1, p2) < tolerance2;
      },
      std::back_inserter(coincident_pairs_high_level));

  print_results("gather_self_ids", coincident_pairs_high_level);

  // --- Approach 4.2: The General-Purpose Approach with `search_self` ---
  // This approach is more verbose but offers more flexibility, as the lambda
  // can perform any action, not just collecting IDs.
  tf::local_vector<std::pair<int, int>> coincident_pairs_low_level;
  tf::search_self(
      form,
      [tolerance](const auto &aabb1, const auto &aabb2) {
        return tf::distance(aabb1, aabb2) < tolerance;
      },
      // The lambda can perform any action. Here, we collect pairs.
      [&](const auto &p1, const auto &p2) {
        if (tf::distance2(p1, p2) < tolerance2) {
          coincident_pairs_low_level.push_back({p1.id(), p2.id()});
        }
      });

  print_results("search_self", coincident_pairs_low_level.to_vector());

  return 0;
}

