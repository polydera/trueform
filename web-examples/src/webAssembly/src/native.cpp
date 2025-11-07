#include "native.h"
#include <emscripten/bind.h>
int add(int a, int b) {
    return a + b;
}
#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
// Assuming a convenience header for the core library and spatial queries.
#include "trueform/trueform.hpp"

// Helper function to print the results in a standardized way.
void print_results(const std::string &method_name,
                   const std::vector<std::pair<int, int>> &pairs) {

  std::cout << "\n--- Results from " << method_name << " ---" << std::endl;
  std::cout << "Found " << pairs.size()
            << " intersecting pairs:" << std::endl;
  for (const auto &pair : pairs) {
    std::cout << "  - Original Triangle " << pair.first
              << " intersects Transformed Triangle " << pair.second
              << std::endl;
  }
}

int p() {
  // This example demonstrates finding all intersecting triangles between a mesh
  // and a transformed version of itself. This is a common task in collision
  // detection for articulated bodies or mesh self-intersection tests.

  // --- 1. Define a simple mesh (a 2x2 quad) ---
  std::vector<tf::point<float, 3>> points = {
      {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f},
      {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}, {2.f, 1.f, 0.f},
      {0.f, 2.f, 0.f}, {1.f, 2.f, 0.f}, {2.f, 2.f, 0.f}};

  std::vector<int> faces = {
      0, 1, 4, 0, 4, 3, // Quad 1
      1, 2, 5, 1, 5, 4, // Quad 2
      3, 4, 7, 3, 7, 6, // Quad 3
      4, 5, 8, 4, 8, 7  // Quad 4
  };

  // --- 2. Create the trueform semantic view ---
  // We create a single `polygons` object representing our mesh.
  auto mesh_polygons = tf::make_polygons(tf::make_blocked_range<3>(faces),
                                         tf::make_points(points));

  // --- 3. Build the acceleration structure ---
  // The tree is built once over the static geometry.
  tf::tree<int, float, 3> mesh_tree(mesh_polygons, tf::config_tree(4, 4));

  // --- 4. Define a transformation ---
  // We'll create a frame that rotates the mesh 90 degrees around the Y-axis
  // and moves it slightly, causing it to self-intersect.
  tf::transformation<float, 3> rotation{0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0};
  tf::transformation<float, 3> translation =
      tf::make_transformation_from_translation(
          tf::vector<float, 3>{1.5f, 1.5f, 0.5f});

  auto transform = tf::transformed(rotation, translation);
  auto frame = tf::make_frame(transform);

  // --- 5. Perform the Intersection Query ---
  // We will query the mesh against a transformed version of itself.

  // Form A: The original, untransformed mesh.
  auto form_A = tf::make_form(mesh_tree, mesh_polygons);

  // Form B: The same mesh and tree, but with the dynamic frame applied
  // on-the-fly during the query.
  auto form_B = tf::make_form(frame, mesh_tree, mesh_polygons);

  std::cout << "Finding all intersecting triangle pairs between the mesh and a "
            << "transformed copy...\n";

  // --- Approach 5.1: The High-Level Approach with `gather_ids` ---
  std::vector<std::pair<int, int>> intersecting_pairs_high_level;
  tf::gather_ids(form_A, form_B,
                 tf::intersects_f, // The intersection predicate
                 std::back_inserter(intersecting_pairs_high_level));

  print_results("gather_ids", intersecting_pairs_high_level);

  // --- Approach 5.2: The General-Purpose Approach with `search` ---
  // This approach is more verbose but offers more flexibility, as the lambda
  // can perform any action, not just collecting IDs.
  tf::local_vector<std::pair<int, int>> intersecting_pairs_low_level;
  tf::search(form_A, form_B,
             // Broad-phase: check if the AABBs of the tree nodes intersect.
             tf::intersects_f,
             // Narrow-phase: for each potential pair, do a precise check and
             // collect IDs.
             [&](const auto &poly1, const auto &poly2) {
               if (tf::intersects(poly1, poly2)) {
                 intersecting_pairs_low_level.push_back(
                     {poly1.id(), poly2.id()});
               }
             });

  print_results("search", intersecting_pairs_low_level.to_vector());

  return 0;
}


EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::function("add", &add);
  emscripten::function("p", &p);
}


