/**
 * Test package for trueform Conan recipe.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include <trueform/version.hpp>
#include <iostream>

int main() {
    // Verify version is accessible
    std::cout << "trueform version: " << tf::version << std::endl;

    // Create a simple triangle mesh
    tf::polygons_buffer<int, float, 3, 3> mesh;
    mesh.points_buffer().emplace_back(0.f, 0.f, 0.f);
    mesh.points_buffer().emplace_back(1.f, 0.f, 0.f);
    mesh.points_buffer().emplace_back(0.f, 1.f, 0.f);
    mesh.faces_buffer().emplace_back(0, 1, 2);

    std::cout << "Created mesh with " << mesh.points().size() << " points and "
              << mesh.faces().size() << " faces" << std::endl;

    // Test basic functionality
    auto aabb = tf::aabb_from(mesh.points());
    std::cout << "AABB computed successfully" << std::endl;

    std::cout << "trueform package test passed!" << std::endl;
    return 0;
}
