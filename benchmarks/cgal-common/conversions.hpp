/**
 * CGAL conversion utilities
 *
 * Helper functions for converting between TrueForm and CGAL data structures.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

namespace benchmark {
namespace cgal {

using Kernel = CGAL::Simple_cartesian<float>;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;


using Kernel_d = CGAL::Simple_cartesian<double>;
using Point_3_d = Kernel_d::Point_3;
using Surface_mesh_d = CGAL::Surface_mesh<Point_3_d>;

/**
 * Convert TrueForm mesh to CGAL Surface_mesh.
 *
 * @param r_polygons TrueForm polygon buffer (result from tf::read_stl)
 * @return CGAL Surface_mesh containing the same geometry
 */
template <typename PolygonBuffer>
Surface_mesh to_cgal_mesh(const PolygonBuffer& r_polygons) {
    Surface_mesh mesh;
    auto points = r_polygons.points();
    auto faces = r_polygons.faces();

    // Add vertices
    std::vector<Surface_mesh::Vertex_index> vertices;
    vertices.reserve(points.size());
    for (const auto& pt : points) {
        vertices.push_back(mesh.add_vertex(Point_3(pt[0], pt[1], pt[2])));
    }

    // Add faces
    for (const auto& face : faces) {
        mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    }

    return mesh;
}

template <typename PolygonBuffer>
Surface_mesh_d to_cgal_mesh_d(const PolygonBuffer& r_polygons) {
    Surface_mesh_d mesh;
    auto points = r_polygons.points();
    auto faces = r_polygons.faces();

    // Add vertices
    std::vector<Surface_mesh_d::Vertex_index> vertices;
    vertices.reserve(points.size());
    for (const auto& pt : points) {
        vertices.push_back(mesh.add_vertex(Point_3_d(pt[0], pt[1], pt[2])));
    }

    // Add faces
    for (const auto& face : faces) {
        mesh.add_face(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    }

    return mesh;
}

/**
 * Convert TrueForm points to vector of CGAL Point_3.
 */
template <typename Policy>
std::vector<Point_3> to_cgal_points(const tf::points<Policy>& points) {
    std::vector<Point_3> result;
    result.reserve(points.size());
    for (const auto& pt : points) {
        result.emplace_back(pt[0], pt[1], pt[2]);
    }
    return result;
}

/**
 * Convert TrueForm faces to vector of index arrays.
 */
template <typename Policy>
std::vector<std::array<std::size_t, 3>> to_cgal_faces(const tf::faces<Policy>& faces) {
    std::vector<std::array<std::size_t, 3>> result;
    result.reserve(faces.size());
    for (const auto& face : faces) {
        result.push_back({static_cast<std::size_t>(face[0]),
                          static_cast<std::size_t>(face[1]),
                          static_cast<std::size_t>(face[2])});
    }
    return result;
}

}  // namespace cgal
}  // namespace benchmark
