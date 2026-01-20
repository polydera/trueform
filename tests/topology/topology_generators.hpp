/**
 * @file topology_generators.hpp
 * @brief Mesh generators for topology testing
 *
 * Provides mesh primitives with known topological properties:
 * - Two triangles (basic open mesh)
 * - Tetrahedron (closed mesh)
 * - Triangle strip (linear topology)
 * - Grid mesh (regular connectivity)
 * - Non-manifold mesh (edge shared by 3+ faces)
 * - Two disconnected components
 * - Dynamic mesh (mixed polygon sizes)
 * - Edge mesh (for vertex_link testing)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <array>
#include <cmath>

namespace tf::test {

// =============================================================================
// Two Triangles - Basic Open Mesh
// =============================================================================

/**
 * @brief Create two triangles sharing an edge (1,2)
 *
 * Topology:
 *   0 ------- 2
 *   |  \   /  |
 *   |   \ /   |
 *   |    1    |
 *   |   / \   |
 *   |  /   \  |
 *   3 ------- (continues)
 *
 * Actually:
 *       0
 *      /|\
 *     / | \
 *    /  |  \
 *   1---+---2
 *    \  |  /
 *     \ | /
 *      \|/
 *       3
 *
 * Face 0: [0, 1, 2]
 * Face 1: [1, 3, 2]
 * Shared edge: (1, 2)
 * Boundary edges: (0,1), (0,2), (1,3), (2,3)
 */
template <typename Index, typename Real>
auto create_two_triangles_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Vertices
    result.points_buffer().emplace_back(Real(0.5), Real(1), Real(0));   // 0 - top
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));     // 1 - bottom left
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));     // 2 - bottom right
    result.points_buffer().emplace_back(Real(0.5), Real(-1), Real(0));  // 3 - very bottom

    // Faces
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));  // Face 0
    result.faces_buffer().emplace_back(Index(1), Index(3), Index(2));  // Face 1

    return result;
}

// =============================================================================
// Tetrahedron - Closed Mesh (no boundary edges)
// =============================================================================

/**
 * @brief Create a closed tetrahedron with 4 vertices and 4 triangular faces
 *
 * A regular tetrahedron with CCW winding (outward normals).
 * Every edge is shared by exactly 2 faces -> no boundary edges.
 *
 * Vertices:
 *   0: (0, 0, 0)
 *   1: (1, 0, 0)
 *   2: (0.5, sqrt(3)/2, 0)
 *   3: (0.5, sqrt(3)/6, sqrt(2/3))
 *
 * Faces (CCW from outside):
 *   F0: [0, 2, 1] - bottom
 *   F1: [0, 1, 3] - front
 *   F2: [1, 2, 3] - right
 *   F3: [2, 0, 3] - left
 */
template <typename Index, typename Real>
auto create_tetrahedron_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Regular tetrahedron vertices
    Real sqrt3_2 = std::sqrt(Real(3)) / Real(2);
    Real sqrt3_6 = std::sqrt(Real(3)) / Real(6);
    Real sqrt2_3 = std::sqrt(Real(2) / Real(3));

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0.5), sqrt3_2, Real(0));
    result.points_buffer().emplace_back(Real(0.5), sqrt3_6, sqrt2_3);

    // Faces with CCW winding (outward normals)
    result.faces_buffer().emplace_back(Index(0), Index(2), Index(1));  // bottom
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(3));  // front
    result.faces_buffer().emplace_back(Index(1), Index(2), Index(3));  // right
    result.faces_buffer().emplace_back(Index(2), Index(0), Index(3));  // left

    return result;
}

// =============================================================================
// Triangle Strip - Linear Topology for k-ring Testing
// =============================================================================

/**
 * @brief Create a triangle strip with n_triangles triangles
 *
 * Creates a strip of triangles along the X axis:
 *
 *   v0 --- v2 --- v4 --- v6 ...
 *    \  0  / \  2  / \  4  /
 *     \   /   \   /   \   /
 *      \ / 1   \ / 3   \ /
 *       v1 --- v3 --- v5 ...
 *
 * Each pair of adjacent triangles shares an edge.
 * Good for testing k-ring traversal in a linear structure.
 *
 * @param n_triangles Number of triangles (default: 5)
 */
template <typename Index, typename Real>
auto create_triangle_strip_3d(std::size_t n_triangles = 5)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Number of vertices = n_triangles + 2
    std::size_t n_vertices = n_triangles + 2;

    // Create vertices in a zigzag pattern
    for (std::size_t i = 0; i < n_vertices; ++i) {
        Real x = static_cast<Real>(i / 2);
        Real y = (i % 2 == 0) ? Real(1) : Real(0);
        result.points_buffer().emplace_back(x, y, Real(0));
    }

    // Create faces
    for (std::size_t i = 0; i < n_triangles; ++i) {
        if (i % 2 == 0) {
            // Even triangle: [i, i+1, i+2]
            result.faces_buffer().emplace_back(
                static_cast<Index>(i),
                static_cast<Index>(i + 1),
                static_cast<Index>(i + 2));
        } else {
            // Odd triangle: [i, i+2, i+1]
            result.faces_buffer().emplace_back(
                static_cast<Index>(i),
                static_cast<Index>(i + 2),
                static_cast<Index>(i + 1));
        }
    }

    return result;
}

// =============================================================================
// Grid Mesh - Regular Connectivity (re-export from spatial_generators)
// =============================================================================

/**
 * @brief Create a triangulated grid mesh in the XY plane
 *
 * Creates a regular grid with vertices at integer positions,
 * triangulated with consistent winding. Useful for testing
 * neighborhoods and connectivity on a regular structure.
 *
 * @param rows Number of rows (vertices in Y direction)
 * @param cols Number of columns (vertices in X direction)
 */
template <typename Index, typename Real>
auto create_grid_mesh_3d(std::size_t rows = 5, std::size_t cols = 5)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Create vertices
    for (std::size_t j = 0; j < rows; ++j) {
        for (std::size_t i = 0; i < cols; ++i) {
            result.points_buffer().emplace_back(
                static_cast<Real>(i),
                static_cast<Real>(j),
                Real(0));
        }
    }

    // Create triangles (2 per grid cell)
    for (std::size_t j = 0; j < rows - 1; ++j) {
        for (std::size_t i = 0; i < cols - 1; ++i) {
            auto v00 = static_cast<Index>(j * cols + i);
            auto v10 = static_cast<Index>(j * cols + i + 1);
            auto v01 = static_cast<Index>((j + 1) * cols + i);
            auto v11 = static_cast<Index>((j + 1) * cols + i + 1);

            // Lower-left triangle
            result.faces_buffer().emplace_back(v00, v10, v01);
            // Upper-right triangle
            result.faces_buffer().emplace_back(v10, v11, v01);
        }
    }

    return result;
}

// =============================================================================
// Non-Manifold Mesh - Edge Shared by 3 Faces
// =============================================================================

/**
 * @brief Create a mesh with a non-manifold edge (shared by 3 triangles)
 *
 * Three triangles all sharing edge (0, 1):
 *
 *           2
 *          /|\
 *         / | \
 *        /  |  \
 *       / F0|F1 \
 *      /   |   \
 *     0----+----1
 *      \   |   /
 *       \ F2  /
 *        \  |  /
 *         \ | /
 *          \|/
 *           3
 *
 * Edge (0,1) is non-manifold because it's shared by faces F0, F1, and F2.
 *
 * Face 0: [0, 1, 2]
 * Face 1: [0, 2, 1] - Note: different winding
 * Face 2: [0, 3, 1]
 *
 * Actually, let's make it cleaner - 3 triangles sharing edge (0,1):
 * Face 0: [0, 1, 2]
 * Face 1: [1, 0, 3]
 * Face 2: [0, 1, 4]
 */
template <typename Index, typename Real>
auto create_non_manifold_mesh_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Vertices forming three triangles sharing edge (0,1)
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));   // 0
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));   // 1
    result.points_buffer().emplace_back(Real(0.5), Real(1), Real(0)); // 2 - above
    result.points_buffer().emplace_back(Real(0.5), Real(-1), Real(0)); // 3 - below
    result.points_buffer().emplace_back(Real(0.5), Real(0), Real(1)); // 4 - in front

    // Three faces all sharing edge (0,1)
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(1), Index(0), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(4));

    return result;
}

// =============================================================================
// Two Disconnected Components
// =============================================================================

/**
 * @brief Create a mesh with two disconnected triangle components
 *
 * Component 1: Vertices 0,1,2 forming a triangle
 * Component 2: Vertices 3,4,5 forming another triangle (translated)
 *
 * Face 0: [0, 1, 2]
 * Face 1: [3, 4, 5]
 */
template <typename Index, typename Real>
auto create_two_components_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Component 1: triangle at origin
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0.5), Real(1), Real(0));

    // Component 2: triangle translated far away
    result.points_buffer().emplace_back(Real(10), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(11), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(10.5), Real(1), Real(0));

    // Faces
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(3), Index(4), Index(5));

    return result;
}

// =============================================================================
// Dynamic Mesh - Mixed Triangles and Quads
// =============================================================================

/**
 * @brief Create a dynamic mesh with mixed polygon sizes
 *
 * Creates a simple mesh with one triangle and one quad sharing an edge:
 *
 *     0 ----- 1
 *     | \     |
 *     |  \    |
 *     |   \   |
 *     |    \  |
 *     |     \ |
 *     3------2----- 4
 *
 * Face 0 (triangle): [0, 2, 3]
 * Face 1 (quad): [0, 1, 4, 2]
 */
template <typename Index, typename Real>
auto create_dynamic_mesh_3d()
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
    tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> result;

    // Vertices
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));   // 0
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));   // 1
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));   // 2
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));   // 3
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));   // 4

    // Triangle
    result.faces_buffer().push_back({Index(0), Index(2), Index(3)});
    // Quad
    result.faces_buffer().push_back({Index(0), Index(1), Index(4), Index(2)});

    return result;
}

// =============================================================================
// Edge Mesh - For vertex_link Testing on Segments
// =============================================================================

/**
 * @brief Create a simple edge mesh (path + branching)
 *
 *     0 --- 1 --- 2
 *           |
 *           3
 *           |
 *           4
 *
 * Edges: (0,1), (1,2), (1,3), (3,4)
 */
template <typename Index, typename Real>
auto create_edge_mesh_3d() -> tf::segments_buffer<Index, Real, 3> {
    tf::segments_buffer<Index, Real, 3> result;

    // Vertices
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));  // 0
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));  // 1
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));  // 2
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(0)); // 3
    result.points_buffer().emplace_back(Real(1), Real(-2), Real(0)); // 4

    // Edges
    result.edges_buffer().emplace_back(Index(0), Index(1));
    result.edges_buffer().emplace_back(Index(1), Index(2));
    result.edges_buffer().emplace_back(Index(1), Index(3));
    result.edges_buffer().emplace_back(Index(3), Index(4));

    return result;
}

// =============================================================================
// Mesh with Hole - Multiple Boundary Loops
// =============================================================================

/**
 * @brief Create a mesh with a hole (donut-like topology)
 *
 * An outer square with an inner square hole. This creates two boundary loops:
 * - Outer boundary: around the outside
 * - Inner boundary: around the hole
 *
 * Outer vertices (0-3): corners of outer square
 * Inner vertices (4-7): corners of inner square (hole)
 *
 *   0 --------- 1
 *   |  4 --- 5  |
 *   |  |     |  |
 *   |  7 --- 6  |
 *   3 --------- 2
 */
template <typename Index, typename Real>
auto create_mesh_with_hole_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Outer square vertices
    result.points_buffer().emplace_back(Real(0), Real(2), Real(0));   // 0
    result.points_buffer().emplace_back(Real(2), Real(2), Real(0));   // 1
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));   // 2
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));   // 3

    // Inner square vertices (hole)
    result.points_buffer().emplace_back(Real(0.5), Real(1.5), Real(0)); // 4
    result.points_buffer().emplace_back(Real(1.5), Real(1.5), Real(0)); // 5
    result.points_buffer().emplace_back(Real(1.5), Real(0.5), Real(0)); // 6
    result.points_buffer().emplace_back(Real(0.5), Real(0.5), Real(0)); // 7

    // Triangles forming the ring between outer and inner squares
    // Top
    result.faces_buffer().emplace_back(Index(0), Index(4), Index(5));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(1));
    // Right
    result.faces_buffer().emplace_back(Index(1), Index(5), Index(6));
    result.faces_buffer().emplace_back(Index(1), Index(6), Index(2));
    // Bottom
    result.faces_buffer().emplace_back(Index(2), Index(6), Index(7));
    result.faces_buffer().emplace_back(Index(2), Index(7), Index(3));
    // Left
    result.faces_buffer().emplace_back(Index(3), Index(7), Index(4));
    result.faces_buffer().emplace_back(Index(3), Index(4), Index(0));

    return result;
}

// =============================================================================
// Mesh with Inconsistent Winding
// =============================================================================

/**
 * @brief Create two triangles with inconsistent winding (for orient_faces_consistently testing)
 *
 * Same geometry as create_two_triangles_3d but with face 1 having reversed winding.
 *
 * Face 0: [0, 1, 2] - CCW
 * Face 1: [2, 3, 1] - CW (inconsistent with face 0)
 */
template <typename Index, typename Real>
auto create_inconsistent_winding_mesh_3d()
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    // Same vertices as two_triangles
    result.points_buffer().emplace_back(Real(0.5), Real(1), Real(0));   // 0 - top
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));     // 1 - bottom left
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));     // 2 - bottom right
    result.points_buffer().emplace_back(Real(0.5), Real(-1), Real(0));  // 3 - very bottom

    // Face 0: CCW
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    // Face 1: CW (reversed winding - inconsistent)
    result.faces_buffer().emplace_back(Index(2), Index(3), Index(1));

    return result;
}

} // namespace tf::test
