/**
 * @file spatial_generators.hpp
 * @brief Larger mesh/point generators for spatial testing
 *
 * These generators create data sets of 20-50 primitives:
 * - Large enough to exercise tree traversal
 * - Small enough for brute-force verification
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <array>
#include <cmath>

namespace tf::test {

// =============================================================================
// Grid Mesh (3D) - Creates (nx-1)*(ny-1)*2 triangles
// =============================================================================

/**
 * @brief Create a triangulated grid mesh in the XY plane at z=0
 *
 * Creates a regular grid with vertices at integer positions,
 * triangulated with consistent winding. A 5x5 grid produces 32 triangles.
 *
 * @param nx Number of vertices in X direction (default 5)
 * @param ny Number of vertices in Y direction (default 5)
 * @param offset Translation offset for the mesh
 * @return A polygons_buffer with (nx-1)*(ny-1)*2 triangles
 */
template <typename index_t, typename real_t>
auto create_grid_mesh_3d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 3> offset = {real_t(0), real_t(0), real_t(0)})
    -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    tf::polygons_buffer<index_t, real_t, 3, 3> result;

    // Create vertices
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1],
                offset[2]);
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
 * @brief Create a triangulated grid mesh in the XY plane (2D)
 */
template <typename index_t, typename real_t>
auto create_grid_mesh_2d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 2> offset = {real_t(0), real_t(0)})
    -> tf::polygons_buffer<index_t, real_t, 2, 3>
{
    tf::polygons_buffer<index_t, real_t, 2, 3> result;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1]);
        }
    }

    for (std::size_t j = 0; j < ny - 1; ++j) {
        for (std::size_t i = 0; i < nx - 1; ++i) {
            auto v00 = static_cast<index_t>(j * nx + i);
            auto v10 = static_cast<index_t>(j * nx + i + 1);
            auto v01 = static_cast<index_t>((j + 1) * nx + i);
            auto v11 = static_cast<index_t>((j + 1) * nx + i + 1);

            result.faces_buffer().emplace_back(v00, v10, v01);
            result.faces_buffer().emplace_back(v10, v11, v01);
        }
    }

    return result;
}

// =============================================================================
// Dynamic Grid Mesh - Mixed triangles and quads
// =============================================================================

/**
 * @brief Create a grid mesh with mixed polygon types (triangles and quads)
 *
 * Alternates between triangle pairs and quads for variety.
 */
template <typename index_t, typename real_t>
auto create_dynamic_grid_mesh_3d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 3> offset = {real_t(0), real_t(0), real_t(0)})
    -> tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size>
{
    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> result;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1],
                offset[2]);
        }
    }

    for (std::size_t j = 0; j < ny - 1; ++j) {
        for (std::size_t i = 0; i < nx - 1; ++i) {
            auto v00 = static_cast<index_t>(j * nx + i);
            auto v10 = static_cast<index_t>(j * nx + i + 1);
            auto v01 = static_cast<index_t>((j + 1) * nx + i);
            auto v11 = static_cast<index_t>((j + 1) * nx + i + 1);

            // Alternate between quad and triangle pair
            if ((i + j) % 2 == 0) {
                // Quad
                result.faces_buffer().push_back({v00, v10, v11, v01});
            } else {
                // Two triangles
                result.faces_buffer().push_back({v00, v10, v01});
                result.faces_buffer().push_back({v10, v11, v01});
            }
        }
    }

    return result;
}

/**
 * @brief Create a 2D dynamic grid mesh with mixed polygons
 */
template <typename index_t, typename real_t>
auto create_dynamic_grid_mesh_2d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 2> offset = {real_t(0), real_t(0)})
    -> tf::polygons_buffer<index_t, real_t, 2, tf::dynamic_size>
{
    tf::polygons_buffer<index_t, real_t, 2, tf::dynamic_size> result;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1]);
        }
    }

    for (std::size_t j = 0; j < ny - 1; ++j) {
        for (std::size_t i = 0; i < nx - 1; ++i) {
            auto v00 = static_cast<index_t>(j * nx + i);
            auto v10 = static_cast<index_t>(j * nx + i + 1);
            auto v01 = static_cast<index_t>((j + 1) * nx + i);
            auto v11 = static_cast<index_t>((j + 1) * nx + i + 1);

            if ((i + j) % 2 == 0) {
                result.faces_buffer().push_back({v00, v10, v11, v01});
            } else {
                result.faces_buffer().push_back({v00, v10, v01});
                result.faces_buffer().push_back({v10, v11, v01});
            }
        }
    }

    return result;
}

// =============================================================================
// Grid Segments - Creates grid of connected edges
// =============================================================================

/**
 * @brief Create a grid of segments (horizontal and vertical edges)
 *
 * A 5x5 grid produces 40 segments (20 horizontal + 20 vertical).
 */
template <typename index_t, typename real_t>
auto create_grid_segments_3d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 3> offset = {real_t(0), real_t(0), real_t(0)})
    -> tf::segments_buffer<index_t, real_t, 3>
{
    tf::segments_buffer<index_t, real_t, 3> result;

    // Create vertices
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1],
                offset[2]);
        }
    }

    // Horizontal edges
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx - 1; ++i) {
            auto v0 = static_cast<index_t>(j * nx + i);
            auto v1 = static_cast<index_t>(j * nx + i + 1);
            result.edges_buffer().emplace_back(v0, v1);
        }
    }

    // Vertical edges
    for (std::size_t j = 0; j < ny - 1; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            auto v0 = static_cast<index_t>(j * nx + i);
            auto v1 = static_cast<index_t>((j + 1) * nx + i);
            result.edges_buffer().emplace_back(v0, v1);
        }
    }

    return result;
}

/**
 * @brief Create a 2D grid of segments
 */
template <typename index_t, typename real_t>
auto create_grid_segments_2d(
    std::size_t nx = 5,
    std::size_t ny = 5,
    std::array<real_t, 2> offset = {real_t(0), real_t(0)})
    -> tf::segments_buffer<index_t, real_t, 2>
{
    tf::segments_buffer<index_t, real_t, 2> result;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points_buffer().emplace_back(
                static_cast<real_t>(i) + offset[0],
                static_cast<real_t>(j) + offset[1]);
        }
    }

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx - 1; ++i) {
            auto v0 = static_cast<index_t>(j * nx + i);
            auto v1 = static_cast<index_t>(j * nx + i + 1);
            result.edges_buffer().emplace_back(v0, v1);
        }
    }

    for (std::size_t j = 0; j < ny - 1; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            auto v0 = static_cast<index_t>(j * nx + i);
            auto v1 = static_cast<index_t>((j + 1) * nx + i);
            result.edges_buffer().emplace_back(v0, v1);
        }
    }

    return result;
}

// =============================================================================
// Grid Point Clouds
// =============================================================================

/**
 * @brief Create a 3D grid of points
 *
 * A 4x4x4 grid produces 64 points.
 */
template <typename real_t>
auto create_grid_points_3d(
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    std::array<real_t, 3> offset = {real_t(0), real_t(0), real_t(0)})
    -> tf::points_buffer<real_t, 3>
{
    tf::points_buffer<real_t, 3> result;
    result.allocate(nx * ny * nz);

    std::size_t idx = 0;
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                result.points()[idx][0] = static_cast<real_t>(i) + offset[0];
                result.points()[idx][1] = static_cast<real_t>(j) + offset[1];
                result.points()[idx][2] = static_cast<real_t>(k) + offset[2];
                ++idx;
            }
        }
    }

    return result;
}

/**
 * @brief Create a 2D grid of points
 */
template <typename real_t>
auto create_grid_points_2d(
    std::size_t nx,
    std::size_t ny,
    std::array<real_t, 2> offset = {real_t(0), real_t(0)})
    -> tf::points_buffer<real_t, 2>
{
    tf::points_buffer<real_t, 2> result;
    result.allocate(nx * ny);

    std::size_t idx = 0;
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            result.points()[idx][0] = static_cast<real_t>(i) + offset[0];
            result.points()[idx][1] = static_cast<real_t>(j) + offset[1];
            ++idx;
        }
    }

    return result;
}

// =============================================================================
// Point Cloud with Duplicates (for gather_self_ids testing)
// =============================================================================

/**
 * @brief Create a point cloud with some near-duplicate points
 *
 * Creates a grid with some points very close together for testing
 * duplicate detection algorithms.
 */
template <typename real_t>
auto create_points_with_duplicates_3d(real_t tolerance = real_t(0.001))
    -> tf::points_buffer<real_t, 3>
{
    tf::points_buffer<real_t, 3> result;

    // Regular grid points
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.push_back({real_t(i), real_t(j), real_t(0)});
        }
    }

    // Add some near-duplicates
    result.push_back({tolerance, real_t(0), real_t(0)});           // near (0,0,0)
    result.push_back({real_t(1) + tolerance, real_t(1), real_t(0)}); // near (1,1,0)
    result.push_back({real_t(2), real_t(2) + tolerance, real_t(0)}); // near (2,2,0)

    return result;
}

// =============================================================================
// Self-Intersecting Mesh (for gather_self_ids testing)
// =============================================================================

/**
 * @brief Create a mesh with self-intersecting triangles
 *
 * Creates two triangles that cross through each other.
 */
template <typename index_t, typename real_t>
auto create_self_intersecting_mesh_3d()
    -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    tf::polygons_buffer<index_t, real_t, 3, 3> result;

    // Triangle 1: in XY plane at z=0
    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(2), real_t(0));

    // Triangle 2: crossing through triangle 1
    result.points_buffer().emplace_back(real_t(1), real_t(0.5), real_t(-1));
    result.points_buffer().emplace_back(real_t(1), real_t(0.5), real_t(1));
    result.points_buffer().emplace_back(real_t(1), real_t(2.5), real_t(0));

    result.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2));
    result.faces_buffer().emplace_back(index_t(3), index_t(4), index_t(5));

    return result;
}

/**
 * @brief Create a grid mesh with some self-intersecting triangles
 *
 * Normal grid plus a few triangles that intersect with grid triangles.
 */
template <typename index_t, typename real_t>
auto create_grid_with_intersections_3d(std::size_t nx = 4, std::size_t ny = 4)
    -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    // Start with a normal grid
    auto result = create_grid_mesh_3d<index_t, real_t>(nx, ny);

    // Add an intersecting triangle
    auto base_idx = static_cast<index_t>(result.points().size());
    result.points_buffer().emplace_back(real_t(1), real_t(1), real_t(-0.5));
    result.points_buffer().emplace_back(real_t(2), real_t(1), real_t(0.5));
    result.points_buffer().emplace_back(real_t(1.5), real_t(2), real_t(0));

    result.faces_buffer().emplace_back(base_idx, base_idx + 1, base_idx + 2);

    return result;
}

} // namespace tf::test
