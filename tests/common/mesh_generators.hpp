/**
 * @file mesh_generators.hpp
 * @brief Mesh creation helpers for testing
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <array>
#include <cmath>

namespace tf::test {

// =============================================================================
// 2D Triangle Polygons
// =============================================================================

/**
 * @brief Create a simple 2D triangle polygons (2 triangles sharing an edge)
 */
template <typename index_t, typename real_t>
auto create_triangle_polygons_2d() -> tf::polygons_buffer<index_t, real_t, 2, 3>
{
    tf::polygons_buffer<index_t, real_t, 2, 3> result;
    result.faces_buffer().emplace_back(0, 1, 2);
    result.faces_buffer().emplace_back(1, 3, 2);

    result.points_buffer().emplace_back(real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(0), real_t(1));
    result.points_buffer().emplace_back(real_t(1), real_t(1));

    return result;
}

// =============================================================================
// 3D Triangle Polygons
// =============================================================================

/**
 * @brief Create a simple 3D triangle polygons (2 triangles sharing an edge)
 */
template <typename index_t, typename real_t>
auto create_triangle_polygons_3d() -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    tf::polygons_buffer<index_t, real_t, 3, 3> result;
    result.faces_buffer().emplace_back(0, 1, 2);
    result.faces_buffer().emplace_back(1, 3, 2);

    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    return result;
}

/**
 * @brief Create a larger 3D triangle polygons (4 triangles forming a strip)
 */
template <typename index_t, typename real_t>
auto create_larger_triangle_polygons_3d() -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    tf::polygons_buffer<index_t, real_t, 3, 3> result;
    result.faces_buffer().emplace_back(0, 1, 2);
    result.faces_buffer().emplace_back(1, 3, 2);
    result.faces_buffer().emplace_back(2, 3, 4);
    result.faces_buffer().emplace_back(3, 5, 4);

    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(2), real_t(0));
    result.points_buffer().emplace_back(real_t(2), real_t(2), real_t(0));

    return result;
}

// =============================================================================
// Cube Polygons
// =============================================================================

/**
 * @brief Create a cube polygons centered at given position with given size
 */
template <typename index_t, typename real_t>
auto create_cube_polygons(
    std::array<real_t, 3> center = {real_t(0), real_t(0), real_t(0)},
    real_t size = real_t(1)) -> tf::polygons_buffer<index_t, real_t, 3, 3>
{
    const real_t half = size / real_t(2);
    const real_t cx = center[0];
    const real_t cy = center[1];
    const real_t cz = center[2];

    tf::polygons_buffer<index_t, real_t, 3, 3> result;

    // 8 vertices
    result.points_buffer().emplace_back(cx - half, cy - half, cz - half);
    result.points_buffer().emplace_back(cx + half, cy - half, cz - half);
    result.points_buffer().emplace_back(cx + half, cy + half, cz - half);
    result.points_buffer().emplace_back(cx - half, cy + half, cz - half);
    result.points_buffer().emplace_back(cx - half, cy - half, cz + half);
    result.points_buffer().emplace_back(cx + half, cy - half, cz + half);
    result.points_buffer().emplace_back(cx + half, cy + half, cz + half);
    result.points_buffer().emplace_back(cx - half, cy + half, cz + half);

    // 12 triangles (2 per face) - CCW winding for outward normals
    // Bottom (z-)
    result.faces_buffer().emplace_back(0, 2, 1);
    result.faces_buffer().emplace_back(0, 3, 2);
    // Top (z+)
    result.faces_buffer().emplace_back(4, 5, 6);
    result.faces_buffer().emplace_back(4, 6, 7);
    // Front (y-)
    result.faces_buffer().emplace_back(0, 1, 5);
    result.faces_buffer().emplace_back(0, 5, 4);
    // Back (y+)
    result.faces_buffer().emplace_back(2, 3, 7);
    result.faces_buffer().emplace_back(2, 7, 6);
    // Left (x-)
    result.faces_buffer().emplace_back(0, 4, 7);
    result.faces_buffer().emplace_back(0, 7, 3);
    // Right (x+)
    result.faces_buffer().emplace_back(1, 2, 6);
    result.faces_buffer().emplace_back(1, 6, 5);

    return result;
}

// =============================================================================
// Dynamic Polygons (mixed n-gons using offset_block_buffer)
// =============================================================================

/**
 * @brief Create a simple 2D dynamic polygons (2 triangles)
 */
template <typename index_t, typename real_t>
auto create_dynamic_polygons_2d()
    -> tf::polygons_buffer<index_t, real_t, 2, tf::dynamic_size>
{
    tf::polygons_buffer<index_t, real_t, 2, tf::dynamic_size> result;
    result.faces_buffer().push_back({0, 1, 2});
    result.faces_buffer().push_back({1, 3, 2});

    result.points_buffer().emplace_back(real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(0), real_t(1));
    result.points_buffer().emplace_back(real_t(1), real_t(1));

    return result;
}

/**
 * @brief Create a simple 3D dynamic polygons (2 triangles)
 */
template <typename index_t, typename real_t>
auto create_dynamic_polygons_3d()
    -> tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size>
{
    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> result;
    result.faces_buffer().push_back({0, 1, 2});
    result.faces_buffer().push_back({1, 3, 2});

    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(1));

    return result;
}

/**
 * @brief Create a 3D dynamic polygons with mixed sizes (triangle + quad)
 */
template <typename index_t, typename real_t>
auto create_mixed_polygons_3d()
    -> tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size>
{
    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> result;
    // Triangle [0, 1, 2]
    result.faces_buffer().push_back({0, 1, 2});
    // Quad [1, 3, 4, 2]
    result.faces_buffer().push_back({1, 3, 4, 2});

    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    return result;
}

// =============================================================================
// Segments
// =============================================================================

/**
 * @brief Create a simple 3D segments (3 edges forming a path)
 */
template <typename index_t, typename real_t>
auto create_segments_3d() -> tf::segments_buffer<index_t, real_t, 3>
{
    tf::segments_buffer<index_t, real_t, 3> result;
    result.edges_buffer().emplace_back(0, 1);
    result.edges_buffer().emplace_back(1, 2);
    result.edges_buffer().emplace_back(2, 3);

    result.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    return result;
}

/**
 * @brief Create a 2D segments
 */
template <typename index_t, typename real_t>
auto create_segments_2d() -> tf::segments_buffer<index_t, real_t, 2>
{
    tf::segments_buffer<index_t, real_t, 2> result;
    result.edges_buffer().emplace_back(0, 1);
    result.edges_buffer().emplace_back(1, 2);
    result.edges_buffer().emplace_back(2, 3);

    result.points_buffer().emplace_back(real_t(0), real_t(0));
    result.points_buffer().emplace_back(real_t(1), real_t(0));
    result.points_buffer().emplace_back(real_t(2), real_t(0));
    result.points_buffer().emplace_back(real_t(3), real_t(0));

    return result;
}

// =============================================================================
// Points
// =============================================================================

/**
 * @brief Create a simple 3D points
 */
template <typename real_t>
auto create_points_3d(std::size_t n = 10) -> tf::points_buffer<real_t, 3>
{
    tf::points_buffer<real_t, 3> result;
    result.allocate(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto t = static_cast<real_t>(i) / static_cast<real_t>(n);
        result.points()[i][0] = t;
        result.points()[i][1] = t * t;
        result.points()[i][2] = real_t(0);
    }
    return result;
}

/**
 * @brief Create a simple 2D points
 */
template <typename real_t>
auto create_points_2d(std::size_t n = 10) -> tf::points_buffer<real_t, 2>
{
    tf::points_buffer<real_t, 2> result;
    result.allocate(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto t = static_cast<real_t>(i) / static_cast<real_t>(n);
        result.points()[i][0] = t;
        result.points()[i][1] = t * t;
    }
    return result;
}

/**
 * @brief Create a grid of 3D points
 */
template <typename real_t>
auto create_grid_points_3d(std::size_t nx, std::size_t ny, std::size_t nz)
    -> tf::points_buffer<real_t, 3>
{
    tf::points_buffer<real_t, 3> result;
    result.allocate(nx * ny * nz);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                result.points()[idx][0] = static_cast<real_t>(i);
                result.points()[idx][1] = static_cast<real_t>(j);
                result.points()[idx][2] = static_cast<real_t>(k);
                ++idx;
            }
        }
    }
    return result;
}

// =============================================================================
// Dynamic Mesh Conversion Utilities
// =============================================================================

/**
 * @brief Convert a fixed-size polygons_buffer to dynamic
 */
template <typename Index, typename Real, std::size_t Dims, std::size_t N>
auto to_dynamic(const tf::polygons_buffer<Index, Real, Dims, N>& fixed)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>
{
    tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> result;

    for (const auto& pt : fixed.points()) {
        result.points_buffer().push_back(pt);
    }

    for (const auto& face : fixed.faces()) {
        result.faces_buffer().push_back(tf::make_range(face));
    }

    return result;
}

/**
 * @brief Conditionally convert to dynamic based on template bool
 */
template <bool ToDynamic, typename Index, typename Real, std::size_t Dims, std::size_t N>
auto maybe_as_dynamic(tf::polygons_buffer<Index, Real, Dims, N>&& mesh) {
    if constexpr (ToDynamic) {
        return to_dynamic(mesh);
    } else {
        return std::move(mesh);
    }
}

} // namespace tf::test
