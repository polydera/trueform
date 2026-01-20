/**
 * @file canonicalize_mesh.hpp
 * @brief Utilities for canonicalizing meshes for comparison
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once
#include <trueform/core.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

namespace tf::test {

/// @brief Canonicalize a mesh for comparison.
///
/// Steps:
/// 1. Sort points lexicographically
/// 2. Remap face indices to new point order
/// 3. Cyclic shift each face so minimal index is first
/// 4. Sort faces lexicographically
///
/// @tparam Index Index type
/// @tparam Real Coordinate type
/// @tparam Dims Number of dimensions
/// @tparam Ngon Vertices per face (static size)
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto canonicalize_mesh(const tf::polygons_buffer<Index, Real, Dims, Ngon>& input)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>
{
    auto points = input.points();
    auto faces = input.faces();

    const auto num_points = points.size();
    const auto num_faces = faces.size();

    if (num_points == 0 || num_faces == 0) {
        return tf::polygons_buffer<Index, Real, Dims, Ngon>{};
    }

    // Step 1: Create sorted point indices
    std::vector<std::size_t> point_order(num_points);
    std::iota(point_order.begin(), point_order.end(), 0);

    std::sort(point_order.begin(), point_order.end(),
        [&points](std::size_t a, std::size_t b) {
            for (std::size_t d = 0; d < Dims; ++d) {
                if (points[a][d] < points[b][d]) return true;
                if (points[a][d] > points[b][d]) return false;
            }
            return false;
        });

    // Step 2: Create reverse mapping (old index -> new index)
    std::vector<Index> old_to_new(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        old_to_new[point_order[i]] = static_cast<Index>(i);
    }

    // Build output
    tf::polygons_buffer<Index, Real, Dims, Ngon> output;

    // Add sorted points
    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<Real, Dims> pt;
        for (std::size_t d = 0; d < Dims; ++d) {
            pt[d] = points[point_order[i]][d];
        }
        output.points_buffer().push_back(pt);
    }

    // Step 3 & 4: Remap faces, cyclic shift, then sort
    std::vector<std::array<Index, Ngon>> remapped_faces;
    remapped_faces.reserve(num_faces);

    for (std::size_t f = 0; f < num_faces; ++f) {
        std::array<Index, Ngon> face;
        for (std::size_t v = 0; v < Ngon; ++v) {
            face[v] = old_to_new[static_cast<std::size_t>(faces[f][v])];
        }

        // Cyclic shift so minimal index is first
        auto min_it = std::min_element(face.begin(), face.end());
        std::size_t min_pos = std::distance(face.begin(), min_it);
        std::array<Index, Ngon> shifted;
        for (std::size_t v = 0; v < Ngon; ++v) {
            shifted[v] = face[(v + min_pos) % Ngon];
        }
        remapped_faces.push_back(shifted);
    }

    // Sort faces lexicographically
    std::sort(remapped_faces.begin(), remapped_faces.end(),
        [](const auto& a, const auto& b) {
            for (std::size_t v = 0; v < Ngon; ++v) {
                if (a[v] < b[v]) return true;
                if (a[v] > b[v]) return false;
            }
            return false;
        });

    // Add sorted faces
    for (const auto& face : remapped_faces) {
        output.faces_buffer().push_back(face);
    }

    return output;
}

/// @brief Canonicalize a dynamic mesh for comparison.
template <typename Index, typename Real, std::size_t Dims>
auto canonicalize_mesh(const tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>& input)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>
{
    auto points = input.points();
    auto faces = input.faces();

    const auto num_points = points.size();
    const auto num_faces = faces.size();

    if (num_points == 0 || num_faces == 0) {
        return tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>{};
    }

    // Step 1: Create sorted point indices
    std::vector<std::size_t> point_order(num_points);
    std::iota(point_order.begin(), point_order.end(), 0);

    std::sort(point_order.begin(), point_order.end(),
        [&points](std::size_t a, std::size_t b) {
            for (std::size_t d = 0; d < Dims; ++d) {
                if (points[a][d] < points[b][d]) return true;
                if (points[a][d] > points[b][d]) return false;
            }
            return false;
        });

    // Step 2: Create reverse mapping
    std::vector<Index> old_to_new(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        old_to_new[point_order[i]] = static_cast<Index>(i);
    }

    // Build output
    tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> output;

    // Add sorted points
    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<Real, Dims> pt;
        for (std::size_t d = 0; d < Dims; ++d) {
            pt[d] = points[point_order[i]][d];
        }
        output.points_buffer().push_back(pt);
    }

    // Step 3 & 4: Remap faces, cyclic shift, then sort
    std::vector<std::vector<Index>> remapped_faces;
    remapped_faces.reserve(num_faces);

    for (std::size_t f = 0; f < num_faces; ++f) {
        const auto ngon = faces[f].size();
        std::vector<Index> face(ngon);
        for (std::size_t v = 0; v < ngon; ++v) {
            face[v] = old_to_new[static_cast<std::size_t>(faces[f][v])];
        }

        // Cyclic shift so minimal index is first
        auto min_it = std::min_element(face.begin(), face.end());
        std::size_t min_pos = std::distance(face.begin(), min_it);
        std::vector<Index> shifted(ngon);
        for (std::size_t v = 0; v < ngon; ++v) {
            shifted[v] = face[(v + min_pos) % ngon];
        }
        remapped_faces.push_back(std::move(shifted));
    }

    // Sort faces lexicographically
    std::sort(remapped_faces.begin(), remapped_faces.end(),
        [](const auto& a, const auto& b) {
            const auto min_size = std::min(a.size(), b.size());
            for (std::size_t v = 0; v < min_size; ++v) {
                if (a[v] < b[v]) return true;
                if (a[v] > b[v]) return false;
            }
            return a.size() < b.size();
        });

    // Add sorted faces to output
    for (const auto& face : remapped_faces) {
        output.faces_buffer().push_back(tf::make_range(face));
    }

    return output;
}

/// @brief Check if two canonicalized meshes are equal.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto meshes_equal(const tf::polygons_buffer<Index, Real, Dims, Ngon>& a,
                  const tf::polygons_buffer<Index, Real, Dims, Ngon>& b,
                  Real tolerance = Real(1e-5)) -> bool
{
    auto pa = a.points();
    auto pb = b.points();
    auto fa = a.faces();
    auto fb = b.faces();

    if (pa.size() != pb.size()) return false;
    if (fa.size() != fb.size()) return false;

    // Compare points
    for (std::size_t i = 0; i < pa.size(); ++i) {
        for (std::size_t d = 0; d < Dims; ++d) {
            if (std::abs(pa[i][d] - pb[i][d]) > tolerance) return false;
        }
    }

    // Compare faces
    for (std::size_t i = 0; i < fa.size(); ++i) {
        for (std::size_t v = 0; v < Ngon; ++v) {
            if (fa[i][v] != fb[i][v]) return false;
        }
    }

    return true;
}

} // namespace tf::test
