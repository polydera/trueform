/**
 * @file canonicalize_segments.hpp
 * @brief Utilities for canonicalizing segments for comparison
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once
#include <trueform/core.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace tf::test {

/// @brief Canonicalize segments for comparison.
///
/// Steps:
/// 1. Sort points lexicographically
/// 2. Remap edge indices to new point order
/// 3. Normalize each edge (smaller index first)
/// 4. Sort edges lexicographically
///
/// @tparam Index Index type
/// @tparam Real Coordinate type
/// @tparam Dims Number of dimensions
template <typename Index, typename Real, std::size_t Dims>
auto canonicalize_segments(const tf::segments_buffer<Index, Real, Dims>& input)
    -> tf::segments_buffer<Index, Real, Dims>
{
    auto points = input.points();
    auto edges = input.edges();

    const auto num_points = points.size();
    const auto num_edges = edges.size();

    if (num_points == 0 || num_edges == 0) {
        return tf::segments_buffer<Index, Real, Dims>{};
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
    tf::segments_buffer<Index, Real, Dims> output;

    // Add sorted points
    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<Real, Dims> pt;
        for (std::size_t d = 0; d < Dims; ++d) {
            pt[d] = points[point_order[i]][d];
        }
        output.points_buffer().push_back(pt);
    }

    // Step 3 & 4: Remap edges, normalize, then sort
    std::vector<std::array<Index, 2>> remapped_edges;
    remapped_edges.reserve(num_edges);

    for (std::size_t e = 0; e < num_edges; ++e) {
        Index v0 = old_to_new[static_cast<std::size_t>(edges[e][0])];
        Index v1 = old_to_new[static_cast<std::size_t>(edges[e][1])];

        // Normalize: smaller index first
        if (v0 > v1) {
            std::swap(v0, v1);
        }
        remapped_edges.push_back({v0, v1});
    }

    // Sort edges lexicographically
    std::sort(remapped_edges.begin(), remapped_edges.end(),
        [](const auto& a, const auto& b) {
            if (a[0] < b[0]) return true;
            if (a[0] > b[0]) return false;
            return a[1] < b[1];
        });

    // Add sorted edges
    for (const auto& edge : remapped_edges) {
        output.edges_buffer().push_back(edge);
    }

    return output;
}

/// @brief Check if two canonicalized segment sets are equal.
///
/// @tparam Index Index type
/// @tparam Real Coordinate type
/// @tparam Dims Number of dimensions
template <typename Index, typename Real, std::size_t Dims>
auto segments_equal(const tf::segments_buffer<Index, Real, Dims>& a,
                    const tf::segments_buffer<Index, Real, Dims>& b,
                    Real tolerance = Real(1e-5)) -> bool
{
    auto pa = a.points();
    auto pb = b.points();
    auto ea = a.edges();
    auto eb = b.edges();

    if (pa.size() != pb.size()) return false;
    if (ea.size() != eb.size()) return false;

    // Compare points
    for (std::size_t i = 0; i < pa.size(); ++i) {
        for (std::size_t d = 0; d < Dims; ++d) {
            if (std::abs(pa[i][d] - pb[i][d]) > tolerance) return false;
        }
    }

    // Compare edges
    for (std::size_t i = 0; i < ea.size(); ++i) {
        if (ea[i][0] != eb[i][0]) return false;
        if (ea[i][1] != eb[i][1]) return false;
    }

    return true;
}

} // namespace tf::test
