/**
 * @file canonicalize_points.hpp
 * @brief Utilities for canonicalizing points for comparison
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

/// @brief Canonicalize points for comparison.
///
/// Steps:
/// 1. Sort points lexicographically
///
/// @tparam Real Coordinate type
/// @tparam Dims Number of dimensions
template <typename Real, std::size_t Dims>
auto canonicalize_points(const tf::points_buffer<Real, Dims>& input)
    -> tf::points_buffer<Real, Dims>
{
    auto points = input.points();
    const auto num_points = points.size();

    if (num_points == 0) {
        return tf::points_buffer<Real, Dims>{};
    }

    // Create sorted point indices
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

    // Build output with sorted points
    tf::points_buffer<Real, Dims> output;
    output.reserve(num_points);

    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<Real, Dims> pt;
        for (std::size_t d = 0; d < Dims; ++d) {
            pt[d] = points[point_order[i]][d];
        }
        output.push_back(pt);
    }

    return output;
}

/// @brief Check if two canonicalized point sets are equal.
///
/// @tparam Real Coordinate type
/// @tparam Dims Number of dimensions
template <typename Real, std::size_t Dims>
auto points_equal(const tf::points_buffer<Real, Dims>& a,
                  const tf::points_buffer<Real, Dims>& b,
                  Real tolerance = Real(1e-5)) -> bool
{
    auto pa = a.points();
    auto pb = b.points();

    if (pa.size() != pb.size()) return false;

    // Compare points
    for (std::size_t i = 0; i < pa.size(); ++i) {
        for (std::size_t d = 0; d < Dims; ++d) {
            if (std::abs(pa[i][d] - pb[i][d]) > tolerance) return false;
        }
    }

    return true;
}

} // namespace tf::test
