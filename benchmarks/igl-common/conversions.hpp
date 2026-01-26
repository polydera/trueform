/**
 * libigl conversion utilities
 *
 * Helper functions for converting between TrueForm and libigl data structures.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <Eigen/Core>

namespace benchmark {
namespace igl {

/**
 * Convert TrueForm points to Eigen matrix (V).
 */
template <typename Policy>
Eigen::MatrixXd to_igl_vertices(const tf::points<Policy>& points) {
    Eigen::MatrixXd V(points.size(), 3);
    for (std::size_t i = 0; i < points.size(); ++i) {
        V(i, 0) = points[i][0];
        V(i, 1) = points[i][1];
        V(i, 2) = points[i][2];
    }
    return V;
}

/**
 * Convert TrueForm faces to Eigen matrix (F).
 */
template <typename Policy>
Eigen::MatrixXi to_igl_faces(const tf::faces<Policy>& faces) {
    Eigen::MatrixXi F(faces.size(), 3);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        F(i, 0) = faces[i][0];
        F(i, 1) = faces[i][1];
        F(i, 2) = faces[i][2];
    }
    return F;
}

}  // namespace igl
}  // namespace benchmark
