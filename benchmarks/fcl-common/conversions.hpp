/**
 * FCL conversion utilities
 *
 * Helper functions for converting between TrueForm and FCL data structures.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <fcl/fcl.h>
#include <vector>

namespace benchmark {
namespace fcl {

using Scalar          = double;
using BV_AABB         = ::fcl::AABB<Scalar>;
using BV_OBB          = ::fcl::OBB<Scalar>;
using BV_OBBRSS       = ::fcl::OBBRSS<Scalar>;
using BV              = BV_AABB;
using Model           = ::fcl::BVHModel<BV>;
using Model_AABB      = ::fcl::BVHModel<BV_AABB>;
using Model_OBB       = ::fcl::BVHModel<BV_OBB>;
using Model_OBBRSS    = ::fcl::BVHModel<BV_OBBRSS>;
using Vec3            = ::fcl::Vector3<Scalar>;
using Transform3      = ::fcl::Transform3<Scalar>;
using DistanceRequest  = ::fcl::DistanceRequest<Scalar>;
using DistanceResult   = ::fcl::DistanceResult<Scalar>;
using CollisionRequest = ::fcl::CollisionRequest<Scalar>;
using CollisionResult  = ::fcl::CollisionResult<Scalar>;

/**
 * Convert TrueForm mesh to FCL vertex and triangle vectors.
 *
 * @param r_polygons TrueForm polygon buffer (result from tf::read_stl)
 * @return Pair of (vertices, triangles) vectors for FCL
 */
template <typename PolygonBuffer>
std::pair<std::vector<Vec3>, std::vector<::fcl::Triangle>>
to_fcl_geometry(const PolygonBuffer& r_polygons) {
    auto points = r_polygons.points();
    auto faces = r_polygons.faces();
    auto n_pts = points.size();
    auto n_faces = faces.size();

    std::vector<Vec3> fcl_vertices;
    std::vector<::fcl::Triangle> fcl_triangles;

    if (n_pts == 0 || n_faces == 0) {
        return {fcl_vertices, fcl_triangles};
    }

    // Copy points to FCL vertices
    fcl_vertices.reserve(n_pts);
    for (const auto& p : points) {
        fcl_vertices.emplace_back(
            static_cast<Scalar>(p[0]),
            static_cast<Scalar>(p[1]),
            static_cast<Scalar>(p[2]));
    }

    // Copy triangle indices
    fcl_triangles.reserve(n_faces);
    for (const auto& f : faces) {
        if (f.size() != 3) {
            // Non-triangular face - return empty
            fcl_vertices.clear();
            fcl_triangles.clear();
            return {fcl_vertices, fcl_triangles};
        }
        fcl_triangles.emplace_back(
            static_cast<int>(f[0]),
            static_cast<int>(f[1]),
            static_cast<int>(f[2]));
    }

    return {fcl_vertices, fcl_triangles};
}

}  // namespace fcl
}  // namespace benchmark
