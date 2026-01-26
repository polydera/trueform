/**
 * Coal conversion utilities
 *
 * Helper functions for converting between TrueForm and Coal data structures.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <coal/fwd.hh>
#include <coal/data_types.h>
#include <coal/BV/AABB.h>
#include <coal/BV/OBB.h>
#include <coal/BV/OBBRSS.h>
#include <coal/BVH/BVH_model.h>
#include <coal/collision_object.h>
#include <coal/collision.h>
#include <coal/distance.h>
#include <vector>

namespace benchmark {
namespace coal {

// Coal uses double precision by default
using Scalar          = double;
using BV_AABB         = ::coal::AABB;
using BV_OBB          = ::coal::OBB;
using BV_OBBRSS       = ::coal::OBBRSS;
using BV              = BV_AABB;
using Model           = ::coal::BVHModel<BV>;
using Model_AABB      = ::coal::BVHModel<BV_AABB>;
using Model_OBB       = ::coal::BVHModel<BV_OBB>;
using Model_OBBRSS    = ::coal::BVHModel<BV_OBBRSS>;
using Vec3            = ::coal::Vec3s;
using Transform3      = ::coal::Transform3s;
using DistanceRequest  = ::coal::DistanceRequest;
using DistanceResult   = ::coal::DistanceResult;
using CollisionRequest = ::coal::CollisionRequest;
using CollisionResult  = ::coal::CollisionResult;

/**
 * Convert TrueForm mesh to Coal vertex and triangle vectors.
 *
 * @param r_polygons TrueForm polygon buffer (result from tf::read_stl)
 * @return Pair of (vertices, triangles) vectors for Coal
 */
template <typename PolygonBuffer>
std::pair<std::vector<Vec3>, std::vector<::coal::Triangle>>
to_coal_geometry(const PolygonBuffer& r_polygons) {
    auto points = r_polygons.points();
    auto faces = r_polygons.faces();
    auto n_pts = points.size();
    auto n_faces = faces.size();

    std::vector<Vec3> coal_vertices;
    std::vector<::coal::Triangle> coal_triangles;

    if (n_pts == 0 || n_faces == 0) {
        return {coal_vertices, coal_triangles};
    }

    // Copy points to Coal vertices
    coal_vertices.reserve(n_pts);
    for (const auto& p : points) {
        coal_vertices.emplace_back(
            static_cast<Scalar>(p[0]),
            static_cast<Scalar>(p[1]),
            static_cast<Scalar>(p[2]));
    }

    // Copy triangle indices
    coal_triangles.reserve(n_faces);
    for (const auto& f : faces) {
        if (f.size() != 3) {
            // Non-triangular face - return empty
            coal_vertices.clear();
            coal_triangles.clear();
            return {coal_vertices, coal_triangles};
        }
        coal_triangles.emplace_back(
            static_cast<size_t>(f[0]),
            static_cast<size_t>(f[1]),
            static_cast<size_t>(f[2]));
    }

    return {coal_vertices, coal_triangles};
}

}  // namespace coal
}  // namespace benchmark
