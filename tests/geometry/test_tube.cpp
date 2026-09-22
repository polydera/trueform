/**
 * @file test_tube.cpp
 * @brief Tests for curve frames and tube meshing at a closed curve's seam
 *
 * Tests for:
 * - make_curve_frames
 * - make_tube_mesh
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>
#include <array>
#include <cmath>
#include <cstddef>

namespace {

/// A unit square in the z = 0 plane as a closed polyline: the path revisits
/// its first vertex, so the stored curve carries the seam point twice.
auto tube_make_closed_square() -> tf::curves_buffer<int, float, 3> {
    tf::curves_buffer<int, float, 3> polyline;
    auto &points = polyline.points_buffer();
    points.allocate(4);
    points[0] = tf::make_point(0.f, 0.f, 0.f);
    points[1] = tf::make_point(1.f, 0.f, 0.f);
    points[2] = tf::make_point(1.f, 1.f, 0.f);
    points[3] = tf::make_point(0.f, 1.f, 0.f);

    auto &path = polyline.paths_buffer().data_buffer();
    auto &offsets = polyline.paths_buffer().offsets_buffer();
    offsets.push_back(0);
    for (int id : {0, 1, 2, 3, 0})
        path.push_back(id);
    offsets.push_back(int(path.size()));
    return polyline;
}

template <typename V0, typename V1>
auto tube_direction_gap(const V0 &a, const V1 &b) -> float {
    float dot = 0.f;
    for (std::size_t k = 0; k < 3; ++k)
        dot += a[k] * b[k];
    return std::abs(1.f - dot);
}

} // namespace

TEST_CASE("a closed curve's repeated point carries the first frame",
          "[geometry][tube]") {
    auto polyline = tube_make_closed_square();
    auto curve = *polyline.curves().begin();
    auto [tangents, normals, binormals] = tf::make_curve_frames(curve);

    const auto last = std::size_t(curve.size()) - 1;
    CHECK(tube_direction_gap(tangents[last], tangents[0]) < 1e-6f);
    CHECK(tube_direction_gap(normals[last], normals[0]) < 1e-6f);
    CHECK(tube_direction_gap(binormals[last], binormals[0]) < 1e-6f);

    for (std::size_t i = 0; i < 4; ++i) {
        auto prev = (i + 3) % 4;
        auto next = (i + 1) % 4;
        auto bisector = tf::make_unit_vector(curve[next] - curve[prev]);
        CHECK(tube_direction_gap(tangents[i], bisector) < 1e-6f);
    }
}

TEST_CASE("a square tube is uniform around its closure", "[geometry][tube]") {
    auto polyline = tube_make_closed_square();
    const float radius = 0.028284f;
    const int sides = 8;
    auto tube = tf::make_tube_mesh(polyline.curves(), radius, sides);

    REQUIRE(tube.points().size() == 4 * sides);
    REQUIRE(tube.faces().size() == 2 * 4 * sides);
    CHECK(tf::is_closed(tube.polygons()));
    CHECK(tf::is_manifold(tube.polygons()));

    // The strip between consecutive rings covers one side of the square;
    // by symmetry all four strips carry the same area, the seam's included.
    std::array<double, 4> strip_area{};
    auto points = tube.points();
    for (auto face : tube.faces()) {
        std::array<std::size_t, 3> ring{};
        for (std::size_t k = 0; k < 3; ++k)
            ring[k] = std::size_t(face[k]) / std::size_t(sides);
        auto low = std::min({ring[0], ring[1], ring[2]});
        auto high = std::max({ring[0], ring[1], ring[2]});
        auto strip = (high == 3 && low == 0) ? 3 : low;

        auto a = points[std::size_t(face[0])];
        auto b = points[std::size_t(face[1])];
        auto c = points[std::size_t(face[2])];
        auto ab = b - a;
        auto ac = c - a;
        auto n = tf::cross(ab, ac);
        strip_area[std::size_t(strip)] +=
            0.5 * std::sqrt(double(tf::dot(n, n)));
    }

    for (std::size_t s = 1; s < 4; ++s)
        CHECK(std::abs(strip_area[s] - strip_area[0]) < 1e-5);
}
