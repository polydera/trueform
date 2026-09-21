/**
 * @file test_face_quality.cpp
 * @brief Tests for per-element quality measures
 *
 * Tests for:
 * - compute_face_quality
 * - compute_dihedral_angles
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

template <typename Real> auto fq_degrees(tf::rad<Real> angle) -> Real {
    return tf::deg<Real>(angle).value;
}

template <typename Real> auto fq_angle_tolerance() -> Real {
    return std::is_same_v<Real, float> ? Real(1e-2) : Real(1e-9);
}

template <typename Real> auto fq_value_tolerance() -> Real {
    return std::is_same_v<Real, float> ? Real(1e-5) : Real(1e-12);
}

/// Equilateral, right isoceles, and a flat cap sliver, in that face order.
template <typename Index, typename Real>
auto fq_make_mixed_triangles() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0.5), std::sqrt(Real(3)) / Real(2),
                                        Real(0));

    result.points_buffer().emplace_back(Real(0), Real(0), Real(4));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(4));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(4));

    result.points_buffer().emplace_back(Real(0), Real(0), Real(8));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(8));
    result.points_buffer().emplace_back(Real(0.5), Real(1e-3), Real(8));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(3), Index(4), Index(5));
    result.faces_buffer().emplace_back(Index(6), Index(7), Index(8));

    return result;
}

template <typename Index, typename Real>
auto fq_make_unit_square() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

} // namespace

// =============================================================================
// compute_face_quality - hand-derived triangles
// =============================================================================

TEMPLATE_TEST_CASE("compute_face_quality_hand_derived", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = fq_make_mixed_triangles<index_t, real_t>();
    auto q = tf::compute_face_quality(mesh.polygons());

    REQUIRE(q.quality.size() == 3);
    REQUIRE(q.min_angle.size() == 3);
    REQUIRE(q.max_angle.size() == 3);
    REQUIRE(q.aspect_ratio.size() == 3);

    const real_t angle_tol = fq_angle_tolerance<real_t>();
    const real_t value_tol = fq_value_tolerance<real_t>();

    // equilateral: q = 1, every corner 60 degrees, every side equal
    REQUIRE(std::abs(q.quality[0] - real_t(1)) < value_tol);
    REQUIRE(std::abs(fq_degrees(q.min_angle[0]) - real_t(60)) < angle_tol);
    REQUIRE(std::abs(fq_degrees(q.max_angle[0]) - real_t(60)) < angle_tol);
    REQUIRE(std::abs(q.aspect_ratio[0] - real_t(1)) < value_tol);

    // right isoceles: 2A = 1, longest side^2 = 2, so q = (2/sqrt3) / 2
    REQUIRE(std::abs(q.quality[1] - real_t(0.5773502691896258)) < value_tol);
    REQUIRE(std::abs(fq_degrees(q.min_angle[1]) - real_t(45)) < angle_tol);
    REQUIRE(std::abs(fq_degrees(q.max_angle[1]) - real_t(90)) < angle_tol);
    REQUIRE(std::abs(q.aspect_ratio[1] - std::sqrt(real_t(2))) < value_tol);

    // cap sliver: 2A = 1e-3, longest side^2 = 1, so q = (2/sqrt3) * 1e-3 --
    // a vanishing quality the aspect ratio alone never sees (it stays near 2)
    REQUIRE(std::abs(q.quality[2] - real_t(1.1547005383792515e-3)) <
            real_t(1e-6));
    REQUIRE(fq_degrees(q.min_angle[2]) > real_t(0.1));
    REQUIRE(fq_degrees(q.min_angle[2]) < real_t(0.13));
    REQUIRE(fq_degrees(q.max_angle[2]) > real_t(179.7));
    REQUIRE(fq_degrees(q.max_angle[2]) < real_t(180));
    REQUIRE(std::abs(q.aspect_ratio[2] - real_t(2)) < real_t(1e-3));
}

// =============================================================================
// compute_face_quality - a face that is not a triangle
// =============================================================================

TEMPLATE_TEST_CASE("compute_face_quality_quad", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = fq_make_unit_square<index_t, real_t>();
    auto q = tf::compute_face_quality(mesh.polygons());

    REQUIRE(q.quality.size() == 1);

    const real_t angle_tol = fq_angle_tolerance<real_t>();
    const real_t value_tol = fq_value_tolerance<real_t>();

    // quality is a triangle measure, so a quad has none
    REQUIRE(q.quality[0] == real_t(-1));
    REQUIRE(std::abs(fq_degrees(q.min_angle[0]) - real_t(90)) < angle_tol);
    REQUIRE(std::abs(fq_degrees(q.max_angle[0]) - real_t(90)) < angle_tol);
    REQUIRE(std::abs(q.aspect_ratio[0] - real_t(1)) < value_tol);
}

// =============================================================================
// compute_face_quality - every face of a sphere
// =============================================================================

TEMPLATE_TEST_CASE("compute_face_quality_sphere", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);
    auto q = tf::compute_face_quality(sphere.polygons());

    REQUIRE(q.quality.size() == sphere.polygons().size());

    for (decltype(q.quality.size()) i = 0; i < q.quality.size(); ++i) {
        REQUIRE(q.quality[i] > real_t(0));
        REQUIRE(q.quality[i] <= real_t(1) + fq_value_tolerance<real_t>());
        REQUIRE(q.min_angle[i] <= q.max_angle[i]);
        REQUIRE(fq_degrees(q.min_angle[i]) > real_t(0));
        REQUIRE(fq_degrees(q.max_angle[i]) < real_t(180));
        REQUIRE(q.aspect_ratio[i] >= real_t(1) - fq_value_tolerance<real_t>());
    }
}

// =============================================================================
// compute_dihedral_angles - a box folds at 90 degrees, a face is flat at 0
// =============================================================================

TEMPLATE_TEST_CASE("compute_dihedral_angles_box", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    auto d = tf::compute_dihedral_angles(box.polygons());

    // 12 box edges and the 6 diagonals splitting its square faces
    REQUIRE(d.angles.size() == 18);
    REQUIRE(std::size_t(d.edges.size()) == d.angles.size());

    const real_t angle_tol = fq_angle_tolerance<real_t>();
    std::size_t flat = 0;
    std::size_t right = 0;
    for (decltype(d.angles.size()) i = 0; i < d.angles.size(); ++i) {
        auto degrees = fq_degrees(d.angles[i]);
        if (std::abs(degrees) < angle_tol)
            ++flat;
        else if (std::abs(degrees - real_t(90)) < angle_tol)
            ++right;
        REQUIRE(d.edges[i][0] < d.edges[i][1]);
    }
    REQUIRE(flat == 6);
    REQUIRE(right == 12);
}

// =============================================================================
// compute_dihedral_angles - a flat plane turns through nothing
// =============================================================================

TEMPLATE_TEST_CASE("compute_dihedral_angles_plane", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto plane = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 5, 5);
    auto d = tf::compute_dihedral_angles(plane.polygons());

    REQUIRE(d.angles.size() > 0);

    const real_t angle_tol = fq_angle_tolerance<real_t>();
    for (decltype(d.angles.size()) i = 0; i < d.angles.size(); ++i)
        REQUIRE(std::abs(fq_degrees(d.angles[i])) < angle_tol);
}

// =============================================================================
// compute_dihedral_angles - a tagged form answers the same
// =============================================================================

TEMPLATE_TEST_CASE("compute_dihedral_angles_tagged", "[geometry][quality]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 12, 12);
    auto bare = tf::compute_dihedral_angles(sphere.polygons());

    auto link = tf::make_manifold_edge_link(sphere.polygons());
    auto normals = tf::compute_normals(sphere.polygons());
    auto tagged = tf::compute_dihedral_angles(
        sphere.polygons() | tf::tag(link) | tf::tag_normals(normals.unit_vectors()));

    REQUIRE(tagged.angles.size() == bare.angles.size());
    for (decltype(bare.angles.size()) i = 0; i < bare.angles.size(); ++i) {
        REQUIRE(tagged.edges[i][0] == bare.edges[i][0]);
        REQUIRE(tagged.edges[i][1] == bare.edges[i][1]);
        REQUIRE(std::abs(fq_degrees(tagged.angles[i]) -
                         fq_degrees(bare.angles[i])) <
                fq_angle_tolerance<real_t>());
    }
}
