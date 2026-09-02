/**
 * @file test_face_orientation.cpp
 * @brief Winding-aware tests for orient_faces_consistently
 *
 * The oracle is the directed edge: a consistently oriented surface never
 * traverses the same directed edge (a, b) in two faces. Undirected edge
 * predicates such as is_manifold() are invariant under any winding and cannot
 * fail here.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "topology_generators.hpp"
#include "type_traits.hpp"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <trueform/trueform.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Mesh>
auto count_doubly_directed_edges(const Mesh &mesh) -> std::size_t {
    using index_t = std::decay_t<decltype(mesh.faces()[0][0])>;
    std::map<std::pair<index_t, index_t>, std::size_t> occurrences;
    const auto faces = mesh.faces();
    for (decltype(faces.size()) f = 0; f < faces.size(); ++f) {
        const auto &face = faces[f];
        const auto n = face.size();
        for (decltype(face.size()) prev = n - 1, i = 0; i < n; prev = i++)
            ++occurrences[{index_t(face[prev]), index_t(face[i])}];
    }
    std::size_t doubled = 0;
    for (const auto &entry : occurrences)
        if (entry.second > 1)
            ++doubled;
    return doubled;
}

template <typename Mesh> auto face_windings(const Mesh &mesh) {
    using index_t = std::decay_t<decltype(mesh.faces()[0][0])>;
    std::vector<std::vector<index_t>> windings;
    const auto faces = mesh.faces();
    windings.reserve(faces.size());
    for (decltype(faces.size()) f = 0; f < faces.size(); ++f)
        windings.emplace_back(faces[f].begin(), faces[f].end());
    return windings;
}

template <typename Mesh> auto face_vertex_sets(const Mesh &mesh) {
    auto sets = face_windings(mesh);
    for (auto &face : sets)
        std::sort(face.begin(), face.end());
    return sets;
}

template <typename Mesh> auto reverse_face(Mesh &mesh, std::size_t id) -> void {
    auto faces = mesh.faces();
    auto &&face = faces[id];
    std::reverse(face.begin(), face.end());
}

/**
 * @brief A triangulated Moebius band: manifold, with boundary, non-orientable.
 *
 * Rung i carries vertices a_i = 2i and b_i = 2i+1. Quad i spans rungs i and
 * i+1 and splits into (a_i, b_i, b_{i+1}) and (a_i, b_{i+1}, a_{i+1}). The
 * closing quad meets rung 0 with the sides exchanged, which is the half twist:
 * every interface around the band agrees except that one, so the parity cycle
 * is odd.
 */
template <typename Index, typename Real>
auto create_moebius_band_3d(Index rungs)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;

    const Real radius = Real(2);
    const Real half_width = Real(0.5);
    for (Index i = 0; i < rungs; ++i) {
        const Real u = tf::two_pi<Real> * Real(i) / Real(rungs);
        for (Index side = 0; side < 2; ++side) {
            const Real w = side == 0 ? -half_width : half_width;
            const Real r = radius + w * std::cos(u / Real(2));
            mesh.points_buffer().emplace_back(r * std::cos(u), r * std::sin(u),
                                              w * std::sin(u / Real(2)));
        }
    }

    for (Index i = 0; i < rungs; ++i) {
        const Index a = 2 * i;
        const Index b = 2 * i + 1;
        const bool closing = i + 1 == rungs;
        const Index next_a = closing ? Index(1) : Index(2 * (i + 1));
        const Index next_b = closing ? Index(0) : Index(2 * (i + 1) + 1);
        mesh.faces_buffer().emplace_back(a, b, next_b);
        mesh.faces_buffer().emplace_back(a, next_b, next_a);
    }
    return mesh;
}

template <typename Mesh> auto append_mesh(Mesh &dst, const Mesh &src) -> void {
    using index_t = std::decay_t<decltype(dst.faces()[0][0])>;
    const auto base = static_cast<index_t>(dst.points().size());
    for (decltype(src.points().size()) p = 0; p < src.points().size(); ++p)
        dst.points_buffer().emplace_back(src.points()[p][0], src.points()[p][1],
                                         src.points()[p][2]);
    for (decltype(src.faces().size()) f = 0; f < src.faces().size(); ++f)
        dst.faces_buffer().emplace_back(index_t(src.faces()[f][0]) + base,
                                        index_t(src.faces()[f][1]) + base,
                                        index_t(src.faces()[f][2]) + base);
}

/**
 * @brief A box on the integer lattice, wide enough that the squared area of a
 * face does not fit the coordinate type it is measured on - for int32 at the
 * cross product, for int64 at the squared length of it.
 */
template <typename Index, typename Coord>
auto create_lattice_box_3d() -> tf::polygons_buffer<Index, Coord, 3, 3> {
    auto reference = tf::make_box_mesh<Index>(2.0f, 2.0f, 2.0f);
    tf::polygons_buffer<Index, Coord, 3, 3> mesh;

    const Coord extent = Coord(7654321);
    for (decltype(reference.points().size()) p = 0;
         p < reference.points().size(); ++p)
        mesh.points_buffer().emplace_back(
            reference.points()[p][0] < 0 ? -extent : extent,
            reference.points()[p][1] < 0 ? -extent : extent,
            reference.points()[p][2] < 0 ? -extent : extent);
    for (decltype(reference.faces().size()) f = 0;
         f < reference.faces().size(); ++f)
        mesh.faces_buffer().emplace_back(Index(reference.faces()[f][0]),
                                         Index(reference.faces()[f][1]),
                                         Index(reference.faces()[f][2]));
    return mesh;
}

} // namespace

// =============================================================================
// Alternating box - one call must finish the job
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_alternating_box_single_call",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    for (std::size_t parity = 0; parity < 2; ++parity) {
        auto mesh =
            tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
        for (std::size_t f = parity; f < mesh.faces().size(); f += 2)
            reverse_face(mesh, f);

        const auto vertex_sets = face_vertex_sets(mesh);
        REQUIRE(count_doubly_directed_edges(mesh) == 14);

        REQUIRE(tf::orient_faces_consistently(mesh.polygons()));

        REQUIRE(count_doubly_directed_edges(mesh) == 0);
        REQUIRE(face_vertex_sets(mesh) == vertex_sets);
    }
}

// =============================================================================
// Shape coverage - open, closed, multi-component, already consistent
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_across_mesh_shapes",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto check = [](auto mesh, bool flip_alternating) {
        if (flip_alternating)
            for (std::size_t f = 0; f < mesh.faces().size(); f += 2)
                reverse_face(mesh, f);

        const auto vertex_sets = face_vertex_sets(mesh);
        REQUIRE(tf::orient_faces_consistently(mesh.polygons()));
        REQUIRE(count_doubly_directed_edges(mesh) == 0);
        REQUIRE(face_vertex_sets(mesh) == vertex_sets);
    };

    check(tf::test::create_two_triangles_3d<index_t, real_t>(), false);
    check(tf::test::create_inconsistent_winding_mesh_3d<index_t, real_t>(),
          false);
    check(tf::test::create_tetrahedron_3d<index_t, real_t>(), true);
    check(tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4), true);
    check(tf::test::create_two_components_3d<index_t, real_t>(), true);
}

// =============================================================================
// Idempotence - the second call changes nothing
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_is_idempotent",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    for (std::size_t f = 0; f < mesh.faces().size(); f += 2)
        reverse_face(mesh, f);

    REQUIRE(tf::orient_faces_consistently(mesh.polygons()));
    const auto settled = face_windings(mesh);

    REQUIRE(tf::orient_faces_consistently(mesh.polygons()));
    REQUIRE(face_windings(mesh) == settled);
}

// =============================================================================
// Randomized flip patterns - box and sphere, one call each
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_random_flip_patterns",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    std::mt19937 rng(20260902u);
    std::bernoulli_distribution flip(0.5);

    for (int trial = 0; trial < 50; ++trial) {
        auto mesh = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
        for (std::size_t f = 0; f < mesh.faces().size(); ++f)
            if (flip(rng))
                reverse_face(mesh, f);

        const auto vertex_sets = face_vertex_sets(mesh);
        REQUIRE(tf::orient_faces_consistently(mesh.polygons()));
        REQUIRE(count_doubly_directed_edges(mesh) == 0);
        REQUIRE(face_vertex_sets(mesh) == vertex_sets);
    }

    auto sphere =
        tf::make_sphere_mesh<index_t>(real_t(1), index_t(16), index_t(24));
    REQUIRE(sphere.faces().size() == 720);
    for (std::size_t f = 0; f < sphere.faces().size(); ++f)
        if (flip(rng))
            reverse_face(sphere, f);

    const auto sphere_vertex_sets = face_vertex_sets(sphere);
    REQUIRE(count_doubly_directed_edges(sphere) > 0);

    REQUIRE(tf::orient_faces_consistently(sphere.polygons()));

    REQUIRE(count_doubly_directed_edges(sphere) == 0);
    REQUIRE(face_vertex_sets(sphere) == sphere_vertex_sets);
}

// =============================================================================
// Weighted vote - the minority orientation is the one that moves
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_majority_wins_the_vote",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    for (std::size_t f = 1; f < mesh.faces().size(); ++f)
        reverse_face(mesh, f);

    const auto before = face_windings(mesh);
    REQUIRE(tf::orient_faces_consistently(mesh.polygons()));
    REQUIRE(count_doubly_directed_edges(mesh) == 0);

    const auto after = face_windings(mesh);
    std::size_t moved = 0;
    for (std::size_t f = 0; f < before.size(); ++f)
        if (before[f] != after[f])
            ++moved;
    REQUIRE(moved == 1);
    REQUIRE(before[0] != after[0]);
}

// =============================================================================
// Integral coordinates - the vote is not paid in an overflowing weight
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_votes_on_integral_lattice",
                   "[topology][orientation]", std::int32_t, std::int64_t)
{
    using coord_t = TestType;
    auto mesh = create_lattice_box_3d<std::int32_t, coord_t>();
    for (std::size_t f = 1; f < mesh.faces().size(); ++f)
        reverse_face(mesh, f);

    const auto before = face_windings(mesh);
    const auto vertex_sets = face_vertex_sets(mesh);

    REQUIRE(tf::orient_faces_consistently(mesh.polygons()));

    REQUIRE(count_doubly_directed_edges(mesh) == 0);
    REQUIRE(face_vertex_sets(mesh) == vertex_sets);

    const auto after = face_windings(mesh);
    std::size_t moved = 0;
    for (std::size_t f = 0; f < before.size(); ++f)
        if (before[f] != after[f])
            ++moved;
    REQUIRE(moved == 1);
    REQUIRE(before[0] != after[0]);
}

// =============================================================================
// Non-orientable surface - the verdict, and not one byte changed
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_reports_non_orientable",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_moebius_band_3d<index_t, real_t>(index_t(12));
    REQUIRE(mesh.faces().size() == 24);
    REQUIRE(tf::is_manifold(mesh.polygons()));

    const auto before = face_windings(mesh);
    REQUIRE_FALSE(tf::orient_faces_consistently(mesh.polygons()));
    REQUIRE(face_windings(mesh) == before);
}

// =============================================================================
// The component is the carrier - a bad component does not hold the good ones
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_repairs_orientable_components",
                   "[topology][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_moebius_band_3d<index_t, real_t>(index_t(12));
    const std::size_t moebius_faces = mesh.faces().size();
    const std::size_t moebius_seam = count_doubly_directed_edges(mesh);
    REQUIRE(moebius_seam > 0);

    for (int copy = 0; copy < 3; ++copy)
        append_mesh(mesh, tf::make_sphere_mesh<index_t>(
                              real_t(1), index_t(6), index_t(8)));

    std::mt19937 rng(20260902u);
    std::bernoulli_distribution flip(0.5);
    for (std::size_t f = moebius_faces; f < mesh.faces().size(); ++f)
        if (flip(rng))
            reverse_face(mesh, f);

    const auto before = face_windings(mesh);
    const auto vertex_sets = face_vertex_sets(mesh);

    REQUIRE_FALSE(tf::orient_faces_consistently(mesh.polygons()));

    REQUIRE(count_doubly_directed_edges(mesh) == moebius_seam);
    REQUIRE(face_vertex_sets(mesh) == vertex_sets);

    const auto after = face_windings(mesh);
    for (std::size_t f = 0; f < moebius_faces; ++f)
        REQUIRE(after[f] == before[f]);
}

// =============================================================================
// ensure_positive_orientation - the true volume after one call
// =============================================================================

TEMPLATE_TEST_CASE("ensure_positive_orientation_recovers_true_volume",
                   "[geometry][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    for (std::size_t f = 0; f < mesh.faces().size(); f += 2)
        reverse_face(mesh, f);

    REQUIRE(tf::ensure_positive_orientation(mesh.polygons()));

    REQUIRE(count_doubly_directed_edges(mesh) == 0);
    REQUIRE(tf::signed_volume(mesh.polygons()) == real_t(8));
}

TEMPLATE_TEST_CASE("ensure_positive_orientation_declines_non_orientable",
                   "[geometry][orientation]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_moebius_band_3d<index_t, real_t>(index_t(12));
    const auto before = face_windings(mesh);

    REQUIRE_FALSE(tf::ensure_positive_orientation(mesh.polygons()));
    REQUIRE(face_windings(mesh) == before);
}
