/**
 * @file test_euler_characteristic.cpp
 * @brief V - E + F on closed, open and multi-component surfaces
 *
 * A boundary edge belongs to one face only, so counting it demands the same
 * "who represents this undirected edge" answer an interior edge gets. These
 * fixtures pin the characteristic of surfaces whose boundary is not empty.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "topology_generators.hpp"
#include "type_traits.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <trueform/trueform.hpp>
#include <type_traits>

namespace {

template <typename Mesh>
auto append_euler_mesh(Mesh &dst, const Mesh &src) -> void {
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

/// @brief An uncapped tube: a quad ring split into triangles, two boundaries.
template <typename Index, typename Real>
auto create_open_tube_3d(Index segments)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    for (Index i = 0; i < segments; ++i) {
        const Real u = tf::two_pi<Real> * Real(i) / Real(segments);
        mesh.points_buffer().emplace_back(std::cos(u), std::sin(u), Real(0));
        mesh.points_buffer().emplace_back(std::cos(u), std::sin(u), Real(1));
    }
    for (Index i = 0; i < segments; ++i) {
        const Index a = 2 * i;
        const Index b = 2 * i + 1;
        const Index next = (i + 1) % segments;
        const Index c = 2 * next;
        const Index d = 2 * next + 1;
        mesh.faces_buffer().emplace_back(a, c, d);
        mesh.faces_buffer().emplace_back(a, d, b);
    }
    return mesh;
}

template <typename Index, typename Real>
auto create_punctured_sphere_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    auto sphere = tf::make_sphere_mesh<Index>(Real(1), Index(6), Index(8));
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    for (decltype(sphere.points().size()) p = 0; p < sphere.points().size(); ++p)
        mesh.points_buffer().emplace_back(sphere.points()[p][0],
                                          sphere.points()[p][1],
                                          sphere.points()[p][2]);
    for (decltype(sphere.faces().size()) f = 1; f < sphere.faces().size(); ++f)
        mesh.faces_buffer().emplace_back(Index(sphere.faces()[f][0]),
                                         Index(sphere.faces()[f][1]),
                                         Index(sphere.faces()[f][2]));
    return mesh;
}

template <typename Index, typename Real>
auto create_single_triangle_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    return mesh;
}

} // namespace

TEMPLATE_TEST_CASE("euler_characteristic_closed_surfaces",
                   "[topology][euler]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    CHECK(tf::euler_characteristic(box.polygons()) == 2);

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), index_t(6),
                                                index_t(8));
    CHECK(tf::euler_characteristic(sphere.polygons()) == 2);

    auto two_boxes = box;
    append_euler_mesh(two_boxes, box);
    CHECK(tf::euler_characteristic(two_boxes.polygons()) == 4);
}

TEMPLATE_TEST_CASE("euler_characteristic_surfaces_with_boundary",
                   "[topology][euler]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto triangle = create_single_triangle_3d<index_t, real_t>();
    CHECK(tf::euler_characteristic(triangle.polygons()) == 1);

    auto patch = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    CHECK(tf::euler_characteristic(patch.polygons()) == 1);

    auto punctured = create_punctured_sphere_3d<index_t, real_t>();
    CHECK(tf::euler_characteristic(punctured.polygons()) == 1);

    auto tube = create_open_tube_3d<index_t, real_t>(index_t(8));
    CHECK(tf::euler_characteristic(tube.polygons()) == 0);
}

TEMPLATE_TEST_CASE("euler_characteristic_reads_the_tagged_link",
                   "[topology][euler]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto tube = create_open_tube_3d<index_t, real_t>(index_t(8));
    auto link = tf::make_manifold_edge_link(tube.polygons());
    CHECK(tf::euler_characteristic(tube.polygons() | tf::tag(link)) ==
            tf::euler_characteristic(tube.polygons()));
}
