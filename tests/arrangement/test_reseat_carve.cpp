/**
 * @file test_reseat_carve.cpp
 * @brief A carved arrangement answers in the caller's id space.
 *
 * Two edges cross one plane a third of a lattice unit apart, so the gate
 * fuses their crossings into one identity, which the second edge does not
 * hold: the carve splits that edge at the identity's lattice point inside
 * the one build. The graph still states one slot per caller face, every
 * original corner a caller vertex and the added point a created identity,
 * and the carved face's slot is spanned by its pieces. A seat within the
 * input's quantum of a carrier corner is not carved, and the operands keep
 * their shape.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/reseat/find_retirement_entrants.hpp>
#include <trueform/arrangement/reseat/reseat_refusal.hpp>
#include <trueform/arrangement/reseat/state_reseat_pieces.hpp>
#include <trueform/arrangement/reseat/state_reseat_reach.hpp>
#include <trueform/core/aabb_from.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/pt_converter.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/plane_identity_collapse.hpp>
#include <trueform/intersect/graph/vertex.hpp>

#include "arrangement_builders.hpp"
#include "tagged_operand.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using carve_index_t = int;
template <typename Real>
using carve_mesh_t = tf::polygons_buffer<carve_index_t, Real, 3, 3>;

template <typename Real>
auto carve_triangle(std::array<Real, 3> a, std::array<Real, 3> b,
                    std::array<Real, 3> c) -> carve_mesh_t<Real> {
  carve_mesh_t<Real> mesh;
  for (const auto &q : {a, b, c})
    mesh.points_buffer().emplace_back(q[0], q[1], q[2]);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  return mesh;
}

/// The scene centred on the origin: edge one on the z axis, the plane
/// `z = 0`, and edge two from `(u, 0, low u)` to `(-u, 0, -2 low u)`,
/// which crosses the plane a third of a lattice unit off the axis. `low`
/// is in lattice units below the plane.
template <typename Real>
auto carve_scene(Real low) -> std::vector<carve_mesh_t<Real>> {
  std::vector<carve_mesh_t<Real>> scene{
      carve_triangle<Real>({0, 0, -1}, {0, 0, 1}, {5, 3, Real(0.5)}),
      carve_triangle<Real>({-10, -10, 0}, {10, -10, 0}, {0, 10, 0})};
  // the lattice unit of the box the scene spans: the third operand stays
  // inside it, so the converter is the one the arrangement states
  using Int = tf::exact::resolve_int_type<tf::none_t, Real>;
  auto box = tf::aabb_from(scene[1].polygons().points());
  box.min[2] = -1;
  box.max[2] = 1;
  const auto converter = tf::exact::make_pt_converter<Int, Real>(box);
  const Real u = Real(1) / converter.scale;
  scene.push_back(carve_triangle<Real>({u, 0, low * u}, {-u, 0, -2 * low * u},
                                       {-4, -6, Real(0.5)}));
  return scene;
}

} // namespace

TEMPLATE_TEST_CASE("reseat carve: a carved graph answers in the caller's id "
                   "space",
                   "[arrangement][reseat]", float, double) {
  std::vector<tf::test::tagged_operand<carve_index_t, TestType>> operands;
  for (auto &mesh : carve_scene<TestType>(-1000))
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.policy().reseat_census().carves == 1);
  REQUIRE(graph.policy().reseat_census().refusal ==
          tf::arrangement::reseat::reseat_refusal::none);
  REQUIRE(graph.failed().size() == 0);

  const auto &ga = graph.global();
  const auto descriptors = ga.exposed_descriptors();
  REQUIRE(descriptors.size() == 3);
  graph.apply_to_form()(2, [](const auto &form) {
    REQUIRE(form.faces().size() == 1);
    REQUIRE(form.points().size() == 3);
  });

  using vsource = tf::intersect::graph::vertex_source;
  for (const auto &triangle : ga.exposed_tris())
    for (const auto &corner : triangle) {
      if (corner.source == vsource::created) {
        REQUIRE(std::size_t(corner.id) < graph.created_points().size());
        continue;
      }
      const auto tag = graph.tag_of_flat(corner.id);
      const auto local = corner.id - graph.vertex_offsets()[std::size_t(tag)];
      graph.apply_to_form()(tag, [&](const auto &form) {
        REQUIRE(std::size_t(local) < std::size_t(form.points().size()));
      });
    }

  // the carved face is cut, and every piece of it spans its slot
  const auto carved = descriptors[2];
  REQUIRE(carved.plane != -1);
  const auto span = graph.slot_range(2);
  REQUIRE(span[1] - span[0] >= 2);
}

TEMPLATE_TEST_CASE("reseat carve: a seat within the input's quantum of a "
                   "carrier corner is not carved",
                   "[arrangement][reseat]", float, double) {
  std::vector<tf::test::tagged_operand<carve_index_t, TestType>> operands;
  for (auto &mesh : carve_scene<TestType>(-30))
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.identities().local_merges().size() >= 1);
  REQUIRE(graph.policy().reseat_census().carves == 0);
  REQUIRE(graph.global().exposed_descriptors().size() == 3);
}

// The carved face holds an original another face of its form duplicates on
// the same lattice point, and the world names only the carved one. The
// coincidence stands in the world, so the carve states nothing of it:
// neither original retires, and no row of the carved face speaks the
// duplicate's name.
TEMPLATE_TEST_CASE("reseat carve: a coincidence of originals the world holds "
                   "stays the gate's",
                   "[arrangement][reseat]", float, double) {
  auto scene = carve_scene<TestType>(-1000);
  {
    const auto source = scene[2].polygons().points();
    carve_mesh_t<TestType> mesh;
    const auto add = [&mesh](const auto &q) {
      mesh.points_buffer().emplace_back(q[0], q[1], q[2]);
    };
    add(source[2]);
    add(source[0]);
    add(source[1]);
    add(source[2]);
    add(std::array<TestType, 3>{-6, -8, TestType(0.5)});
    add(std::array<TestType, 3>{-8, -6, TestType(0.5)});
    mesh.faces_buffer().emplace_back(1, 2, 3);
    mesh.faces_buffer().emplace_back(0, 4, 5);
    scene[2] = std::move(mesh);
  }
  std::vector<tf::test::tagged_operand<carve_index_t, TestType>> operands;
  for (auto &mesh : scene)
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.policy().reseat_census().carves == 1);
  REQUIRE(graph.policy().reseat_census().refusal ==
          tf::arrangement::reseat::reseat_refusal::none);
  REQUIRE(graph.failed().size() == 0);

  const auto &world = graph.world();
  const auto &merges = graph.identities().local_merges();
  const auto base = world.vertex_offsets()[2];
  REQUIRE(tf::intersect::graph::plane_merge_target(merges, carve_index_t(0),
                                                   base) == carve_index_t(-1));
  REQUIRE(tf::intersect::graph::plane_merge_target(merges, carve_index_t(0),
                                                   base + carve_index_t(3)) ==
          carve_index_t(-1));

  using vsource = tf::intersect::graph::vertex_source;
  const auto &ga = graph.global();
  const auto span = graph.slot_range(2);
  REQUIRE(span[1] - span[0] >= 2);
  for (auto e = span[0]; e < span[1]; ++e)
    for (const auto &corner : ga.exposed_tris()[e])
      REQUIRE((corner.source == vsource::created || corner.id != base));
}

// Two forms stand a vertex each on one lattice point of the plane the carve
// restates, away from its seat, and the world identifies them: their faces'
// loops name the point by its anchor. The delta names the same coincidence
// again, and the carve speaks the world's anchor for it without retiring
// either original.
TEMPLATE_TEST_CASE("reseat carve: an identified vertex the carve restates "
                   "keeps the world's name",
                   "[arrangement][reseat]", float, double) {
  auto scene = carve_scene<TestType>(-1000);
  scene.push_back(carve_triangle<TestType>({2, -3, 0}, {4, -3, 0}, {3, -3, 2}));
  scene.push_back(carve_triangle<TestType>({2, -3, 0}, {2, -5, 1}, {1, -4, 1}));
  std::vector<tf::test::tagged_operand<carve_index_t, TestType>> operands;
  for (auto &mesh : scene)
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  const auto &census = graph.policy().reseat_census();
  REQUIRE(census.carves == 1);
  REQUIRE(census.refusal == tf::arrangement::reseat::reseat_refusal::none);
  REQUIRE(graph.failed().size() == 0);

  const auto &world = graph.world();
  const auto offsets = world.vertex_offsets();
  // the premise: the world identifies the two vertices and the carve's delta
  // held their faces
  REQUIRE(world.intersections().canonical_vertex(
              std::int16_t(4), carve_index_t(0)) == offsets[3]);
  REQUIRE(census.reach.neighbours + census.reach.restated >= 3);

  CHECK(census.reach.foreign_retirements == 0);
  const auto &merges = graph.identities().local_merges();
  for (const auto flat :
       {offsets[3], offsets[3] + carve_index_t(1), offsets[4]})
    CHECK(tf::intersect::graph::plane_merge_target(merges, carve_index_t(0),
                                                   flat) == carve_index_t(-1));
}

namespace {

/// The scene on a lattice coarse enough that every lattice point near it is
/// a float: a distant form centred on the origin spans the box. The carve
/// splits the third form at the origin, and the fourth, a plane leaning across
/// the new side the split draws, crosses that side at a point the cut world
/// never names.
auto coarse_carve_scene() -> std::vector<carve_mesh_t<float>> {
  carve_mesh_t<float> distant;
  for (const float s : {1.0f, -1.0f}) {
    const auto b = carve_index_t(distant.points().size());
    distant.points_buffer().emplace_back(4096 * s, 4096 * s, 4096 * s);
    distant.points_buffer().emplace_back(4095 * s, 4096 * s, 4096 * s);
    distant.points_buffer().emplace_back(4096 * s, 4095 * s, 4096 * s);
    distant.faces_buffer().emplace_back(b, b + 1, b + 2);
  }
  using Int = tf::exact::resolve_int_type<tf::none_t, float>;
  const auto converter = tf::exact::make_pt_converter<Int, float>(
      tf::aabb_from(distant.polygons().points()));
  const float u = 1.0f / converter.scale;
  std::vector<carve_mesh_t<float>> scene{
      carve_triangle<float>({0, 0, -1}, {0, 0, 1}, {5, 3, 0.5f}),
      carve_triangle<float>({-10, -10, 0}, {10, -10, 0}, {0, 10, 0}),
      carve_triangle<float>({u, 0, -1000 * u}, {-u, 0, 2000 * u},
                            {-4, -6, 0.5f}),
      carve_triangle<float>({-3.2f, -8, 2.05f}, {-3.15f, 4, 1.95f},
                            {1.8f, -2, -3})};
  scene.push_back(std::move(distant));
  return scene;
}

} // namespace

// A class the carve's delta creates rounds onto the lattice point of an
// original the world names and no record lands on. The delta joins the two,
// so the carve seats there; no created identity stands at the seat, so it
// mints, and the original retires into the mint — the weld the entrance
// reads.
TEST_CASE("reseat carve: a mint landing on an original retires it",
          "[arrangement][reseat]") {
  using int_t = tf::exact::resolve_int_type<tf::none_t, float>;
  using lattice_t = tf::point<int_t, 3>;
  using vec_t = std::array<double, 3>;
  const auto sub = [](const lattice_t &x, const lattice_t &y) {
    return vec_t{double(x[0]) - y[0], double(x[1]) - y[1], double(x[2]) - y[2]};
  };
  const auto dot = [](const vec_t &x, const vec_t &y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  const auto cross = [](const vec_t &x, const vec_t &y) {
    return vec_t{x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2],
                 x[0] * y[1] - x[1] * y[0]};
  };
  const auto unit_of = [&dot](vec_t x) {
    const double n = std::sqrt(dot(x, x));
    for (auto &c : x)
      c /= n;
    return x;
  };
  lattice_t landing{};
  float unit = 0;
  std::array<std::array<lattice_t, 3>, 2> walls{};
  {
    auto scene = coarse_carve_scene();
    std::vector<tf::test::tagged_operand<carve_index_t, float>> operands;
    for (auto &mesh : scene)
      operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
    auto forms = tf::test::tagged_forms(operands);
    const auto graph = tf::test::build_range_arrangement(
        tf::test::forms_range(forms), tf::arrangement_config{});
    REQUIRE(graph.policy().reseat_census().carves == 1);
    const auto &converter = graph.converter();
    unit = converter.deconvert(lattice_t{1, 0, 0})[0] -
           converter.deconvert(lattice_t{0, 0, 0})[0];
    for (std::size_t w = 0; w < 2; ++w)
      graph.apply_to_form()(carve_index_t(2 + w), [&](const auto &form) {
        for (std::size_t c = 0; c < 3; ++c)
          walls[w][c] = converter.convert(form.points()[form.faces()[0][c]]);
      });
    // the new side runs from the seat at the origin to the third form's far
    // corner, and the leaning plane crosses it
    const lattice_t origin{0, 0, 0};
    const auto corner = sub(walls[0][2], origin);
    const auto n =
        cross(sub(walls[1][1], walls[1][0]), sub(walls[1][2], walls[1][0]));
    const double t = dot(n, sub(walls[1][0], origin)) / dot(n, corner);
    bool found = false;
    for (const auto &p : graph.created_points()) {
      bool near_crossing = true;
      for (std::size_t i = 0; i < 3; ++i)
        near_crossing =
            near_crossing && std::abs(double(p[i]) - t * corner[i]) < 1;
      if (near_crossing) {
        REQUIRE_FALSE(found);
        found = true;
        landing = p;
      }
    }
    REQUIRE(found);
  }
  // an original on the landing point, its face leaving it parallel to both
  // walls and away from each on the side the point stands, down through the
  // plane the carve restates, so a record names the face far from the point
  REQUIRE(tf::exact::orient3d_sign(walls[0][0], walls[0][1], walls[0][2],
                                   landing) != 0);
  REQUIRE(tf::exact::orient3d_sign(walls[1][0], walls[1][1], walls[1][2],
                                   landing) != 0);
  std::array<vec_t, 2> normals{};
  std::array<double, 2> sides{};
  for (std::size_t w = 0; w < 2; ++w) {
    normals[w] = unit_of(
        cross(sub(walls[w][1], walls[w][0]), sub(walls[w][2], walls[w][0])));
    sides[w] = dot(normals[w], sub(landing, walls[w][0])) < 0 ? -1.0 : 1.0;
  }
  auto along = unit_of(cross(normals[0], normals[1]));
  if (along[2] > 0)
    for (auto &c : along)
      c = -c;
  std::array<vec_t, 2> leave{};
  for (std::size_t w = 0; w < 2; ++w) {
    const auto &other = normals[1 - w];
    const double k = dot(normals[w], other);
    for (std::size_t i = 0; i < 3; ++i)
      leave[w][i] = normals[w][i] - k * other[i];
    leave[w] = unit_of(leave[w]);
  }
  const std::array<float, 3> o{float(landing[0]) * unit,
                               float(landing[1]) * unit,
                               float(landing[2]) * unit};
  std::array<float, 3> q1{}, q2{};
  for (std::size_t i = 0; i < 3; ++i) {
    const double away =
        64 * double(unit) * (sides[0] * leave[0][i] + sides[1] * leave[1][i]);
    q1[i] = o[i] + float(0.5 * along[i] + away);
    q2[i] = o[i] + float(0.5 * along[i] + 0.05 * sides[0] * leave[0][i] + away);
  }
  auto scene = coarse_carve_scene();
  scene.push_back(carve_triangle<float>(o, q1, q2));
  std::vector<tf::test::tagged_operand<carve_index_t, float>> operands;
  for (auto &mesh : scene)
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.policy().reseat_census().carves == 1);
  REQUIRE(graph.policy().reseat_census().refusal ==
          tf::arrangement::reseat::reseat_refusal::none);
  REQUIRE(graph.failed().size() == 0);
  const carve_index_t landed = 5;
  REQUIRE(graph.converter().convert(o) == landing);
  REQUIRE(o[2] > 0);
  REQUIRE(q1[2] < 0);
  // the premise: a record names the original's face, none at the original
  std::size_t records = 0;
  for (const auto &block : graph.world().intersections().intersections(landed))
    for (const auto &record : block) {
      ++records;
      REQUIRE_FALSE((record.target.label == tf::topo_type::vertex &&
                     record.target.id == carve_index_t(0)));
    }
  REQUIRE(records != 0);

  const auto target = tf::intersect::graph::plane_merge_target(
      graph.identities().local_merges(), carve_index_t(0),
      graph.vertex_offsets()[std::size_t(landed)]);
  REQUIRE(target != carve_index_t(-1));
  REQUIRE(std::size_t(target) < graph.created_points().size());
  CHECK(graph.created_points()[std::size_t(target)] == landing);
}

// A carve founds new planes and a wave entrant is seated after it: the
// standing planes' names are the world's, so a plane whose first member is
// a piece is never named off the piece's corners.
TEST_CASE("reseat carve: a wave entrant is seated beside the planes a carve "
          "founded",
          "[arrangement][reseat]") {
  carve_mesh_t<double> surface;
  surface.points_buffer().emplace_back(-0.791000009, 0.93064099600000005,
                                       0.75900000300000003);
  surface.points_buffer().emplace_back(
      -0.52423996775321835, 0.93064099600000005, 0.63649168642092468);
  surface.points_buffer().emplace_back(-0.52342701000000003, 0.926259995,
                                       0.63691699499999999);
  surface.points_buffer().emplace_back(
      -0.52167193040744209, 0.93064099600000005, 0.63323328887725294);
  surface.points_buffer().emplace_back(
      -0.52167193040744197, 0.93064099600000005, 0.63323328887725294);
  surface.points_buffer().emplace_back(
      -0.37298700200000001, 0.80484598900000004, 0.69194400299999992);
  surface.points_buffer().emplace_back(
      -0.35141106166147607, 0.82876556408752489, 0.64425581597286041);
  surface.points_buffer().emplace_back(
      -0.35141106166147595, 0.82876556408752489, 0.64425581597286063);
  surface.faces_buffer().emplace_back(3, 0, 1);
  surface.faces_buffer().emplace_back(4, 0, 3);
  surface.faces_buffer().emplace_back(6, 5, 7);
  surface.faces_buffer().emplace_back(1, 2, 3);
  carve_mesh_t<double> tool;
  tool.points_buffer().emplace_back(-0.55008900199999999, 0.92509299499999997,
                                    0.68855500199999997);
  tool.points_buffer().emplace_back(-0.55008900199999999, 0.96209001500000002,
                                    0.56855499700000001);
  tool.points_buffer().emplace_back(-0.53595298499999999, 0.881826997,
                                    0.73312598500000004);
  tool.points_buffer().emplace_back(-0.37298700200000001, 0.80484598900000004,
                                    0.69194400300000003);
  tool.points_buffer().emplace_back(-0.37298700200000001, 0.84184300899999998,
                                    0.57194399799999995);
  tool.points_buffer().emplace_back(-0.31032899000000003, 0.81572097499999996,
                                    0.68819701700000002);
  tool.faces_buffer().emplace_back(0, 1, 2);
  tool.faces_buffer().emplace_back(3, 4, 5);
  std::vector<tf::test::tagged_operand<carve_index_t, double>> operands;
  operands.push_back(tf::test::make_tagged_operand(std::move(surface)));
  operands.push_back(tf::test::make_tagged_operand(std::move(tool)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.policy().reseat_census().carves == 1);
  REQUIRE(graph.policy().reseat_census().refusal ==
          tf::arrangement::reseat::reseat_refusal::none);
  REQUIRE(graph.arrangement().census().entrant_planes != 0);
  REQUIRE(graph.failed().size() == 0);
  // the precondition: some standing plane's first member is a piece
  const auto &world = graph.world();
  bool piece_first = false;
  for (carve_index_t plane = 0; plane < world.n_planes(); ++plane) {
    if (world.member_count(plane) == 0)
      continue;
    const auto &d = world.descriptor(world.member(plane, carve_index_t(0)));
    graph.apply_to_form()(carve_index_t(d.tag), [&](const auto &form) {
      piece_first = piece_first ||
                    std::size_t(d.object) >= std::size_t(form.faces().size());
    });
  }
  REQUIRE(piece_first);
}

// A carve's weld moves a created identity that survives on a split root of
// the standing world: the root's survivors speak the grown rewrite table,
// so none is an identity the table retired.
TEST_CASE("reseat carve: the split survivors speak the carve's welds",
          "[arrangement][reseat]") {
  std::vector<tf::test::tagged_operand<carve_index_t, float>> operands;
  {
    carve_mesh_t<float> mesh;
    mesh.points_buffer().emplace_back(0.0f, 0.0f, -1.0f);
    mesh.points_buffer().emplace_back(0.0f, 0.0f, 1.0f);
    mesh.points_buffer().emplace_back(5.0f, 3.0f, 0.5f);
    mesh.faces_buffer().emplace_back(0, 1, 2);
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  }
  {
    carve_mesh_t<float> mesh;
    mesh.points_buffer().emplace_back(-10.0f, -10.0f, 0.0f);
    mesh.points_buffer().emplace_back(10.0f, -10.0f, 0.0f);
    mesh.points_buffer().emplace_back(0.0f, 10.0f, 0.0f);
    mesh.faces_buffer().emplace_back(0, 1, 2);
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  }
  {
    carve_mesh_t<float> mesh;
    mesh.points_buffer().emplace_back(1.49011612e-08f, 0.0f, -1.55364814e-05f);
    mesh.points_buffer().emplace_back(-1.49011612e-08f, 0.0f, 3.10729629e-05f);
    mesh.points_buffer().emplace_back(-4.0f, -6.0f, 0.5f);
    mesh.faces_buffer().emplace_back(0, 1, 2);
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  }
  {
    carve_mesh_t<float> mesh;
    mesh.points_buffer().emplace_back(1.23883545f, -0.54395473f, 0.136364043f);
    mesh.points_buffer().emplace_back(-1.55993366f, 3.48246241f, -0.336503685f);
    mesh.points_buffer().emplace_back(-2.55103517f, 0.340344608f,
                                      -0.268971354f);
    mesh.faces_buffer().emplace_back(0, 1, 2);
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  }
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  REQUIRE(graph.policy().reseat_census().carves == 1);
  const auto &world = graph.world();
  REQUIRE(world.split_roots().size() != 0);
  for (std::size_t root = 0; root < world.split_roots().size(); ++root)
    for (const auto id : world.split_survivors(root))
      REQUIRE(tf::intersect::graph::plane_merge_target(
                  graph.identities().local_merges(), carve_index_t(1), id) ==
              carve_index_t(-1));
}

// A carve that cuts nothing states no piece: a face target whose point the
// face does not hold places nothing, which the carve refuses as `no_pieces`.
TEST_CASE("reseat carve: a target no face holds places no piece",
          "[arrangement][reseat]") {
  std::vector<tf::test::tagged_operand<carve_index_t, double>> operands;
  for (auto &mesh : carve_scene<double>(-1000))
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  using int_t = typename std::decay_t<decltype(graph)>::resolved_int_type;
  using corner_t = tf::arrangement::reseat::piece_corner<carve_index_t>;
  const auto apply_to_form = graph.apply_to_form();
  const auto reader = graph.lattice().reader(apply_to_form);
  tf::arrangement::reseat::carve_round<carve_index_t, int_t> round;
  // the plane `z = 0` and a point inside the scene but off its triangle
  round.faces.push_back(
      {carve_index_t(1),
       carve_index_t(0),
       {corner_t{std::int16_t(1), carve_index_t(0)},
        corner_t{std::int16_t(1), carve_index_t(1)},
        corner_t{std::int16_t(1), carve_index_t(2)}},
       graph.converter().convert(tf::point<double, 3>{8, 8, 0})});
  std::vector<decltype(round)> rounds{round};
  const tf::arrangement::reseat::standing_pieces<carve_index_t> standing;
  const auto pieces =
      tf::arrangement::reseat::state_reseat_pieces<carve_index_t, int_t>(
          rounds, standing, graph.n_tags(), apply_to_form, reader,
          graph.created_points());
  REQUIRE(pieces.pieces.size() == 0);
  REQUIRE(pieces.carved.size() == 0);
}

// A face the carve cut answers its own retirement: a weld retiring one of
// its originals reaches no face the carve does not already state.
TEST_CASE("reseat carve: a carved face answers its own retirement",
          "[arrangement][reseat]") {
  std::vector<tf::test::tagged_operand<carve_index_t, double>> operands;
  for (auto &mesh : carve_scene<double>(-1000))
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  // a face far from the scene, which no record names
  operands.push_back(tf::test::make_tagged_operand(carve_triangle<double>(
      {100, 100, 100}, {101, 100, 100}, {100, 101, 100})));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::arrangement_config{});
  const auto &world = graph.world();
  const carve_index_t remote = carve_index_t(3);
  struct package_t {
    tf::buffer<std::array<carve_index_t, 3>> welds;
    tf::buffer<std::array<carve_index_t, 2>> piece_sources;
    tf::buffer<tf::intersect::graph::face_descriptor<carve_index_t>>
        piece_descriptors;
  } package;
  package.welds.push_back({carve_index_t(0),
                           world.vertex_offsets()[std::size_t(remote)],
                           carve_index_t(0)});
  package.piece_sources.push_back({remote, carve_index_t(0)});
  package.piece_descriptors.push_back({remote, carve_index_t(1)});
  const tf::arrangement::reseat::world_face_lookup<
      carve_index_t, std::decay_t<decltype(world)>>
      world_face_of(world);
  REQUIRE(world_face_of(remote, carve_index_t(0)) == carve_index_t(-1));
  const auto never_named = [](auto, auto) { return false; };
  const auto reached =
      tf::arrangement::reseat::find_retirement_entrants<carve_index_t>(
          world, world_face_of, package, graph.apply_to_form(), never_named);
  REQUIRE(reached.size() == 0);
  // the same retirement on a face the carve never cut reaches it
  package.piece_sources.clear();
  package.piece_descriptors.clear();
  const auto uncarved =
      tf::arrangement::reseat::find_retirement_entrants<carve_index_t>(
          world, world_face_of, package, graph.apply_to_form(), never_named);
  REQUIRE(uncarved.size() == 1);
  REQUIRE(uncarved[0] ==
          std::array<carve_index_t, 2>{remote, carve_index_t(0)});
}
