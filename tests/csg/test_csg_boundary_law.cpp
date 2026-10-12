/**
 * @file test_csg_boundary_law.cpp
 * @brief The boundary law: a face emits the chain its own loop bounds
 *
 * A needle — a thin pyramid of 32 slivers meeting at its apex — has its tip
 * cut by a box face a few dozen lattice units above the apex. There every
 * sliver is narrower than one lattice unit, so the rounding of the cut's
 * points carries sides of one sliver across each other; the recovery wave
 * states the crossing on both sides, over several rounds, and the sliver's
 * loop comes back touching itself around a lobe the rounding turned inside
 * out. Emitted in the face's own winding, that lobe runs a side its neighbour
 * also runs the same way: the surface folds, the flood joins two sides of one
 * surface, and components of two closed operands come out self-sided. Emitted
 * the way the loop winds around it, every side is run once in each direction.
 *
 * Two oracles read the law. The arrangement's own: every edge of every
 * operand's emitted surface — the exposed triangles, and the faces no cut
 * reached — is run as often one way as the other. The classification's: every
 * component of a closed operand separates two domains. The census proves the
 * scene reaches the mechanism: the wave crossed sides and ran more than one
 * round.
 *
 * The scene is built from IEEE-exact operations only — a rational
 * parametrization of the circle, a dyadic radius, shear and cut height — so
 * its lattice is the same bits on every platform, and the bounding box is
 * symmetric about the apex, so the cut height reaches the lattice whole.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/csg.hpp>
#include <trueform/intersect/graph/vertex.hpp>

#include "csg_builders.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

/// The needle's apex is the origin and its ring stands at z = 4: `k` points
/// of the unit circle at parameters `(i - k / 2) * step / 1024`, scaled by
/// 2^-radius_exp and sheared by (shear_x, shear_y) / 16 along z.
template <typename Index, typename Real>
auto boundary_law_needle(int k, int step, int radius_exp, int shear_x,
                         int shear_y) -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  mesh.points_buffer().allocate(std::size_t(k + 1));
  for (int c = 0; c < 3; ++c)
    mesh.points_buffer()[0][c] = Real(0);
  const double q = 1024.0;
  const double radius = std::ldexp(1.0, -radius_exp);
  for (int i = 0; i < k; ++i) {
    const double p = double((i - k / 2) * step);
    const double d = q * q + p * p;
    const double x = (q * q - p * p) / d * radius;
    const double y = (2.0 * p * q) / d * radius;
    auto point = mesh.points_buffer()[std::size_t(i + 1)];
    point[0] = Real(x + double(shear_x) / 16.0 * 4.0);
    point[1] = Real(y + double(shear_y) / 16.0 * 4.0);
    point[2] = Real(4);
  }
  mesh.faces_buffer().allocate(std::size_t(2 * k - 2));
  std::size_t face = 0;
  for (int i = 0; i < k; ++i) {
    mesh.faces_buffer()[face][0] = Index(0);
    mesh.faces_buffer()[face][1] = Index(1 + (i + 1) % k);
    mesh.faces_buffer()[face][2] = Index(1 + i);
    ++face;
  }
  for (int i = 1; i + 1 < k; ++i) {
    mesh.faces_buffer()[face][0] = Index(1);
    mesh.faces_buffer()[face][1] = Index(1 + i);
    mesh.faces_buffer()[face][2] = Index(2 + i);
    ++face;
  }
  return mesh;
}

/// The box [-8, 8]^2 x [-4, top]: with the needle it spans a bounding box
/// symmetric about the apex.
template <typename Index, typename Real>
auto boundary_law_box(double top) -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  const double xy[2] = {-8.0, 8.0};
  const double z[2] = {-4.0, top};
  mesh.points_buffer().allocate(8);
  for (int i = 0; i < 8; ++i) {
    auto point = mesh.points_buffer()[std::size_t(i)];
    point[0] = Real(xy[i & 1]);
    point[1] = Real(xy[(i >> 1) & 1]);
    point[2] = Real(z[(i >> 2) & 1]);
  }
  const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                           {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
  mesh.faces_buffer().allocate(12);
  for (int quad = 0; quad < 6; ++quad) {
    auto first = mesh.faces_buffer()[std::size_t(2 * quad)];
    first[0] = Index(quads[quad][0]);
    first[1] = Index(quads[quad][1]);
    first[2] = Index(quads[quad][2]);
    auto second = mesh.faces_buffer()[std::size_t(2 * quad + 1)];
    second[0] = Index(quads[quad][0]);
    second[1] = Index(quads[quad][2]);
    second[2] = Index(quads[quad][3]);
  }
  return mesh;
}

/// Per operand, per undirected edge, how often its emitted surface runs the
/// edge one way minus the other — the exposed triangles, and the source faces
/// no cut reached. A closed surface runs every edge once each way.
template <typename Graph, typename Mesh0, typename Mesh1>
auto boundary_law_unbalanced_edges(const Graph &graph, const Mesh0 &mesh0,
                                   const Mesh1 &mesh1) -> std::size_t {
  const auto &arrangement = graph.arrangement();
  const auto triangles = arrangement.global().exposed_tris();
  const auto tags = arrangement.triangle_tags();
  const auto key_of = [](const auto &vertex) {
    return std::int64_t(vertex.id) * 2 +
           (vertex.source == tf::intersect::graph::vertex_source::created);
  };
  std::vector<std::array<std::int64_t, 4>> runs;
  const auto run = [&runs](std::int64_t tag, std::int64_t a, std::int64_t b) {
    runs.push_back({tag, std::min(a, b), std::max(a, b), a < b ? 1 : -1});
  };
  for (std::size_t t = 0; t < std::size_t(triangles.size()); ++t)
    for (int c = 0; c < 3; ++c) {
      const auto a = key_of(triangles[t][c]);
      const auto b = key_of(triangles[t][(c + 1) % 3]);
      if (a != b)
        run(std::int64_t(tags[t]), a, b);
    }
  const auto uncut = [&](std::size_t tag, const auto &mesh) {
    const auto base = std::int64_t(arrangement.vertex_offsets()[tag]);
    const auto slot_base = std::size_t(arrangement.global().face_slot_base()[tag]);
    for (std::size_t f = 0; f < std::size_t(mesh.faces().size()); ++f) {
      const auto span = arrangement.slot_range(
          static_cast<std::decay_t<decltype(mesh.faces()[0][0])>>(slot_base + f));
      if (span[0] != span[1])
        continue;
      const auto face = mesh.faces()[f];
      for (int c = 0; c < 3; ++c)
        run(std::int64_t(tag), (base + std::int64_t(face[c])) * 2,
            (base + std::int64_t(face[(c + 1) % 3])) * 2);
    }
  };
  uncut(0, mesh0);
  uncut(1, mesh1);
  std::sort(runs.begin(), runs.end());
  std::size_t unbalanced = 0;
  for (std::size_t at = 0; at < runs.size();) {
    auto next = at;
    std::int64_t sum = 0;
    while (next < runs.size() && runs[next][0] == runs[at][0] &&
           runs[next][1] == runs[at][1] && runs[next][2] == runs[at][2])
      sum += runs[next++][3];
    unbalanced += sum != 0;
    at = next;
  }
  return unbalanced;
}

template <typename Graph> auto boundary_law_self_sided(const Graph &graph) {
  const auto &descriptor = graph.descriptor();
  std::size_t self_sided = 0;
  for (std::size_t c = 0; c < descriptor.tag_of_component.size(); ++c)
    self_sided +=
        descriptor.domain_of_side[2 * c] == descriptor.domain_of_side[2 * c + 1];
  return self_sided;
}

} // namespace

TEMPLATE_TEST_CASE("csg boundary law: a needle tip cut below the lattice",
                   "[csg][boundary_law][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  // the same needle on both lattices, cut a few dozen lattice units above the
  // apex, where its slivers are narrower than one
  const bool f32 = std::is_same<real_t, float>::value;
  auto needle = boundary_law_needle<index_t, real_t>(32, 32, f32 ? 3 : 4, 24, 5);
  auto box = boundary_law_box<index_t, real_t>(std::ldexp(80.0, f32 ? -29 : -60));
  auto op0 = tf::test::make_tagged_operand(needle);
  auto op1 = tf::test::make_tagged_operand(box);
  auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                              tf::test::no_sheets(), {});

  REQUIRE(graph.failed().size() == 0);
  const auto &census = graph.arrangement().arrangement().census();
  REQUIRE(census.crossings >= 1);
  REQUIRE(census.rounds >= 3);
  REQUIRE(boundary_law_unbalanced_edges(graph, needle, box) == 0);
  REQUIRE(boundary_law_self_sided(graph) == 0);
}

/// THE RING CERTIFICATE on the same needle, cut where its rings stand below
/// the lattice. One ring's piece welds several exact lines into one, so its
/// order comes off the pages' own carriers; another's carriers are emitted on
/// a piece they do not pass through. Either way the order the carriers state
/// is not the one the emitted surface stands in, and published it joins two
/// sides of one closed surface. The ring states no sector instead, and still
/// states that the components meeting at it touch.
TEMPLATE_TEST_CASE("csg ring certificate: a needle ring the emission does not "
                   "certify publishes no order",
                   "[csg][boundary_law][ring_certificate]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  struct needle_case {
    int k, step, radius_exp, shear_x, shear_y, cut_exp, cut;
  };
  const bool f32 = std::is_same<real_t, float>::value;
  const std::array<needle_case, 2> cases =
      f32 ? std::array<needle_case, 2>{{{64, 64, 4, 24, 5, 29, 80},
                                        {64, 32, 4, -16, 18, 29, 200}}}
          : std::array<needle_case, 2>{{{32, 16, 2, 12, -12, 60, 130},
                                        {32, 16, 4, -22, -14, 60, 200}}};
  for (const auto &c : cases) {
    auto needle = boundary_law_needle<index_t, real_t>(
        c.k, c.step, c.radius_exp, c.shear_x, c.shear_y);
    auto box = boundary_law_box<index_t, real_t>(
        std::ldexp(double(c.cut), -c.cut_exp));
    auto op0 = tf::test::make_tagged_operand(needle);
    auto op1 = tf::test::make_tagged_operand(box);
    auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                                tf::test::no_sheets(), {});
    REQUIRE(graph.failed().size() == 0);
    CHECK(graph.descriptor().fans.n_withheld >= 1);
    CHECK(boundary_law_unbalanced_edges(graph, needle, box) == 0);
    CHECK(boundary_law_self_sided(graph) == 0);
    // a withheld ring still states that its components touch: the needle
    // and the box it pierces are one bundle, and no cast repairs a split
    CHECK(graph.descriptor().n_bundles == 1);
    CHECK(graph.domain_nesting_merges().size() == 0);
  }
}
