/**
 * @file test_mesh_triangulation.cpp
 * @brief The mesh as a world of the resolving tier
 *
 * `tf::arrangement::mesh_triangulation` is the tier `tf::triangulated` answers
 * from, so there is no second triangulator to ask: every law here is read
 * against the FACE, one face at a time.
 *
 *   S   THE SPAN — a face that needs no resolution triangulates into n - 2
 *       triangles, and no carrier refuses.
 *   K   THE CORNER SET — those triangles name the face's own input vertex ids
 *       and no other: a mesh with no crossing face mints nothing.
 *   A   THE AREA — the emitted triangles sum, exactly on the lattice, to the
 *       face's own signed area in the face's own frame. This is what
 *       certifies the product on its own: same-winding triangles over the
 *       face's own corners summing exactly to its area can neither overlap
 *       nor leave a gap.
 *
 * The diagonal is deliberately NOT pinned: two triangulators of one simple
 * loop may cut it differently and both be right.
 *
 * The instrument is proven able to fail: a bow-tie face has no triangulation
 * over its own points, so the area must break and the product must stand on
 * an identity the input did not carry.
 *
 * What a face that DOES need resolution gets is the second half:
 *
 *   R   THE RESOLUTION — a crossing loop states its crossing, mints the one
 *       identity that names it, and holds a product over it.
 *   M   THE MINT LAW — an identity the wave mints while resolving stands where
 *       the face already stood. `elect_plane_identity_components` never elects
 *       an ORIGINAL vertex as a component's survivor: an existing created
 *       identity wins, else a fresh class mints, else — for a component of
 *       originals alone — `mint_original` appends a rootless class at that
 *       original's own coordinate and retires every original into it. So two
 *       corners at one lattice point become ONE minted identity there, and no
 *       product ever names two identities standing at one point.
 *   N   THE PROPAGATION — a split of a canonical group is a fact of the GROUP,
 *       so a neighbour holding the other instance of a split mesh edge
 *       re-triangulates with the new identity on its boundary and stays
 *       watertight. This is the whole reason the mesh is a world.
 *   E   THE EMPTINESS SURFACE — a face `failed()` names holds an empty span,
 *       and a face that bounds no area holds one too: emptiness is the answer
 *       either way, and the surface is what names whose it was.
 *   V   THE PROVENANCE — on an uncut face every corner sub is the corner's own
 *       ordinal on that face's polygon.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_mesh_fixture.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/mesh/mesh_triangulation.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient2d.hpp>
#include <trueform/exact/signed_area.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/topology/topo_id.hpp>
#include <trueform/topology/topo_type.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

using mesh_triangulation_index_t = tf::test::plane_index_t;

template <typename Int> struct mesh_triangulation_real_of;
template <> struct mesh_triangulation_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct mesh_triangulation_real_of<tf::exact::int64> {
  using type = double;
};

struct face_report_t {
  std::size_t faces = 0;
  std::size_t triangles = 0;
  std::size_t failed = 0;
  std::size_t created = 0;
  std::size_t foreign_corners = 0;
  std::size_t span_defects = 0;
  std::size_t corner_defects = 0;
  std::size_t area_defects = 0;
  std::size_t empty_span_defects = 0;
};

auto sorted_unique(std::vector<mesh_triangulation_index_t> ids)
    -> std::vector<mesh_triangulation_index_t> {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

template <typename T> auto magnitude(T value) -> T {
  return value < T(0) ? -value : value;
}

/// The lattice position of a flat identity: the input's own point below the
/// original extent, this build's mint past it.
template <typename Int, typename Product, typename Policy>
auto flat_point(const Product &product, const tf::polygons<Policy> &polygons,
                mesh_triangulation_index_t flat) -> tf::point<Int, 3> {
  const auto n_original = product.n_original_points();
  return flat < n_original
             ? product.converter().convert(
                   polygons.points()[std::size_t(flat)])
             : product.created_points()[std::size_t(flat - n_original)];
}

/// Every corner one face's span names.
template <typename Product>
auto face_corners(const Product &product, mesh_triangulation_index_t face)
    -> std::vector<mesh_triangulation_index_t> {
  const auto span = product.face_range(face);
  std::vector<mesh_triangulation_index_t> corners;
  for (mesh_triangulation_index_t t = span[0]; t < span[1]; ++t)
    for (const auto corner : product.triangles()[std::size_t(t)])
      corners.push_back(corner);
  return corners;
}

/// One face's emitted area against its own, over every identity the product
/// names — the split that lands ON an edge leaves the area it bounds alone.
template <typename Int, typename Product, typename Policy>
auto area_conserved(const Product &product,
                    const tf::polygons<Policy> &polygons,
                    mesh_triangulation_index_t face) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto frame = product.world().frame(face);
  const auto at = [&](mesh_triangulation_index_t flat) -> tf::exact::pt2<Int> {
    const auto q = flat_point<Int>(product, polygons, flat);
    return {q[frame.ax0], q[frame.ax1]};
  };
  const auto span = product.face_range(face);
  T2 area = 0;
  for (mesh_triangulation_index_t t = span[0]; t < span[1]; ++t) {
    const auto &triangle = product.triangles()[std::size_t(t)];
    area += tf::exact::orient2d(at(triangle[0]), at(triangle[1]),
                                at(triangle[2]));
  }
  return magnitude(area) ==
         magnitude(tf::exact::signed_area_2x(
             polygons.faces()[std::size_t(face)], [&](auto corner) {
               return at(mesh_triangulation_index_t(corner));
             }));
}

/// THE MINT LAW, half one: an election that joins identities cannot invent a
/// position, so every minted point stands on a corner the input already had.
template <typename Product, typename Policy>
auto minted_off_input(const Product &product,
                      const tf::polygons<Policy> &polygons) -> std::size_t {
  std::size_t defects = 0;
  for (const auto &minted : product.created_points()) {
    bool stood = false;
    for (mesh_triangulation_index_t id = 0; id < product.n_original_points();
         ++id)
      if (product.converter().convert(polygons.points()[std::size_t(id)]) ==
          minted)
        stood = true;
    if (!stood)
      ++defects;
  }
  return defects;
}

/// THE MINT LAW, half two: no product names two identities standing at one
/// lattice point — the election retires all but one of them.
template <typename Int, typename Product, typename Policy>
auto coincident_identity_defects(const Product &product,
                                 const tf::polygons<Policy> &polygons)
    -> std::size_t {
  std::vector<mesh_triangulation_index_t> named;
  for (mesh_triangulation_index_t face = 0; face < product.n_faces(); ++face) {
    const auto corners = face_corners(product, face);
    named.insert(named.end(), corners.begin(), corners.end());
  }
  std::vector<tf::point<Int, 3>> positions;
  for (const auto flat : sorted_unique(std::move(named)))
    positions.push_back(flat_point<Int>(product, polygons, flat));
  std::sort(positions.begin(), positions.end());
  std::size_t defects = 0;
  for (std::size_t at = 1; at < positions.size(); ++at)
    if (positions[at - 1] == positions[at])
      ++defects;
  return defects;
}

/// THE EMPTINESS SURFACE: a face `failed()` names holds an empty span.
template <typename Product>
auto emptiness_defects(const Product &product) -> std::size_t {
  std::size_t defects = 0;
  for (const auto face : product.failed()) {
    const auto span = product.face_range(face);
    if (span[1] != span[0])
      ++defects;
  }
  return defects;
}

/// On an uncut face every corner sub is that corner's own ordinal.
template <typename Product, typename Policy>
auto sub_defects(const Product &product, const tf::polygons<Policy> &polygons)
    -> std::size_t {
  std::size_t defects = 0;
  const auto subs = product.corner_subs();
  for (mesh_triangulation_index_t face = 0; face < product.n_faces(); ++face) {
    const auto corners = polygons.faces()[std::size_t(face)];
    const auto span = product.face_range(face);
    for (mesh_triangulation_index_t t = span[0]; t < span[1]; ++t)
      for (std::size_t slot = 0; slot < 3; ++slot) {
        const auto corner = product.triangles()[std::size_t(t)][slot];
        const auto sub = subs[std::size_t(t)][slot];
        std::size_t ordinal = corners.size();
        for (std::size_t position = 0; position < corners.size(); ++position)
          if (mesh_triangulation_index_t(corners[position]) == corner) {
            ordinal = position;
            break;
          }
        if (ordinal == corners.size() || sub.label != tf::topo_type::vertex ||
            std::size_t(sub.id) != ordinal)
          ++defects;
      }
  }
  return defects;
}

template <typename Int, typename Policy>
auto read_faces(const tf::polygons<Policy> &polygons) -> face_report_t {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto product = tf::arrangement::make_mesh_triangulation<Int>(polygons);
  const auto n_original = product.n_original_points();
  tf::buffer<tf::point<Int, 3>> points;
  points.allocate(std::size_t(n_original));
  for (mesh_triangulation_index_t id = 0; id < n_original; ++id)
    points[std::size_t(id)] =
        product.converter().convert(polygons.points()[std::size_t(id)]);

  face_report_t report;
  report.faces = std::size_t(product.n_faces());
  report.triangles = product.triangles().size();
  report.failed = product.failed().size();
  report.created = product.created_points().size();
  report.empty_span_defects = emptiness_defects(product);
  for (mesh_triangulation_index_t face = 0; face < product.n_faces(); ++face) {
    const auto corners = polygons.faces()[std::size_t(face)];
    const auto span = product.face_range(face);
    const auto count = std::size_t(span[1] - span[0]);
    const auto frame = product.world().frame(face);
    const auto at =
        [&](mesh_triangulation_index_t flat) -> tf::exact::pt2<Int> {
      const auto q = points[std::size_t(flat)];
      return {q[frame.ax0], q[frame.ax1]};
    };

    std::vector<mesh_triangulation_index_t> emitted;
    T2 area = 0;
    for (mesh_triangulation_index_t t = span[0]; t < span[1]; ++t) {
      const auto &triangle = product.triangles()[std::size_t(t)];
      bool own = true;
      for (const auto corner : triangle) {
        if (corner >= n_original) {
          ++report.foreign_corners;
          own = false;
          continue;
        }
        emitted.push_back(corner);
      }
      if (own)
        area += tf::exact::orient2d(at(triangle[0]), at(triangle[1]),
                                    at(triangle[2]));
    }
    const auto face_area = tf::exact::signed_area_2x(corners, [&](auto corner) {
      return at(mesh_triangulation_index_t(corner));
    });
    if (magnitude(area) != magnitude(face_area))
      ++report.area_defects;

    if (count + 2 != corners.size())
      ++report.span_defects;
    std::vector<mesh_triangulation_index_t> own_ids;
    for (const auto corner : corners)
      own_ids.push_back(mesh_triangulation_index_t(corner));
    if (sorted_unique(std::move(emitted)) != sorted_unique(std::move(own_ids)))
      ++report.corner_defects;
  }
  return report;
}

/// What this tier owes for a face that needs no resolution, every one read
/// against the face itself.
auto expect_own_laws(const face_report_t &report) -> void {
  CHECK(report.failed == 0);
  CHECK(report.created == 0);
  CHECK(report.foreign_corners == 0);
  CHECK(report.span_defects == 0);
  CHECK(report.corner_defects == 0);
  CHECK(report.area_defects == 0);
  CHECK(report.empty_span_defects == 0);
}

} // namespace

TEMPLATE_TEST_CASE("mesh triangulation: a convex face is its own world",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto quad = tf::test::make_mesh_convex_quad<Real>();
  const auto report = read_faces<Int>(quad.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: a tilted face is its own world",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto tilted = tf::test::make_mesh_tilted_quad<Real>();
  const auto report = read_faces<Int>(tilted.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: a reflex corner is not a refusal",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto shape = tf::test::make_mesh_non_convex_l<Real>();
  const auto report = read_faces<Int>(shape.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 4);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: a collinear leading run is not a line",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto shape = tf::test::make_mesh_collinear_run<Real>();
  const auto report = read_faces<Int>(shape.polygons());
  CHECK(report.faces == 1);
  CHECK(report.triangles == 4);
  // the frame producer scans for its supporting triple, so the carrier bounds
  // area and the face's whole triangulation stands over its own corners
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: a shared edge stays one edge",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_shared_edge<Real>();
  const auto report = read_faces<Int>(mesh.polygons());
  CHECK(report.faces == 2);
  CHECK(report.triangles == 2);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: a closed box is every face at once",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_box<Real>();
  const auto report = read_faces<Int>(mesh.polygons());
  CHECK(report.faces == 12);
  CHECK(report.triangles == 12);
  expect_own_laws(report);
}

TEMPLATE_TEST_CASE("mesh triangulation: the instrument can fail",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto bow_tie = tf::test::make_mesh_bow_tie<Real>();
  const auto report = read_faces<Int>(bow_tie.polygons());
  // a crossing loop has no triangulation on its own boundary: whatever this
  // tier answers, it is not the loop's own area over the loop's own points
  CHECK(report.area_defects == 1);
  CHECK(report.foreign_corners > 0);
  CHECK(report.created + report.failed > 0);
}

TEMPLATE_TEST_CASE("mesh triangulation: a clean build states one round",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_box<Real>();
  const auto product =
      tf::arrangement::make_mesh_triangulation<Int>(mesh.polygons());
  const auto &census = product.census();
  CHECK(census.planes == 12);
  CHECK(census.rounds == 1);
  CHECK(census.refusals == 0);
  CHECK(census.created == 0);
  CHECK(census.failed_planes == 0);
  CHECK(census.triangles == 12);
}

TEMPLATE_TEST_CASE("mesh triangulation: a crossing loop resolves into a product",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto bow_tie = tf::test::make_mesh_bow_tie<Real>();
  const auto product =
      tf::arrangement::make_mesh_triangulation<Int>(bow_tie.polygons());
  CHECK(product.failed().size() == 0);
  CHECK(emptiness_defects(product) == 0);
  REQUIRE(product.created_points().size() == 1);
  CHECK(product.triangles().size() >= 2);

  // both lobes stand on the one identity the crossing minted
  const auto corners = face_corners(product, 0);
  const auto minted = std::count_if(
      corners.begin(), corners.end(), [&](mesh_triangulation_index_t corner) {
        return corner >= product.n_original_points();
      });
  CHECK(minted >= 2);

  const auto &census = product.census();
  CHECK(census.refusals >= 1);
  CHECK(census.rounds > 1);
  CHECK(census.created == 1);
}

TEMPLATE_TEST_CASE("mesh triangulation: a split reaches the shared edge's "
                   "other face",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_split_neighbour<Real>();
  const auto polygons = mesh.polygons();
  const auto product = tf::arrangement::make_mesh_triangulation<Int>(polygons);
  CHECK(product.failed().size() == 0);
  REQUIRE(product.created_points().size() == 1);

  // face 1 states nothing of its own, and re-triangulates anyway: the split
  // is a fact of the canonical group, and face 1 holds the other instance
  const auto span = product.face_range(1);
  CHECK(span[1] - span[0] == 2);
  const std::vector<mesh_triangulation_index_t> expected{
      mesh_triangulation_index_t(0), mesh_triangulation_index_t(1),
      mesh_triangulation_index_t(4), product.n_original_points()};
  CHECK(sorted_unique(face_corners(product, 1)) == expected);
  // the identity landed ON the shared edge, so the neighbour bounds the same
  // area it always did
  CHECK(area_conserved<Int>(product, polygons, mesh_triangulation_index_t(1)));
}

TEMPLATE_TEST_CASE("mesh triangulation: a doubled corner becomes one identity",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_doubled_vertex<Real>();
  const auto polygons = mesh.polygons();
  const auto product = tf::arrangement::make_mesh_triangulation<Int>(polygons);
  CHECK(product.failed().size() == 0);
  CHECK(product.triangles().size() >= 2);
  CHECK(area_conserved<Int>(product, polygons, mesh_triangulation_index_t(0)));

  // TWO ORIGINAL IDENTITIES AT ONE LATTICE POINT ARE ONE IDENTITY IN THE
  // PRODUCT. The election ranks an original last and never elects one as a
  // component's survivor, so a component of originals alone mints a rootless
  // class at their shared coordinate and retires both into it.
  REQUIRE(product.created_points().size() == 1);
  CHECK(minted_off_input(product, polygons) == 0);
  CHECK(coincident_identity_defects<Int>(product, polygons) == 0);
  // four corners still bound the area, one of them the minted survivor
  const auto corners = sorted_unique(face_corners(product, 0));
  CHECK(corners.size() == 4);
}

TEMPLATE_TEST_CASE("mesh triangulation: a face that bounds no area is empty",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto mesh = tf::test::make_mesh_degenerate_face<Real>();
  const auto polygons = mesh.polygons();
  const auto product = tf::arrangement::make_mesh_triangulation<Int>(polygons);
  CHECK(product.triangles().size() == 0);
  const auto span = product.face_range(0);
  CHECK(span[1] - span[0] == 0);
  CHECK(emptiness_defects(product) == 0);
  // its three sides overlap on one line, so resolving them states a landing on
  // the corner between them and the election mints a class there. Nothing
  // stands on it — the carrier bounds no area — and it invents no position.
  CHECK(minted_off_input(product, polygons) == 0);
}

TEMPLATE_TEST_CASE("mesh triangulation: a corner sub is the corner's ordinal",
                   "[arrangement][planes][mesh_triangulation]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename mesh_triangulation_real_of<Int>::type;
  auto quad = tf::test::make_mesh_convex_quad<Real>();
  auto shape = tf::test::make_mesh_non_convex_l<Real>();
  auto box = tf::test::make_mesh_box<Real>();
  CHECK(sub_defects(tf::arrangement::make_mesh_triangulation<Int>(
                        quad.polygons()),
                    quad.polygons()) == 0);
  CHECK(sub_defects(tf::arrangement::make_mesh_triangulation<Int>(
                        shape.polygons()),
                    shape.polygons()) == 0);
  CHECK(sub_defects(tf::arrangement::make_mesh_triangulation<Int>(
                        box.polygons()),
                    box.polygons()) == 0);
}
