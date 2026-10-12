/**
 * @file test_door_placement.cpp
 * @brief The door: where an input vertex stands once a tolerance is given.
 *
 * THE BAND IS A SPARSE SET OF PULLS; everywhere it does not act it is
 * tolerance zero. A vertex moves only when a contact or a pool pulled it,
 * and then only onto a lattice point that lies EXACTLY on every plane it
 * owes, within the band of where it stood. Both halves are asserted here
 * rather than measured: the exact solves land on their planes, and every
 * placement, however adversarial the planes, either stands within the band
 * on all of them or leaves the vertex where it was.
 *
 * Every lattice the library resolves is driven: a float input names int32
 * and a double input int64. The generic-direction scene separates them — an
 * axis plane is solved on any rung, and only a face whose exact normal
 * needs the product rung reports the width.
 *
 * The POOLED scene is two walls of two triangles each, standing at different
 * heights with a further corner pushed off, so the operands arrive as four
 * exact planes and no plane. Afterwards they are one plane and one canonical
 * NAME, standing on original vertices that did not move — and at a band of
 * zero the whole tier is inert and the reader is the plain converter.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/frame_of.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/small_vector.hpp>
#include <trueform/core/transformed.hpp>
#include <trueform/core/vector.hpp>
#include <trueform/exact/canonical_plane.hpp>
#include <trueform/exact/door/admits_placement.hpp>
#include <trueform/exact/door/make_support_plane.hpp>
#include <trueform/exact/door/place_at_exact_meet.hpp>
#include <trueform/exact/door/place_on_exact_line.hpp>
#include <trueform/exact/door/place_on_plane.hpp>
#include <trueform/exact/door/place_vertex.hpp>
#include <trueform/exact/door/plane_frame.hpp>
#include <trueform/exact/door/point_on_face.hpp>
#include <trueform/exact/door/round_div.hpp>
#include <trueform/exact/door/round_to_wide.hpp>
#include <trueform/exact/door/segments_cross.hpp>
#include <trueform/exact/door/segments_touch.hpp>
#include <trueform/exact/door/wide_dot.hpp>
#include <trueform/exact/input_lattice.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/plane_support.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/exact/vertex_converter.hpp>
#include <trueform/geometry/make_box_mesh.hpp>

#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using index_t = int;
using real_t = float;
using int_t = tf::exact::resolve_int_type<tf::none_t, real_t>;
using wide_t = tf::exact::meta<int_t>::T1;
using wider_t = tf::exact::meta<int_t>::T2;
using plane_t = tf::exact::canonical_plane<int_t>;

/// The operands, the lattice view over their union, and the reader every
/// consumer of the pipeline programs against.
template <typename RealType> struct scene_t {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, RealType>;
  using mesh_t = tf::polygons_buffer<index_t, RealType, 3, 3>;
  using operand_t = tf::test::tagged_operand<index_t, RealType>;
  using tables_t =
      tf::exact::door::placement_tables<index_t, lattice_int_t, RealType>;

  std::vector<std::unique_ptr<operand_t>> operands;
  tf::exact::input_lattice<index_t, RealType, lattice_int_t> lattice;

  auto add(mesh_t mesh) -> void {
    operands.push_back(std::make_unique<operand_t>(std::move(mesh)));
  }

  auto apply_to_form() const {
    const auto *scene = this;
    return [scene](index_t tag, auto &&f) {
      f(scene->operands[std::size_t(tag)]->form());
    };
  }

  auto build(double tolerance) -> void {
    using form_t = decltype(std::declval<operand_t &>().form());
    std::vector<form_t> forms;
    for (auto &operand : operands)
      forms.push_back(operand->form());
    lattice.build(
        tf::exact::make_vertex_converter<lattice_int_t, RealType>(
            tf::make_range(forms.data(), forms.data() + forms.size())),
        apply_to_form(), index_t(operands.size()), tolerance);
  }

  auto placed(index_t tag, index_t id) const -> tf::point<lattice_int_t, 3> {
    return lattice.reader(apply_to_form())(int(tag), id);
  }

  /// Every vertex where the pipeline reads it, in flat order.
  auto placed_all() const -> std::vector<tf::point<lattice_int_t, 3>> {
    std::vector<tf::point<lattice_int_t, 3>> out;
    for (index_t tag = 0; tag < index_t(operands.size()); ++tag)
      for (index_t id = 0;
           id < index_t(operands[std::size_t(tag)]->mesh.points().size()); ++id)
        out.push_back(placed(tag, id));
    return out;
  }
};

/// A quad in the plane `z = height`, as two triangles.
template <typename RealType>
auto wall_mesh(RealType height)
    -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  tf::polygons_buffer<index_t, RealType, 3, 3> mesh;
  mesh.points_buffer().emplace_back(RealType(-0.5), RealType(-0.5), height);
  mesh.points_buffer().emplace_back(RealType(0.5), RealType(-0.5), height);
  mesh.points_buffer().emplace_back(RealType(0.5), RealType(0.5), height);
  mesh.points_buffer().emplace_back(RealType(-0.5), RealType(0.5), height);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  return mesh;
}

/// Two quads a hair apart, in ONE form — the coincidence no intersection
/// record can witness, because both sheets may be uncut.
template <typename RealType>
auto doubled_wall_mesh(RealType gap)
    -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  auto lower = wall_mesh(RealType(0));
  auto upper = wall_mesh(gap);
  tf::polygons_buffer<index_t, RealType, 3, 3> mesh;
  mesh.points_buffer().allocate(8);
  for (std::size_t i = 0; i < 4; ++i) {
    mesh.points_buffer()[i] = lower.points_buffer()[i];
    mesh.points_buffer()[i + 4] = upper.points_buffer()[i];
  }
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  mesh.faces_buffer().emplace_back(4, 5, 6);
  mesh.faces_buffer().emplace_back(4, 6, 7);
  return mesh;
}

template <typename RealType>
auto translated_box(RealType dx, RealType dy, RealType dz)
    -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  auto mesh = tf::make_box_mesh<index_t>(RealType(1), RealType(1), RealType(1));
  for (std::size_t i = 0; i < mesh.points_buffer().size(); ++i) {
    mesh.points_buffer()[i][0] += dx;
    mesh.points_buffer()[i][1] += dy;
    mesh.points_buffer()[i][2] += dz;
  }
  return mesh;
}

/// A unit cube turned by two rational rotations: every face states a GENERIC
/// direction, whose offset needs the whole rung above the lattice — where an
/// axis-aligned one asks for a single lattice unit and hides the width.
template <typename RealType>
auto rotated_box() -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  auto mesh = tf::make_box_mesh<index_t>(RealType(1), RealType(1), RealType(1));
  for (std::size_t i = 0; i < mesh.points_buffer().size(); ++i) {
    auto point = mesh.points_buffer()[i];
    const auto x = RealType(0.6) * point[0] - RealType(0.8) * point[1];
    const auto y = RealType(0.8) * point[0] + RealType(0.6) * point[1];
    const auto z = point[2];
    point[0] = (RealType(5) * x - RealType(12) * z) / RealType(13);
    point[1] = y;
    point[2] = (RealType(12) * x + RealType(5) * z) / RealType(13);
  }
  return mesh;
}

/// A small quad on a generic plane, off the centre of the scene it sits in,
/// so its offset is as large as the scene allows. Its vertices carry ONE
/// name, which is the smooth case rank 1 answers.
template <typename RealType>
auto tilted_wall_mesh() -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  const RealType centre[3]{RealType(0.3), RealType(-0.2), RealType(0.25)};
  const RealType along0[3]{RealType(0.08), RealType(-0.04), RealType(0)};
  const RealType along1[3]{RealType(0.024), RealType(0.048), RealType(-0.04)};
  tf::polygons_buffer<index_t, RealType, 3, 3> mesh;
  for (const auto &corner :
       {std::array<RealType, 2>{RealType(1), RealType(1)},
        std::array<RealType, 2>{RealType(-1), RealType(1)},
        std::array<RealType, 2>{RealType(-1), RealType(-1)},
        std::array<RealType, 2>{RealType(1), RealType(-1)}})
    mesh.points_buffer().emplace_back(
        centre[0] + corner[0] * along0[0] + corner[1] * along1[0],
        centre[1] + corner[0] * along0[1] + corner[1] * along1[1],
        centre[2] + corner[0] * along0[2] + corner[1] * along1[2]);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  return mesh;
}

/// A quad of two triangles spanning `[x0, x1] x [0, 1]`, standing at `lift`
/// with its far corner pushed a further `tilt` off it.
///
/// Its two triangles state DIFFERENT exact planes whenever `tilt` is not
/// zero, so a wall of this shape is not planar to begin with — which is the
/// premise the pooled fixture below rests on.
template <typename RealType>
auto pooled_wall_mesh(RealType x0, RealType x1, RealType lift, RealType tilt)
    -> tf::polygons_buffer<index_t, RealType, 3, 3> {
  tf::polygons_buffer<index_t, RealType, 3, 3> mesh;
  mesh.points_buffer().emplace_back(x0, RealType(0), lift);
  mesh.points_buffer().emplace_back(x1, RealType(0), lift);
  mesh.points_buffer().emplace_back(x1, RealType(1), lift + tilt);
  mesh.points_buffer().emplace_back(x0, RealType(1), lift);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  return mesh;
}

/// The exact plane four lattice points stand on, or a support short of three
/// when they state no plane at all.
template <typename Int, typename Placed>
auto pooled_plane_of(const Placed &placed, std::size_t from)
    -> tf::exact::plane_support<Int> {
  tf::exact::plane_support<Int> support;
  for (std::size_t k = from; k < from + 4; ++k)
    support.offer(placed[k]);
  return support;
}

/// The exact planes of every face of a form on the lattice its converter
/// states, and the corners as lattice points.
template <typename RealType, typename Scene>
auto lattice_faces(const Scene &scene, index_t tag)
    -> std::pair<std::vector<tf::point<
                     tf::exact::resolve_int_type<tf::none_t, RealType>, 3>>,
                 std::vector<tf::exact::canonical_plane<
                     tf::exact::resolve_int_type<tf::none_t, RealType>>>> {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, RealType>;
  std::vector<tf::point<lattice_int_t, 3>> corners;
  std::vector<tf::exact::canonical_plane<lattice_int_t>> planes;
  const auto form = scene.operands[std::size_t(tag)]->form();
  const auto frame = tf::frame_of(form);
  for (const auto &point : form.points())
    corners.push_back(
        scene.lattice.converter().convert(tf::transformed(point, frame)));
  for (const auto &face : form.faces()) {
    std::array<tf::point<lattice_int_t, 3>, 3> at{
        corners[std::size_t(face[0])], corners[std::size_t(face[1])],
        corners[std::size_t(face[2])]};
    tf::exact::canonical_plane<lattice_int_t> plane{};
    tf::exact::door::make_support_plane<lattice_int_t>(at, plane);
    planes.push_back(plane);
  }
  return {corners, planes};
}

auto axis_plane(int axis, wider_t offset) -> plane_t {
  plane_t plane{};
  plane[std::size_t(axis)] = wider_t(1);
  plane[3] = offset;
  return plane;
}

/// A deterministic spread of adversarial planes: axis planes pushed far off
/// the vertex, near-parallel pairs, and directions at the lattice's own
/// magnitude, where every product the solvers take is at its widest.
auto adversarial_planes(std::uint32_t seed) -> std::vector<plane_t> {
  const auto next = [&seed]() -> std::uint32_t {
    seed = seed * 1664525u + 1013904223u;
    return seed;
  };
  const auto span = [&next](wider_t magnitude) -> wider_t {
    return wider_t(std::int64_t(next() % 2000001u) - 1000000) * magnitude;
  };
  const auto plane_of = [](wider_t nx, wider_t ny, wider_t nz, wider_t offset) {
    return tf::exact::canonicalize_plane<int_t>(plane_t{nx, ny, nz, offset});
  };
  std::vector<plane_t> planes;
  // planes a few units off the vertex the solves reach, then the far ones
  planes.push_back(plane_of(1, 0, 0, wider_t(1000 + int(next() % 7) - 3)));
  planes.push_back(plane_of(0, 0, 1, wider_t(3000 + int(next() % 7) - 3)));
  planes.push_back(plane_of(1, 0, 0, span(1000)));
  planes.push_back(plane_of(0, 1, 0, span(1000)));
  planes.push_back(plane_of(0, 0, 1, span(1000)));
  planes.push_back(plane_of(1000000, 1, 0, span(1000000)));
  planes.push_back(plane_of(1000000, 2, 0, span(1000000)));
  planes.push_back(plane_of(wider_t(next() % 100000u) + 1,
                            wider_t(next() % 100000u) + 1,
                            wider_t(next() % 100000u) + 1, span(100000)));
  planes.push_back(
      plane_of(2147483647, 2147483646, 2147483645, span(1000000000)));
  return planes;
}

} // namespace

TEST_CASE("door: the rounding divide answers every numerator its type holds",
          "[exact][door]") {
  using wider_t = tf::exact::meta<int_t>::T2;
  const wider_t top = wider_t(1)
                      << unsigned(tf::exact::meta<int_t>::t2_bits - 1);
  const wider_t largest = (top - wider_t(1)) + top;
  const wider_t half = top >> 1u;

  // A numerator past half the type's reach: the tie is decided on the
  // remainder, so the answer is the rounded quotient and not a wrapped one.
  REQUIRE(tf::exact::door::round_div(top, wider_t(2)) == half);
  REQUIRE(tf::exact::door::round_div(top + wider_t(1), wider_t(2)) ==
          half + wider_t(1));
  REQUIRE(tf::exact::door::round_div(-(top + wider_t(1)), wider_t(2)) ==
          -(half + wider_t(1)));
  REQUIRE(tf::exact::door::round_div(top, wider_t(-2)) == -half);
  REQUIRE(tf::exact::door::round_div(largest, wider_t(8)) == (half >> 1u));
  REQUIRE(tf::exact::door::round_div(largest, largest) == wider_t(1));

  // And the same answer as the doubled-numerator form states wherever that
  // one fits, over every sign of both operands.
  for (int num = -40; num <= 40; ++num)
    for (int den = -7; den <= 7; ++den) {
      if (den == 0)
        continue;
      const long long n = den < 0 ? -num : num;
      const long long d = den < 0 ? -den : den;
      const long long expected =
          n >= 0 ? (2 * n + d) / (2 * d) : -((-2 * n + d) / (2 * d));
      REQUIRE(tf::exact::door::round_div(wider_t(num), wider_t(den)) ==
              wider_t(expected));
    }
}

TEST_CASE("door: a narrow one-plane landing lies exactly on its plane",
          "[exact][door]") {
  const std::array<std::array<wide_t, 3>, 5> directions{
      {{{1, 0, 0}}, {{0, 0, 1}}, {{3, 4, 12}}, {{1, 1, 1}}, {{7, -5, 11}}}};
  for (const auto &normal : directions)
    for (wide_t residual : {wide_t(0), wide_t(1), wide_t(-1), wide_t(97),
                            wide_t(-12345), wide_t(1000003)}) {
      std::array<wide_t, 3> step{};
      REQUIRE(tf::exact::door::place_on_plane<int_t>(
          tf::exact::door::make_plane_frame<int_t>(normal), residual, step));
      REQUIRE(tf::exact::door::wide_dot<int_t>(normal, step) ==
              wider_t(residual));
    }
}

TEST_CASE("door: the narrow solve refuses the step it cannot state",
          "[exact][door]") {
  const std::array<wide_t, 3> normal{1073741827, 2147483647, 999999937};
  const wide_t residual = 1000000000000000;

  // The premise: the Bezout vector of a direction at the lattice's own
  // magnitude, times a residual this large, leaves the rung the narrow
  // solve is stated on — so it refuses rather than wraps, and the
  // certificate's wide lane carries the direction instead.
  const auto frame = tf::exact::door::make_plane_frame<int_t>(normal);
  const auto bound = wider_t(tf::exact::door::wide_placement_bound<int_t>());
  const auto onto = wider_t(residual) * wider_t(frame.s[0]);
  REQUIRE((onto > bound || onto < -bound));

  std::array<wide_t, 3> step{};
  REQUIRE_FALSE(tf::exact::door::place_on_plane<int_t>(frame, residual, step));
}

TEMPLATE_TEST_CASE("door: the exact meet of a generic corner is the corner, "
                   "on every lattice",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(rotated_box<TestType>());
  scene.build(0.0);
  const auto [corners, planes] = lattice_faces<TestType>(scene, 0);
  const auto form = scene.operands[0]->form();
  int meets = 0, lines = 0;
  for (index_t v = 0; v < index_t(corners.size()); ++v) {
    std::vector<tf::exact::canonical_plane<lattice_int_t>> incident;
    for (std::size_t f = 0; f < planes.size(); ++f)
      for (const auto corner : form.faces()[index_t(f)])
        if (corner == v)
          incident.push_back(planes[f]);
    // THE MEET: every incident plane passes through the corner, so any
    // independent triple of them meets exactly there, on the product rung
    for (std::size_t a = 0; a < incident.size(); ++a)
      for (std::size_t b = a + 1; b < incident.size(); ++b)
        for (std::size_t c = b + 1; c < incident.size(); ++c) {
          tf::point<lattice_int_t, 3> at{};
          if (!tf::exact::door::place_at_exact_meet<lattice_int_t>(
                  incident[a], incident[b], incident[c], at))
            continue;
          REQUIRE(at == corners[std::size_t(v)]);
          ++meets;
        }
    // THE LINE: from a point off the corner, two independent incident
    // planes give a lattice point ON both, no farther than the corner is
    const tf::point<lattice_int_t, 3> from{
        lattice_int_t(corners[std::size_t(v)][0] + 3),
        lattice_int_t(corners[std::size_t(v)][1] - 2),
        lattice_int_t(corners[std::size_t(v)][2] + 1)};
    for (std::size_t a = 0; a < incident.size(); ++a)
      for (std::size_t b = a + 1; b < incident.size(); ++b) {
        if (incident[a][0] == incident[b][0] &&
            incident[a][1] == incident[b][1] &&
            incident[a][2] == incident[b][2])
          continue;
        tf::point<lattice_int_t, 3> at{};
        REQUIRE(tf::exact::door::place_on_exact_line<lattice_int_t>(
            incident[a], incident[b], from, at));
        REQUIRE(tf::exact::orient3d_plane_sign<lattice_int_t>(incident[a],
                                                              at) == 0);
        REQUIRE(tf::exact::orient3d_plane_sign<lattice_int_t>(incident[b],
                                                              at) == 0);
        const auto d_at = (at - from).length2();
        const auto d_corner = (corners[std::size_t(v)] - from).length2();
        REQUIRE(d_at <= d_corner);
        ++lines;
      }
  }
  REQUIRE(meets > 0);
  REQUIRE(lines > 0);
}

TEST_CASE("door: the motion bound is a theorem, not a measurement",
          "[exact][door]") {
  const tf::point<int_t, 3> original{1000, -2000, 3000};
  int moved = 0;
  for (std::uint32_t seed = 1; seed <= 64; ++seed) {
    const auto all = adversarial_planes(seed);
    for (std::size_t owed = 1; owed <= 3; ++owed) {
      std::vector<plane_t> committed(all.begin(), all.begin() + owed);
      std::vector<plane_t> helpers(all.begin() + owed, all.end());
      std::sort(committed.begin(), committed.end());
      committed.erase(std::unique(committed.begin(), committed.end()),
                      committed.end());
      for (int_t tolerance :
           {int_t(1), int_t(17), int_t(4096), int_t(1000003)}) {
        tf::exact::door::plane_landing<int_t> landing;
        const auto placement = tf::exact::door::place_vertex(
            original, committed, helpers, tolerance, landing);
        const auto at = placement.point;
        if (!placement.placed)
          REQUIRE(at == original);
        if (at == original)
          continue;
        ++moved;
        REQUIRE(tf::exact::door::admits_placement(
            original,
            std::array<wide_t, 3>{wide_t(at[0]), wide_t(at[1]), wide_t(at[2])},
            tolerance));
        for (const auto &plane : committed)
          REQUIRE(tf::exact::orient3d_plane_sign<int_t>(plane, at) == 0);
      }
    }
  }
  // the spread reaches placements that move, so the bound is proven on
  // answers and not only on refusals
  REQUIRE(moved > 0);
}

TEST_CASE("door: a placement is a pure function of the planes and the band",
          "[exact][door]") {
  std::vector<plane_t> committed{axis_plane(0, 4000), axis_plane(1, -8000),
                                 axis_plane(2, 12000)};
  std::sort(committed.begin(), committed.end());
  const std::vector<plane_t> helpers;
  const int_t tolerance = 1000;
  tf::exact::door::plane_landing<int_t> landing;
  const auto first =
      tf::exact::door::place_vertex(tf::point<int_t, 3>{4300, -8200, 11700},
                                    committed, helpers, tolerance, landing)
          .point;
  const auto second =
      tf::exact::door::place_vertex(tf::point<int_t, 3>{3800, -7900, 12400},
                                    committed, helpers, tolerance, landing)
          .point;
  REQUIRE(first == second);
  REQUIRE(first == (tf::point<int_t, 3>{4000, -8000, 12000}));
}

TEST_CASE("door: a vertex already on every plane it owes does not move",
          "[exact][door]") {
  std::vector<plane_t> committed{axis_plane(0, 4000), axis_plane(2, 12000)};
  std::sort(committed.begin(), committed.end());
  const std::vector<plane_t> helpers{axis_plane(1, -7000)};
  tf::exact::door::plane_landing<int_t> landing;
  const tf::point<int_t, 3> on{4000, -8000, 12000};
  const auto stays = tf::exact::door::place_vertex(on, committed, helpers,
                                                   int_t(5000), landing);
  REQUIRE(stays.placed);
  REQUIRE(stays.point == on);
  // and one with no lattice point on its planes within the band keeps its
  // input point, and says it refused
  const tf::point<int_t, 3> distant{4000, -8000, 30000};
  const auto refused = tf::exact::door::place_vertex(
      distant, committed, helpers, int_t(5000), landing);
  REQUIRE_FALSE(refused.placed);
  REQUIRE(refused.point == distant);
}

TEMPLATE_TEST_CASE("door: a tolerance of zero is the identity", "[exact][door]",
                   float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(tf::make_box_mesh<index_t>(TestType(1), TestType(1), TestType(1)));
  scene.build(0.0);

  REQUIRE(scene.lattice.tolerance_int() == lattice_int_t(0));
  REQUIRE(scene.lattice.moved_points().size() == 0);
  REQUIRE(scene.lattice.moved_tickets().size() == 0);

  const auto reader = scene.lattice.reader(scene.apply_to_form());
  const auto &converter = scene.lattice.converter();
  const auto form = scene.operands[0]->form();
  const auto frame = tf::frame_of(form);
  const auto points = form.points();
  for (index_t id = 0; id < index_t(points.size()); ++id) {
    REQUIRE(reader(0, id) ==
            converter.convert(tf::transformed(points[id], frame)));
    const auto exported = reader.real_point_in(points, frame, index_t(0), id);
    const auto input =
        tf::point<TestType, 3>(tf::transformed(points[id], frame));
    REQUIRE(exported[0] == input[0]);
    REQUIRE(exported[1] == input[1]);
    REQUIRE(exported[2] == input[2]);
  }
}

TEST_CASE("door: a band below one lattice unit is still the identity",
          "[exact][door]") {
  scene_t<real_t> scene;
  scene.add(tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
  scene.build(1e-12);
  REQUIRE(scene.lattice.tolerance_int() == int_t(0));
  REQUIRE(scene.lattice.moved_points().size() == 0);
}

TEMPLATE_TEST_CASE("door: a form nothing pulls is read and exported as at "
                   "zero",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(rotated_box<TestType>());
  scene.add(translated_box(TestType(3), TestType(0), TestType(0)));
  scene.build(1e-4);

  REQUIRE(scene.lattice.tolerance_int() > lattice_int_t(0));
  REQUIRE(scene.lattice.moved_points().size() == 0);
  const auto reader = scene.lattice.reader(scene.apply_to_form());
  const auto &converter = scene.lattice.converter();
  for (index_t tag = 0; tag < index_t(2); ++tag) {
    const auto form = scene.operands[std::size_t(tag)]->form();
    const auto frame = tf::frame_of(form);
    const auto points = form.points();
    const auto base = scene.lattice.vertex_offsets()[std::size_t(tag)];
    for (index_t id = 0; id < index_t(points.size()); ++id) {
      REQUIRE(reader(int(tag), id) ==
              converter.convert(tf::transformed(points[id], frame)));
      const auto exported = reader.real_point_in(points, frame, base, id);
      const auto input =
          tf::point<TestType, 3>(tf::transformed(points[id], frame));
      REQUIRE(exported[0] == input[0]);
      REQUIRE(exported[1] == input[1]);
      REQUIRE(exported[2] == input[2]);
    }
  }
}

TEMPLATE_TEST_CASE("door: a doubled wall becomes one, and the door says so",
                   "[exact][door][pool]", float, double) {
  scene_t<TestType> scene;
  scene.add(doubled_wall_mesh(TestType(1e-5)));
  scene.build(5e-5);

  const auto placed = scene.placed_all();
  REQUIRE(placed.size() == 8);
  for (std::size_t i = 0; i < 4; ++i)
    REQUIRE(placed[i] == placed[i + 4]);
}

TEMPLATE_TEST_CASE("door: a wall vertex stands on its own plane and does "
                   "not move",
                   "[exact][door]", float, double) {
  scene_t<TestType> scene;
  scene.add(wall_mesh(TestType(0)));
  scene.build(1e-4);
  REQUIRE(scene.lattice.moved_points().size() == 0);
}

TEMPLATE_TEST_CASE("door: two banded walls of four planes become one plane",
                   "[exact][door][pool]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  const auto tilt = TestType(5e-7);
  const auto lift = TestType(6e-5);
  const double tolerance = 1e-4;

  scene_t<TestType> scene;
  scene.add(pooled_wall_mesh<TestType>(TestType(0), TestType(0.4), TestType(0),
                                       tilt));
  scene.add(pooled_wall_mesh<TestType>(TestType(0.6), TestType(1), lift, tilt));
  scene.build(tolerance);

  REQUIRE(scene.lattice.tolerance_int() > lattice_int_t(0));
  const auto placed = scene.placed_all();
  REQUIRE(placed.size() == 8);
  const auto &converter = scene.lattice.converter();
  tf::buffer<tf::point<lattice_int_t, 3>> input;
  for (index_t tag = 0; tag < index_t(2); ++tag) {
    const auto form = scene.operands[std::size_t(tag)]->form();
    const auto frame = tf::frame_of(form);
    const auto points = form.points();
    for (index_t id = 0; id < index_t(points.size()); ++id)
      input.push_back(converter.convert(tf::transformed(points[id], frame)));
  }
  REQUIRE(input.size() == 8);

  // THE PREMISE: the four faces are four planes and the input is not one.
  // Asked of the converter alone, which is where the operands stand before
  // the door moves anything.
  int off_plane = 0;
  for (std::size_t k = 3; k < input.size(); ++k)
    off_plane += tf::exact::orient3d_sign<lattice_int_t>(
                     input[0], input[1], input[2], input[k]) != 0
                     ? 1
                     : 0;
  REQUIRE(off_plane > 0);

  // BOTH WAYS: every placed vertex stands on the plane the other wall's own
  // corners state, and on the plane its own do.
  const auto first = pooled_plane_of<lattice_int_t>(placed, 0);
  const auto second = pooled_plane_of<lattice_int_t>(placed, 4);
  REQUIRE(first.size == 3);
  REQUIRE(second.size == 3);
  for (std::size_t k = 0; k < placed.size(); ++k) {
    REQUIRE(tf::exact::orient3d_sign<lattice_int_t>(
                first.point[0], first.point[1], first.point[2], placed[k]) ==
            0);
    REQUIRE(tf::exact::orient3d_sign<lattice_int_t>(
                second.point[0], second.point[1], second.point[2], placed[k]) ==
            0);
  }

  // and the two walls carry ONE plane NAME, which is the fact the arrangement
  // reads: coplanarity through a shared canonical quadruple, not proximity
  REQUIRE(tf::exact::make_canonical_plane<lattice_int_t>(first) ==
          tf::exact::make_canonical_plane<lattice_int_t>(second));

  // AND IT IS THE OPERANDS' OWN PLANE. What the walls came to rest on stands
  // on original vertices that did not move at all, which no grid plane can
  // promise — and at int32, where the door's naming pitch and the chart's are
  // the same number, it is the whole of what the partition changed here.
  int unmoved = 0;
  for (std::size_t k = 0; k < placed.size(); ++k)
    unmoved += placed[k] == input[k] ? 1 : 0;
  REQUIRE(unmoved >= 3);
}

TEMPLATE_TEST_CASE("door: the pooled scene is byte-inert at a zero band",
                   "[exact][door][pool]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(pooled_wall_mesh<TestType>(TestType(0), TestType(0.4), TestType(0),
                                       TestType(3e-7)));
  scene.add(pooled_wall_mesh<TestType>(TestType(0.6), TestType(1),
                                       TestType(6e-5), TestType(3e-7)));
  scene.build(0.0);

  REQUIRE(scene.lattice.tolerance_int() == lattice_int_t(0));
  REQUIRE(scene.lattice.moved_points().size() == 0);

  const auto reader = scene.lattice.reader(scene.apply_to_form());
  const auto &converter = scene.lattice.converter();
  for (index_t tag = 0; tag < index_t(2); ++tag) {
    const auto form = scene.operands[std::size_t(tag)]->form();
    const auto frame = tf::frame_of(form);
    const auto points = form.points();
    for (index_t id = 0; id < index_t(points.size()); ++id)
      REQUIRE(reader(int(tag), id) ==
              converter.convert(tf::transformed(points[id], frame)));
  }
}

TEST_CASE("door: two forms whose corners nearly coincide land on one integer",
          "[exact][door]") {
  const real_t scatter = real_t(1e-5);
  scene_t<real_t> scene;
  scene.add(tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
  scene.add(translated_box(real_t(1) + scatter * real_t(0.5),
                           real_t(1) - scatter * real_t(0.3),
                           real_t(1) + scatter * real_t(0.4)));
  scene.build(double(scatter));

  const auto placed = scene.placed_all();
  REQUIRE(placed.size() == 16);
  index_t welds = 0;
  for (index_t a = 0; a < index_t(8); ++a)
    for (index_t b = index_t(8); b < index_t(16); ++b)
      if (placed[std::size_t(a)] == placed[std::size_t(b)])
        ++welds;
  REQUIRE(welds == index_t(1));
}

TEMPLATE_TEST_CASE("door: a tilted near-miss drops onto the plane its face "
                   "commits, or the census says the plane could not commit",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  // a box turned generically, and a triangle whose first corner stands a
  // third of the band off the centre of the box's first face
  const double band = 1e-6;
  auto box =
      tf::make_box_mesh<index_t>(TestType(10), TestType(10), TestType(10));
  for (std::size_t i = 0; i < box.points_buffer().size(); ++i) {
    auto p = box.points_buffer()[i];
    const double x = 0.6 * double(p[0]) - 0.8 * double(p[1]);
    const double y = 0.8 * double(p[0]) + 0.6 * double(p[1]);
    const double z = double(p[2]);
    p[0] = TestType((5 * x - 12 * z) / 13 + 0.123);
    p[1] = TestType(y - 0.077);
    p[2] = TestType((12 * x + 5 * z) / 13 + 0.031);
  }
  const auto face = box.faces()[0];
  std::array<double, 3> centre{0, 0, 0}, e1{}, e2{}, normal{};
  for (int d = 0; d < 3; ++d) {
    for (int k = 0; k < 3; ++k)
      centre[std::size_t(d)] += double(box.points()[face[k]][d]) / 3;
    e1[std::size_t(d)] =
        double(box.points()[face[1]][d]) - double(box.points()[face[0]][d]);
    e2[std::size_t(d)] =
        double(box.points()[face[2]][d]) - double(box.points()[face[0]][d]);
  }
  normal = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]};
  const double length = std::sqrt(
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  tf::polygons_buffer<index_t, TestType, 3, 3> visitor;
  for (const double lift : {0.3 * band, 3.0, 3.0})
    visitor.points_buffer().emplace_back(
        TestType(centre[0] + normal[0] / length * lift +
                 (lift > 1 ? e1[0] * 0.1 * (visitor.points().size()) : 0.0)),
        TestType(centre[1] + normal[1] / length * lift +
                 (lift > 1 ? e2[1] * 0.1 * (visitor.points().size()) : 0.0)),
        TestType(centre[2] + normal[2] / length * lift));
  visitor.faces_buffer().emplace_back(0, 1, 2);

  scene_t<TestType> scene;
  scene.add(box);
  scene.add(visitor);
  scene.build(band);
  const auto &census = scene.lattice.census();
  REQUIRE(census.contacts == 1);
  const auto placed = scene.placed_all();
  if (std::is_same<TestType, double>::value) {
    // the face commits a small-normal plane its corners and its visitor all
    // reach within the band, and the visitor drops onto it exactly
    REQUIRE(census.moved_commits == 1);
    REQUIRE(tf::exact::orient3d_sign<lattice_int_t>(
                placed[std::size_t(face[0])], placed[std::size_t(face[1])],
                placed[std::size_t(face[2])], placed[8]) == 0);
  } else {
    // the face is far larger than the band at this width: no small-normal
    // plane keeps its corners within the band, so it stays its exact self,
    // the visitor keeps its input point, and the census says so
    REQUIRE(census.uncommittable == 1);
    REQUIRE(census.unheld_contacts == 1);
    REQUIRE(scene.lattice.moved_points().size() == 0);
  }
}

TEMPLATE_TEST_CASE("door: a wall beside a pooled one does not fold",
                   "[exact][door][pool]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  // two walls a hair apart, the first carrying a perpendicular flange. A
  // face commits to a plane only when every corner of it lands, so no face
  // is ever left half on its plane: the second wall pools and comes out
  // flat, and the flanged one — whose faces carry feature corners and
  // speak for no pool — keeps every vertex where the input put it.
  const double tilt = 5e-7, lift = 6e-5;
  tf::polygons_buffer<index_t, TestType, 3, 3> flanged;
  for (const auto &q :
       {std::array<double, 3>{0, 0, 0}, std::array<double, 3>{0.4, 0, 0},
        std::array<double, 3>{0.4, 1, tilt}, std::array<double, 3>{0, 1, 0},
        std::array<double, 3>{0.4, 0, -0.5},
        std::array<double, 3>{0.4, 1, -0.5}})
    flanged.points_buffer().emplace_back(TestType(q[0]), TestType(q[1]),
                                         TestType(q[2]));
  flanged.faces_buffer().emplace_back(0, 1, 2);
  flanged.faces_buffer().emplace_back(0, 2, 3);
  flanged.faces_buffer().emplace_back(1, 4, 5);
  flanged.faces_buffer().emplace_back(1, 5, 2);

  scene_t<TestType> scene;
  scene.add(std::move(flanged));
  scene.add(pooled_wall_mesh<TestType>(TestType(0.6), TestType(1),
                                       TestType(lift), TestType(tilt)));
  scene.build(1e-4);
  REQUIRE(scene.lattice.moved_points().size() > 0);
  const auto placed = scene.placed_all();
  const auto &converter = scene.lattice.converter();
  const auto first = scene.operands[0]->form();
  for (index_t id = 0; id < index_t(6); ++id)
    REQUIRE(placed[std::size_t(id)] ==
            converter.convert(
                tf::transformed(first.points()[id], tf::frame_of(first))));
  REQUIRE(tf::exact::orient3d_sign<lattice_int_t>(placed[6], placed[7],
                                                  placed[8], placed[9]) == 0);
}

TEST_CASE("door: two segments touch exactly when they share a point",
          "[exact][door]") {
  using p_t = tf::point<int_t, 3>;
  const auto touch = [](p_t a0, p_t a1, p_t b0, p_t b1) {
    return tf::exact::door::segments_touch<int_t>(a0, a1, b0, b1);
  };
  // a crossing, an end on the other's interior, a shared end
  REQUIRE(touch({-5, 0, 7}, {5, 0, 7}, {0, -5, 7}, {0, 5, 7}));
  REQUIRE(touch({-5, 0, 7}, {5, 0, 7}, {0, 0, 7}, {0, 5, 7}));
  REQUIRE(touch({-5, 0, 7}, {5, 0, 7}, {5, 0, 7}, {9, 9, 9}));
  // a crossing of the lines beyond one segment's end
  REQUIRE_FALSE(touch({-5, 0, 7}, {5, 0, 7}, {0, 1, 7}, {0, 5, 7}));
  // skew lines
  REQUIRE_FALSE(touch({-5, 0, 7}, {5, 0, 7}, {0, -5, 8}, {0, 5, 8}));
  // parallel lines in one plane that are not one line never meet
  REQUIRE_FALSE(touch({-5, -5, -5}, {5, -5, -5}, {-5, -5, 0}, {5, -5, 0}));
  // one line: overlapping spans touch, disjoint ones do not
  REQUIRE(touch({0, 0, 0}, {4, 0, 0}, {3, 0, 0}, {9, 0, 0}));
  REQUIRE_FALSE(touch({0, 0, 0}, {4, 0, 0}, {5, 0, 0}, {9, 0, 0}));
  // a segment of no length touches only where its point lies on the other
  REQUIRE(touch({2, 0, 0}, {2, 0, 0}, {0, 0, 0}, {4, 0, 0}));
  REQUIRE_FALSE(touch({2, 1, 0}, {2, 1, 0}, {0, 0, 0}, {4, 0, 0}));
}

TEST_CASE("door: two segments cross exactly when they meet inside both",
          "[exact][door]") {
  using p_t = tf::point<int_t, 3>;
  const auto cross = [](p_t a0, p_t a1, p_t b0, p_t b1) {
    return tf::exact::door::segments_cross<int_t>(a0, a1, b0, b1);
  };
  REQUIRE(cross({-5, 0, 7}, {5, 0, 7}, {0, -5, 7}, {0, 5, 7}));
  // an end on the other's interior, a shared end: an end on the other
  REQUIRE_FALSE(cross({-5, 0, 7}, {5, 0, 7}, {0, 0, 7}, {0, 5, 7}));
  REQUIRE_FALSE(cross({-5, 0, 7}, {5, 0, 7}, {5, 0, 7}, {9, 9, 9}));
  REQUIRE_FALSE(cross({-5, 0, 7}, {5, 0, 7}, {0, 1, 7}, {0, 5, 7}));
  REQUIRE_FALSE(cross({-5, 0, 7}, {5, 0, 7}, {0, -5, 8}, {0, 5, 8}));
  REQUIRE_FALSE(cross({-5, -5, -5}, {5, -5, -5}, {-5, -5, 0}, {5, -5, 0}));
  // one line overlapping is ends on the other, not a crossing
  REQUIRE_FALSE(cross({0, 0, 0}, {4, 0, 0}, {3, 0, 0}, {9, 0, 0}));
  REQUIRE_FALSE(cross({2, 0, 0}, {2, 0, 0}, {0, 0, 0}, {4, 0, 0}));
}

TEST_CASE("door: a point stands on a closed face exactly or not at all",
          "[exact][door]") {
  using p_t = tf::point<int_t, 3>;
  const std::array<p_t, 3> face{p_t{0, 0, 3}, p_t{8, 0, 3}, p_t{0, 8, 3}};
  REQUIRE(tf::exact::door::point_on_face(p_t{2, 2, 3}, face));
  REQUIRE(tf::exact::door::point_on_face(p_t{4, 0, 3}, face));
  REQUIRE(tf::exact::door::point_on_face(p_t{0, 0, 3}, face));
  REQUIRE(tf::exact::door::point_on_face(p_t{4, 4, 3}, face));
  REQUIRE_FALSE(tf::exact::door::point_on_face(p_t{5, 5, 3}, face));
  REQUIRE_FALSE(tf::exact::door::point_on_face(p_t{2, 2, 4}, face));
  // a face bounding no area holds nothing
  const std::array<p_t, 3> sliver{p_t{0, 0, 0}, p_t{4, 0, 0}, p_t{8, 0, 0}};
  REQUIRE_FALSE(tf::exact::door::point_on_face(p_t{2, 0, 0}, sliver));
}
