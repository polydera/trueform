/**
 * @file test_door_placement.cpp
 * @brief The door: where an input vertex stands once a tolerance is given.
 *
 * The door moves every original vertex onto a lattice point of the planes it
 * lies on, and the arrangement is then the EXACT arrangement of the moved
 * mesh. Two properties carry that contract and both are asserted here rather
 * than measured: a rank-1 placement lies ON its plane, and every placement —
 * whatever rank stated it, however adversarial the names — stands within the
 * tolerance of the vertex it replaced. The second is what the pair search's
 * `2 T` descent rests on, so it is proven against names built to break it.
 *
 * Every lattice the library resolves is driven: a float input names int32 and
 * a double input int64, and the scenes run on both. The generic-direction
 * scene is what separates them — an axis triple survives any bound, so a door
 * inert on the wider lattice still answers a cube, and only a face whose
 * offset needs the wider rung reports it.
 *
 * The scenes are an axis-aligned cube whose corners meet three representable
 * planes and stay an exact cube, a doubled wall whose two sheets become one,
 * and two forms whose corner names agree and therefore land on ONE integer —
 * the weld the contract promises is identity and not proximity.
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
#include <trueform/exact/door/admits_placement.hpp>
#include <trueform/exact/door/gather_vertex_candidates.hpp>
#include <trueform/exact/door/place_on_plane.hpp>
#include <trueform/exact/door/plane_frame.hpp>
#include <trueform/exact/door/place_vertex.hpp>
#include <trueform/exact/door/placement_tables.hpp>
#include <trueform/exact/door/plane_step.hpp>
#include <trueform/exact/door/quantized_plane.hpp>
#include <trueform/exact/door/round_div.hpp>
#include <trueform/exact/door/round_to_wide.hpp>
#include <trueform/exact/door/wide_dot.hpp>
#include <trueform/exact/input_lattice.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/exact/vertex_converter.hpp>
#include <trueform/geometry/make_box_mesh.hpp>

#include "tagged_operand.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using index_t = int;
using real_t = float;
using int_t = tf::exact::resolve_int_type<tf::none_t, real_t>;
using wide_t = tf::exact::meta<int_t>::T1;
using plane_t = tf::exact::door::quantized_plane<int_t>;

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
    lattice.build(tf::exact::make_vertex_converter<lattice_int_t, RealType>(
                      tf::make_range(forms.data(), forms.data() + forms.size())),
                  apply_to_form(), index_t(operands.size()), tolerance);
  }

  auto placed(index_t tag, index_t id) const -> tf::point<lattice_int_t, 3> {
    return lattice.reader(apply_to_form())(int(tag), id);
  }

  /// The tables the door builds for itself, so a fixture can ask by which
  /// RANK a vertex was placed and on which names it stands.
  auto build_tables(tables_t &tables) const -> void {
    const auto n_tags = index_t(operands.size());
    tf::buffer<index_t> face_offsets;
    face_offsets.allocate(std::size_t(n_tags) + 1);
    face_offsets[0] = index_t(0);
    for (index_t tag = 0; tag < n_tags; ++tag)
      apply_to_form()(tag, [&, tag](const auto &form) {
        face_offsets[std::size_t(tag) + 1] =
            face_offsets[std::size_t(tag)] + index_t(form.faces().size());
      });
    tables.build(lattice.converter(), apply_to_form(), n_tags,
                 lattice.vertex_offsets(), face_offsets,
                 lattice.tolerance_int());
  }
};

/// One vertex through the door exactly as `place_vertices` runs it, with the
/// candidate list left for the caller to read.
template <typename Tables, typename Int, typename Candidates>
auto placement_of(const Tables &tables, index_t flat, Int tolerance,
                  Candidates &candidates)
    -> tf::exact::door::vertex_placement<Int> {
  const auto direction = tf::exact::door::gather_vertex_candidates(
      tables, flat, tolerance, candidates);
  return tf::exact::door::place_vertex(
      tables.points[std::size_t(flat)],
      tf::make_range(candidates.begin(), candidates.end()), direction,
      tolerance);
}

/// A quad in the plane `z = height`, as two triangles.
template <typename RealType>
auto wall_mesh(RealType height) -> tf::polygons_buffer<index_t, RealType, 3, 3> {
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

/// The band the generic-direction scene is driven at, per lattice. A band is
/// a fraction of the model, but the rank-1 solve states its residual against
/// a Bezout vector on the rung above the lattice, so the finer lattice — the
/// one a double resolves — carries the same fraction as a much larger number
/// and admits a smaller one.
template <typename RealType> auto generic_scene_band() -> double {
  return std::is_same<RealType, float>::value ? 1e-4 : 1e-7;
}

auto plane_of(wide_t nx, wide_t ny, wide_t nz, wide_t offset) -> plane_t {
  plane_t plane;
  plane.normal = {nx, ny, nz};
  plane.offset = offset;
  return plane;
}

/// A deterministic spread of adversarial names: axis triples pushed far off
/// the vertex, near-parallel pairs, and directions at the lattice's own
/// magnitude, where every product the solvers take is at its widest.
auto adversarial_candidates(std::uint32_t seed) -> std::vector<plane_t> {
  const auto next = [&seed]() -> std::uint32_t {
    seed = seed * 1664525u + 1013904223u;
    return seed;
  };
  const auto span = [&next](wide_t magnitude) -> wide_t {
    return wide_t(std::int64_t(next() % 2000001u) - 1000000) * magnitude;
  };
  std::vector<plane_t> candidates;
  candidates.push_back(plane_of(1, 0, 0, span(1000)));
  candidates.push_back(plane_of(0, 1, 0, span(1000)));
  candidates.push_back(plane_of(0, 0, 1, span(1000)));
  candidates.push_back(plane_of(1000000, 1, 0, span(1000000)));
  candidates.push_back(plane_of(1000000, 2, 0, span(1000000)));
  candidates.push_back(plane_of(wide_t(next() % 100000u) + 1,
                                wide_t(next() % 100000u) + 1,
                                wide_t(next() % 100000u) + 1, span(100000)));
  candidates.push_back(plane_of(2147483647, 2147483646, 2147483645,
                                span(1000000000)));
  return candidates;
}

} // namespace

TEST_CASE("door: the rounding divide answers every numerator its type holds",
          "[exact][door]") {
  using wider_t = tf::exact::meta<int_t>::T2;
  const wider_t top =
      wider_t(1) << unsigned(tf::exact::meta<int_t>::t2_bits - 1);
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

TEST_CASE("door: a rank-1 placement lies exactly on its plane",
          "[exact][door]") {
  const std::array<std::array<wide_t, 3>, 5> directions{
      {{{1, 0, 0}}, {{0, 0, 1}}, {{3, 4, 12}}, {{1, 1, 1}}, {{7, -5, 11}}}};
  for (const auto &normal : directions)
    for (wide_t residual : {wide_t(0), wide_t(1), wide_t(-1), wide_t(97),
                            wide_t(-12345), wide_t(1000003)}) {
      std::array<wide_t, 3> step{};
      REQUIRE(tf::exact::door::place_on_plane<int_t>(
          tf::exact::door::make_plane_frame<int_t>(normal), residual,
          step));
      REQUIRE(tf::exact::door::wide_dot<int_t>(normal, step) ==
              tf::exact::meta<int_t>::T2(residual));
    }
}

TEST_CASE("door: the rank-1 solve refuses the step it cannot state, and the "
          "pair still certifies",
          "[exact][door]") {
  using wider_t = tf::exact::meta<int_t>::T2;
  const std::array<wide_t, 3> normal{1073741827, 2147483647, 999999937};
  const tf::point<int_t, 3> original{1000, -2000, 3000};
  const auto height = tf::exact::door::wide_dot<int_t>(normal, original);
  const wide_t residual = 1000000000000000;

  // The premise: the Bezout vector of a direction at the lattice's own
  // magnitude, times a residual of the size a pair hands it, leaves the rung
  // the rank-1 solve is stated on.
  const auto frame = tf::exact::door::make_plane_frame<int_t>(normal);
  const auto bound =
      wider_t(tf::exact::door::wide_placement_bound<int_t>());
  const auto onto = wider_t(residual) * wider_t(frame.s[0]);
  REQUIRE((onto > bound || onto < -bound));

  std::array<wide_t, 3> step{};
  REQUIRE_FALSE(
      tf::exact::door::place_on_plane<int_t>(frame, residual, step));

  // And the rank the refusal falls to still answers inside the band: the
  // pair's own line carries the vertex when no lattice point of the first
  // plane can be stated.
  const std::vector<plane_t> candidates{
      plane_of(normal[0], normal[1], normal[2],
               static_cast<wide_t>(height) + residual),
      plane_of(0, 0, 1, 3100)};
  const int_t tolerance = 1000003;
  const auto placement = tf::exact::door::place_vertex(
      original,
      tf::make_range(candidates.data(), candidates.data() + candidates.size()),
      tf::vector<double, 3>{0.0, 0.0, 1.0}, tolerance);
  const std::array<wide_t, 3> at{wide_t(placement.point[0]),
                                 wide_t(placement.point[1]),
                                 wide_t(placement.point[2])};
  REQUIRE(placement.rank == 2);
  REQUIRE(tf::exact::door::admits_placement(original, at, tolerance));
}

TEST_CASE("door: the motion bound is a theorem, not a measurement",
          "[exact][door]") {
  const tf::point<int_t, 3> original{1000, -2000, 3000};
  const tf::vector<double, 3> direction{0.3, -0.5, 0.81};
  std::array<int, 4> census{};
  for (std::uint32_t seed = 1; seed <= 64; ++seed) {
    const auto candidates = adversarial_candidates(seed);
    for (int_t tolerance : {int_t(1), int_t(17), int_t(4096), int_t(1000003)}) {
      const auto placement = tf::exact::door::place_vertex(
          original, tf::make_range(candidates.data(),
                                   candidates.data() + candidates.size()),
          direction, tolerance);
      const std::array<wide_t, 3> at{wide_t(placement.point[0]),
                                     wide_t(placement.point[1]),
                                     wide_t(placement.point[2])};
      REQUIRE(tf::exact::door::admits_placement(original, at, tolerance));
      if (placement.rank == 0)
        REQUIRE(placement.point == original);
      ++census[std::size_t(placement.rank)];
    }
  }
  // The names carry offsets far enough off the vertex that the rank-1 solve
  // inside the pair is asked for a product past its rung, so the spread does
  // reach the rank whose refusal path this proves.
  REQUIRE(census[2] > 0);
}

TEST_CASE("door: a placement is a pure function of the names and the band",
          "[exact][door]") {
  const std::vector<plane_t> candidates{
      plane_of(1, 0, 0, 4000), plane_of(0, 1, 0, -8000),
      plane_of(0, 0, 1, 12000)};
  const auto names =
      tf::make_range(candidates.data(), candidates.data() + candidates.size());
  const tf::vector<double, 3> direction{0.0, 0.0, 1.0};
  const int_t tolerance = 1000;
  const auto first = tf::exact::door::place_vertex(
      tf::point<int_t, 3>{4300, -8200, 11700}, names, direction, tolerance);
  const auto second = tf::exact::door::place_vertex(
      tf::point<int_t, 3>{3800, -7900, 12400}, names, direction, tolerance);
  REQUIRE(first.rank == 3);
  REQUIRE(second.rank == 3);
  REQUIRE(first.point == second.point);
  REQUIRE(first.point == (tf::point<int_t, 3>{4000, -8000, 12000}));
}

TEMPLATE_TEST_CASE("door: a tolerance of zero is the identity",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(tf::make_box_mesh<index_t>(TestType(1), TestType(1), TestType(1)));
  scene.build(0.0);

  REQUIRE(scene.lattice.tolerance_int() == lattice_int_t(0));
  REQUIRE(scene.lattice.placed_points().size() == 0);

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
  REQUIRE(scene.lattice.placed_points().size() == 0);
}

TEMPLATE_TEST_CASE("door: an axis-aligned cube stays an exact cube",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(tf::make_box_mesh<index_t>(TestType(1), TestType(1), TestType(1)));
  scene.build(1e-4);

  const auto tolerance = scene.lattice.tolerance_int();
  REQUIRE(tolerance > lattice_int_t(0));
  const auto placed = scene.lattice.placed_points();
  REQUIRE(placed.size() == 8);
  auto extent = placed[0][0] < lattice_int_t(0) ? -placed[0][0] : placed[0][0];
  for (std::size_t i = 0; i < placed.size(); ++i)
    for (std::size_t k = 0; k < 3; ++k) {
      const auto value = placed[i][k];
      REQUIRE(value % tolerance == lattice_int_t(0));
      REQUIRE((value < lattice_int_t(0) ? -value : value) == extent);
    }
}

TEMPLATE_TEST_CASE("door: a generic direction is placed on every lattice",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  using lattice_wide_t = typename tf::exact::meta<lattice_int_t>::T1;
  using lattice_wider_t = typename tf::exact::meta<lattice_int_t>::T2;

  scene_t<TestType> scene;
  scene.add(rotated_box<TestType>());
  scene.add(tilted_wall_mesh<TestType>());
  scene.build(generic_scene_band<TestType>());

  const auto tolerance = scene.lattice.tolerance_int();
  REQUIRE(tolerance > lattice_int_t(0));
  const auto placed = scene.lattice.placed_points();
  REQUIRE(placed.size() == 12);

  typename scene_t<TestType>::tables_t tables;
  scene.build_tables(tables);
  tf::small_vector<tf::exact::door::quantized_plane<lattice_int_t>, 16>
      candidates;

  // The cube's eight corners: three names meet at each, so rank 3 answers,
  // and it answers with a lattice point the vertex did not already occupy.
  for (index_t flat = 0; flat < index_t(8); ++flat) {
    const auto placement = placement_of(tables, flat, tolerance, candidates);
    const auto &original = tables.points[std::size_t(flat)];
    const std::array<lattice_wide_t, 3> at{lattice_wide_t(placement.point[0]),
                                           lattice_wide_t(placement.point[1]),
                                           lattice_wide_t(placement.point[2])};
    REQUIRE(placement.point == placed[std::size_t(flat)]);
    REQUIRE(tf::exact::door::admits_placement(original, at, tolerance));
    REQUIRE(placement.rank == 3);
    REQUIRE(candidates.size() >= 3);
    REQUIRE(placement.point != original);
  }

  // The tilted quad's four: one name each, so rank 1 states the answer, and
  // the four land on ONE plane of the direction's own grid — which is what
  // keeps a coplanar group coplanar.
  lattice_wider_t height(0);
  std::array<lattice_wide_t, 3> name{};
  for (index_t flat = index_t(8); flat < index_t(12); ++flat) {
    const auto placement = placement_of(tables, flat, tolerance, candidates);
    const auto &original = tables.points[std::size_t(flat)];
    const std::array<lattice_wide_t, 3> at{lattice_wide_t(placement.point[0]),
                                           lattice_wide_t(placement.point[1]),
                                           lattice_wide_t(placement.point[2])};
    REQUIRE(placement.point == placed[std::size_t(flat)]);
    REQUIRE(tf::exact::door::admits_placement(original, at, tolerance));
    REQUIRE(placement.rank == 1);
    REQUIRE(candidates.size() == 1);
    const auto on_plane =
        tf::exact::door::wide_dot<lattice_int_t>(candidates[0].normal, at);
    if (flat == index_t(8)) {
      name = candidates[0].normal;
      height = on_plane;
    } else {
      REQUIRE(candidates[0].normal == name);
      REQUIRE(on_plane == height);
    }
  }
  const auto step =
      tf::exact::door::plane_step<lattice_int_t>(name,
                                                    lattice_wide_t(tolerance));
  REQUIRE(height % lattice_wider_t(step) == lattice_wider_t(0));
}

TEMPLATE_TEST_CASE("door: a doubled wall becomes one, and the door says so",
                   "[exact][door]", float, double) {
  scene_t<TestType> scene;
  scene.add(doubled_wall_mesh(TestType(1e-5)));
  scene.build(5e-5);

  const auto placed = scene.lattice.placed_points();
  REQUIRE(placed.size() == 8);
  for (std::size_t i = 0; i < 4; ++i)
    REQUIRE(placed[i] == placed[i + 4]);
}

TEMPLATE_TEST_CASE("door: a wall vertex stands on its own plane and does "
                   "not move",
                   "[exact][door]", float, double) {
  using lattice_int_t = tf::exact::resolve_int_type<tf::none_t, TestType>;
  scene_t<TestType> scene;
  scene.add(wall_mesh(TestType(0)));
  scene.build(1e-4);

  const auto placed = scene.lattice.placed_points();
  REQUIRE(placed.size() == 4);
  for (std::size_t i = 0; i < placed.size(); ++i)
    REQUIRE(placed[i][2] == lattice_int_t(0));
}

TEST_CASE("door: two forms whose corner names agree land on one integer",
          "[exact][door]") {
  const real_t scatter = real_t(1e-5);
  scene_t<real_t> scene;
  scene.add(tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
  scene.add(translated_box(real_t(1) + scatter * real_t(0.5),
                           real_t(1) - scatter * real_t(0.3),
                           real_t(1) + scatter * real_t(0.4)));
  scene.build(double(scatter));

  const auto placed = scene.lattice.placed_points();
  REQUIRE(placed.size() == 16);
  index_t welds = 0;
  for (index_t a = 0; a < index_t(8); ++a)
    for (index_t b = index_t(8); b < index_t(16); ++b)
      if (placed[std::size_t(a)] == placed[std::size_t(b)])
        ++welds;
  REQUIRE(welds == index_t(1));
}
