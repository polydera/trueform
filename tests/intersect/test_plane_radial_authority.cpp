/**
 * @file test_plane_radial_authority.cpp
 * @brief The one producer of a canonical piece's radial carrier line.
 *
 * A piece is handed its COMPLETE definition span, and the answer is the line
 * they all stand on, in the piece's own key order. The scenes here state each
 * language a definition may speak — an original edge of a source face, and an
 * interior cut's producing pair — and then the cases that made the producer
 * central: a piece carrying several statements, an original edge that is also
 * an intersection edge, a split piece inheriting its parent's line, and the
 * ill-posed pieces whose statements name different lines or different
 * directions and are refused rather than silently ordered.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/intersect/graph/plane_edge_radial_authority.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using index_t = int;
using radial_authority_int_t = tf::exact::int32;
using pt3_t = tf::exact::pt3<radial_authority_int_t>;
using def_t = tf::intersect::graph::plane_edge_def<index_t>;

namespace graph = tf::intersect::graph;

/// One tag's points and faces, plus the face groups a definition names.
///
/// Face group `g` is `(tag, object)` of `descriptors[g]`; the scenes below
/// keep one group per face of one tag, so a group id and an object id agree.
struct scene_t {
  std::vector<pt3_t> points;
  std::vector<std::vector<index_t>> faces;
  std::vector<graph::face_descriptor<index_t>> descriptors;
};

auto authority_of(const scene_t &scene, const std::vector<def_t> &definitions)
    -> graph::plane_edge_radial_authority {
  const auto descriptor_of_face =
      [&](index_t face) -> const graph::face_descriptor<index_t> & {
    return scene.descriptors[std::size_t(face)];
  };
  const auto apply_to_face = [&](int, index_t object, const auto &apply) {
    apply(tf::make_range(scene.faces[std::size_t(object)]));
  };
  const auto get_point = [&](int, index_t id) {
    return scene.points[std::size_t(id)];
  };
  return graph::make_plane_edge_radial_authority<index_t,
                                                 radial_authority_int_t>(
      tf::make_range(definitions), descriptor_of_face, apply_to_face,
      get_point);
}

/// Face 0 in z = 0 and face 1 in y = 0, both standing on the x axis segment
/// (0,0,0)-(100,0,0); face 2 in x = 0, which meets face 0 on the y axis
/// instead. Face 3 repeats face 0's edge with the opposite winding, the way
/// the neighbour across a shared edge states it.
auto axis_scene() -> scene_t {
  scene_t scene;
  scene.points = {pt3_t{0, 0, 0},   pt3_t{100, 0, 0}, pt3_t{0, 100, 0},
                  pt3_t{0, 0, 100}, pt3_t{0, 60, 60}};
  scene.faces = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 0, 4}};
  for (std::size_t f = 0; f < scene.faces.size(); ++f)
    scene.descriptors.push_back({index_t(0), index_t(f)});
  return scene;
}

/// The boundary definition one face states on one of its sides.
auto boundary_def(const scene_t &scene, index_t object, std::size_t side)
    -> def_t {
  return graph::make_plane_boundary_side_def<index_t>(
      std::int16_t(0), tf::make_range(scene.faces[std::size_t(object)]), side,
      index_t(0), object, object);
}

/// The interior-cut definition a face states against a partner, carrying the
/// key-order sense against that pair's carrier line.
auto cut_def(index_t face, index_t partner, int radial) -> def_t {
  std::uint8_t flags = graph::plane_edge_fan_flag | graph::plane_edge_radial_flag;
  if (radial < 0)
    flags = std::uint8_t(flags | graph::plane_edge_radial_reversed_flag);
  return def_t{index_t(0),      index_t(1),      index_t(0),
               face,            partner,         std::int16_t(-1),
               std::int16_t(-1), std::int16_t(0), std::int16_t(-1),
               std::int16_t(-1), flags};
}

} // namespace

TEST_CASE("radial authority: an original edge names the line, whichever way "
          "its face is wound",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // face 0 walks the x axis segment forward, face 3 walks it back; both
  // definitions key on the same endpoint pair, so both must state the same
  // key-order direction
  const auto forward = authority_of(scene, {boundary_def(scene, 0, 0)});
  CHECK(forward.valid);
  CHECK(forward.axis == 0);
  CHECK(forward.sign == 1);

  const auto backward = authority_of(scene, {boundary_def(scene, 3, 0)});
  CHECK(backward.valid);
  CHECK(backward.axis == 0);
  CHECK(backward.sign == 1);
}

TEST_CASE("radial authority: two faces on one original edge reconcile",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();
  const auto both =
      authority_of(scene, {boundary_def(scene, 0, 0), boundary_def(scene, 3, 0)});
  CHECK(both.valid);
  CHECK(both.axis == 0);
  CHECK(both.sign == 1);
}

TEST_CASE("radial authority: statements naming different lines refuse",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // face 2's side 0 runs the y axis: two original edges welded into one
  // canonical identity, and no order around them is meaningful
  const auto crossed =
      authority_of(scene, {boundary_def(scene, 0, 0), boundary_def(scene, 2, 0)});
  CHECK_FALSE(crossed.valid);
}

TEST_CASE("radial authority: statements naming opposite directions refuse",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // the same line, but face 3's definition is stated as if its emission ran
  // with the key order instead of against it
  auto opposed = boundary_def(scene, 3, 0);
  opposed.flags =
      std::uint8_t(opposed.flags & ~graph::plane_edge_reversed_flag);
  const auto refused =
      authority_of(scene, {boundary_def(scene, 0, 0), opposed});
  CHECK_FALSE(refused.valid);
}

TEST_CASE("radial authority: a split piece inherits its parent's line",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // a piece of the edge keeps the side and drops the whole-side claim; its
  // endpoints moved, its line did not
  auto piece = boundary_def(scene, 0, 0);
  piece.flags =
      std::uint8_t(piece.flags & ~graph::plane_edge_whole_side_flag);
  piece.point_0 = index_t(7);
  piece.point_1 = index_t(9);
  piece.point_tag_0 = std::int16_t(-1);
  piece.point_tag_1 = std::int16_t(-1);
  const auto split = authority_of(scene, {piece});
  CHECK(split.valid);
  CHECK(split.axis == 0);
  CHECK(split.sign == 1);

  // a piece whose own key order runs against its parent's carries the flip
  auto flipped = piece;
  flipped.flags =
      std::uint8_t(flipped.flags ^ graph::plane_edge_reversed_flag);
  const auto turned = authority_of(scene, {flipped});
  CHECK(turned.valid);
  CHECK(turned.axis == 0);
  CHECK(turned.sign == -1);
}

TEST_CASE("radial authority: an interior cut names its producing pair's line",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // n(face 0) x n(face 1) runs along +x, so the stated sense IS the answer
  const auto along = authority_of(scene, {cut_def(0, 1, 1)});
  CHECK(along.valid);
  CHECK(along.axis == 0);
  CHECK(along.sign == 1);

  const auto against = authority_of(scene, {cut_def(0, 1, -1)});
  CHECK(against.valid);
  CHECK(against.axis == 0);
  CHECK(against.sign == -1);
}

TEST_CASE("radial authority: both members of one pair reconcile",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // each face states the sense against ITS OWN cross, and the two crosses are
  // opposite, so the raw bits differ and the reconciled answer does not
  const auto pair = authority_of(scene, {cut_def(0, 1, 1), cut_def(1, 0, -1)});
  CHECK(pair.valid);
  CHECK(pair.axis == 0);
  CHECK(pair.sign == 1);

  const auto disagreeing =
      authority_of(scene, {cut_def(0, 1, 1), cut_def(1, 0, 1)});
  CHECK_FALSE(disagreeing.valid);
}

TEST_CASE("radial authority: a pair carrying another line refuses",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // face 0 with face 2 meets on the y axis, not the x axis
  const auto crossed =
      authority_of(scene, {cut_def(0, 1, 1), cut_def(0, 2, 1)});
  CHECK_FALSE(crossed.valid);
}

TEST_CASE("radial authority: an original edge outranks the cut that shares it",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();

  // the shared-edge case: face 1 cuts face 0 exactly along face 0's own side,
  // so the piece carries both languages and they must agree
  const auto shared =
      authority_of(scene, {boundary_def(scene, 0, 0), cut_def(1, 0, -1)});
  CHECK(shared.valid);
  CHECK(shared.axis == 0);
  CHECK(shared.sign == 1);

  // the same statement turned against the original edge is a contradiction
  const auto conflicting =
      authority_of(scene, {boundary_def(scene, 0, 0), cut_def(1, 0, 1)});
  CHECK_FALSE(conflicting.valid);
}

TEST_CASE("radial authority: a piece stating no line refuses",
          "[intersect][graph][radial-authority]") {
  const auto scene = axis_scene();
  auto silent = cut_def(0, 1, 1);
  silent.flags = std::uint8_t(silent.flags & ~graph::plane_edge_radial_flag);
  CHECK_FALSE(authority_of(scene, {silent}).valid);
}
