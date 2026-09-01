/**
 * @file test_plane_graph_fan_bits.cpp
 * @brief Tests for the plane graph's intersection-edge statements
 *
 * Extraction has TWO sources — the base loops and the intersection records —
 * and a record's edge is an intersection edge whether or not it runs along a
 * side of the face that states it. So every non-coplanar record edge leaves
 * extraction carrying the fan flag and its producing pair's radial
 * orientation, and an edge that covers one of its face's sides carries that
 * side's ordinal and original edge too: one instance per side, complete.
 *
 * The scenes state the cases apart:
 *
 *   EXACT-EDGE MERGE   four faces meeting on one shared original edge — the
 *                      configuration where no interior cut exists at all;
 *   TILED PLANE x BOX  a box sheet whose tessellation edge lands directly on
 *                      a tiled plane's edge, continuing through it or resting
 *                      on it;
 *   EDGE ON FACE       an edge laid across a face's interior, whose own side
 *                      is the contact — the cut and the side in one span;
 *   BARE TOUCH         two forms sharing one whole original edge: a record is
 *                      a record, so the span is marked;
 *   CREASE             one form's own fold — no record, no mark;
 *   COPLANAR STACKS    contacts that pool instead of emitting: no mark
 *                      anywhere in the pooled plane.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"
#include "input_lattice_for.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/graph/plane_edge_def.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

using fan_bits_index_t = tf::test::plane_index_t;

template <typename Int> struct fan_bits_real_of;
template <> struct fan_bits_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct fan_bits_real_of<tf::exact::int64> {
  using type = double;
};

template <typename Real>
using fan_bits_mesh_t = tf::polygons_buffer<fan_bits_index_t, Real, 3, 3>;

template <typename Real>
auto fan_bits_translated(fan_bits_mesh_t<Real> mesh, Real dx, Real dy)
    -> fan_bits_mesh_t<Real> {
  for (std::size_t point = 0; point < mesh.points().size(); ++point) {
    mesh.points()[point][0] += dx;
    mesh.points()[point][1] += dy;
  }
  return mesh;
}

/// One mesh out of several, so every contact of the result is same-tag.
template <typename Real>
auto fan_bits_merged(std::vector<fan_bits_mesh_t<Real>> parts)
    -> fan_bits_mesh_t<Real> {
  fan_bits_mesh_t<Real> out;
  fan_bits_index_t base = 0;
  for (auto &part : parts) {
    for (std::size_t point = 0; point < part.points().size(); ++point)
      out.points_buffer().push_back(part.points()[point]);
    for (std::size_t face = 0; face < part.faces().size(); ++face) {
      const auto corners = part.faces()[face];
      out.faces_buffer().push_back(
          {corners[0] + base, corners[1] + base, corners[2] + base});
    }
    base += fan_bits_index_t(part.points().size());
  }
  return out;
}

template <typename Real>
auto push_quad(fan_bits_mesh_t<Real> &mesh, tf::point<Real, 3> a,
               tf::point<Real, 3> b, tf::point<Real, 3> c, tf::point<Real, 3> d)
    -> void {
  const auto base = fan_bits_index_t(mesh.points().size());
  mesh.points_buffer().push_back(a);
  mesh.points_buffer().push_back(b);
  mesh.points_buffer().push_back(c);
  mesh.points_buffer().push_back(d);
  mesh.faces_buffer().push_back({base, base + 1, base + 2});
  mesh.faces_buffer().push_back({base, base + 2, base + 3});
}

/// One canonical group of the plane graph, as its assertions read it.
struct span_t {
  fan_bits_index_t instances = 0;
  fan_bits_index_t fans = 0; // instances born from an intersection record
  fan_bits_index_t radials =
      0; // instances carrying their pair's radial orientation
  fan_bits_index_t with_ordinal = 0;
  fan_bits_index_t tags = 0; // distinct forms stating it
};

/// Everything a scene's assertions read off the graph.
struct graph_facts_t {
  fan_bits_index_t n_planes = 0;
  fan_bits_index_t n_canon = 0;
  fan_bits_index_t n_defs = 0;
  fan_bits_index_t fans = 0;
  fan_bits_index_t duplicate_sides =
      0; // one instance per (face, ordinal), always
  std::vector<span_t> spans;
};

/// The span of the widest group holding instances of `tags` distinct forms.
auto widest_shared(const graph_facts_t &facts, fan_bits_index_t tags)
    -> span_t {
  span_t best;
  for (const auto &span : facts.spans)
    if (span.tags >= tags && span.instances > best.instances)
      best = span;
  return best;
}

template <typename Int, typename Real>
auto fan_bits_measure(std::vector<fan_bits_mesh_t<Real>> meshes)
    -> graph_facts_t {
  std::vector<std::unique_ptr<tf::test::tagged_operand<fan_bits_index_t, Real>>>
      scenes;
  for (auto &mesh : meshes)
    scenes.push_back(
        std::make_unique<tf::test::tagged_operand<fan_bits_index_t, Real>>(
            std::move(mesh)));
  std::vector<decltype(scenes[0]->form())> forms;
  for (auto &scene : scenes)
    forms.push_back(scene->form());
  const auto form_range =
      tf::make_range(forms.data(), forms.data() + forms.size());
  const auto apply_to_face = [&](int tag, fan_bits_index_t object,
                                 const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[object]);
  };
  const auto apply_to_form = [&](fan_bits_index_t tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<fan_bits_index_t> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + fan_bits_index_t(forms[tag].faces().size());

  // a single-form scene needs its own faces intersected to state anything
  const auto mode = forms.size() == 1
                        ? tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours |
                              tf::intersect_mode::within
                        : tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours;
  tf::polygon_intersections<fan_bits_index_t, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(form_range, 0.0);
  intersections.build(form_range, intersections_lattice, tf::intersect_config{mode, 0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_mesh_point = [&](int tag,
                                  fan_bits_index_t id) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[id]);
  };
  tf::intersect::graph::local_arrangement<fan_bits_index_t, Real, Int> world;
  world.build(std::move(intersections), get_mesh_point, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), false, false);

  namespace g2 = tf::intersect::graph;
  const auto &graph = world.graph();
  graph_facts_t facts;
  facts.n_planes = graph.n_planes();
  facts.n_canon = graph.n_canon();
  facts.n_defs = fan_bits_index_t(graph.edge_defs().size());

  std::vector<std::pair<fan_bits_index_t, fan_bits_index_t>> sides;
  for (fan_bits_index_t canon = 0; canon < graph.n_canon(); ++canon) {
    span_t span;
    std::vector<fan_bits_index_t> tags;
    for (const auto &def : graph.canon_group(canon)) {
      const auto &descriptor = graph.descriptors()[std::size_t(def.face)];
      const bool fan = (def.flags & g2::plane_edge_fan_flag) != 0;
      ++span.instances;
      span.fans += fan_bits_index_t(fan);
      span.radials += fan_bits_index_t(g2::plane_edge_radial_sign(def) != 0);
      span.with_ordinal += fan_bits_index_t(def.ordinal >= 0);
      tags.push_back(fan_bits_index_t(descriptor.tag));
      if (fan)
        ++facts.fans;
      if (def.ordinal >= 0)
        sides.push_back({def.face, fan_bits_index_t(def.ordinal)});
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    span.tags = fan_bits_index_t(tags.size());
    facts.spans.push_back(span);
  }
  std::sort(sides.begin(), sides.end());
  for (std::size_t i = 1; i < sides.size(); ++i)
    facts.duplicate_sides += fan_bits_index_t(sides[i] == sides[i - 1]);
  return facts;
}

/// Form 0 a wing pair in y = 0, form 1 a wing pair in z = 0, all of them on
/// the shared original edge (0,0,0)-(8,0,0). `wings` drops one of form 1's.
template <typename Real>
auto fan_bits_exact_edge_merge(int wings)
    -> std::vector<fan_bits_mesh_t<Real>> {
  fan_bits_mesh_t<Real> a;
  a.points_buffer().push_back({Real(0), Real(0), Real(0)});
  a.points_buffer().push_back({Real(8), Real(0), Real(0)});
  a.points_buffer().push_back({Real(4), Real(0), Real(5)});
  a.points_buffer().push_back({Real(4), Real(0), Real(-5)});
  a.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  a.faces_buffer().push_back(
      {fan_bits_index_t(1), fan_bits_index_t(0), fan_bits_index_t(3)});
  fan_bits_mesh_t<Real> b;
  b.points_buffer().push_back({Real(0), Real(0), Real(0)});
  b.points_buffer().push_back({Real(8), Real(0), Real(0)});
  b.points_buffer().push_back({Real(4), Real(5), Real(0)});
  b.points_buffer().push_back({Real(4), Real(-5), Real(0)});
  b.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  if (wings == 2)
    b.faces_buffer().push_back(
        {fan_bits_index_t(1), fan_bits_index_t(0), fan_bits_index_t(3)});
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(std::move(a));
  meshes.push_back(std::move(b));
  return meshes;
}

/// A 4x4 plane tiled 2x2 — tessellation edges at x = 0 and y = 0 — and the
/// four side sheets of a box footprint standing exactly on them, carrying
/// tessellation edge at `zm`. `through` continues the sheet past the plane.
template <typename Real>
auto tiled_plane_and_box(bool through) -> std::vector<fan_bits_mesh_t<Real>> {
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<fan_bits_index_t, Real>(4, 4, 2, 2));
  const Real z0 = through ? Real(-2) : Real(0);
  const Real zm = through ? Real(0) : Real(1);
  const Real xs[5] = {Real(-2), Real(0), Real(0), Real(-2), Real(-2)};
  const Real ys[5] = {Real(-2), Real(-2), Real(0), Real(0), Real(-2)};
  fan_bits_mesh_t<Real> sides;
  for (int side = 0; side < 4; ++side) {
    const auto a = std::size_t(side), b = std::size_t(side + 1);
    push_quad<Real>(sides, {xs[a], ys[a], z0}, {xs[b], ys[b], z0},
                    {xs[b], ys[b], zm}, {xs[a], ys[a], zm});
    push_quad<Real>(sides, {xs[a], ys[a], zm}, {xs[b], ys[b], zm},
                    {xs[b], ys[b], Real(2)}, {xs[a], ys[a], Real(2)});
  }
  meshes.push_back(std::move(sides));
  return meshes;
}

/// A triangle in z = 0 and a standing triangle whose bottom side lies in
/// z = 0 across its interior. `pooled` lays a second face of form 1 flat, so
/// the same contact is asked beside a pooled coplanar plane.
template <typename Real>
auto edge_on_face(bool pooled) -> std::vector<fan_bits_mesh_t<Real>> {
  fan_bits_mesh_t<Real> face;
  face.points_buffer().push_back({Real(0), Real(0), Real(0)});
  face.points_buffer().push_back({Real(8), Real(0), Real(0)});
  face.points_buffer().push_back({Real(0), Real(8), Real(0)});
  face.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  fan_bits_mesh_t<Real> blade;
  blade.points_buffer().push_back({Real(-1), Real(2), Real(0)});
  blade.points_buffer().push_back({Real(9), Real(2), Real(0)});
  blade.points_buffer().push_back({Real(4), Real(2), Real(5)});
  blade.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  if (pooled) {
    blade.points_buffer().push_back({Real(4), Real(6), Real(0)});
    blade.faces_buffer().push_back(
        {fan_bits_index_t(0), fan_bits_index_t(3), fan_bits_index_t(1)});
  }
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(std::move(face));
  meshes.push_back(std::move(blade));
  return meshes;
}

/// Two forms sharing one whole original edge and nothing else.
template <typename Real>
auto fan_bits_bare_touch() -> std::vector<fan_bits_mesh_t<Real>> {
  fan_bits_mesh_t<Real> a;
  a.points_buffer().push_back({Real(0), Real(0), Real(0)});
  a.points_buffer().push_back({Real(8), Real(0), Real(0)});
  a.points_buffer().push_back({Real(0), Real(8), Real(0)});
  a.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  fan_bits_mesh_t<Real> b;
  b.points_buffer().push_back({Real(0), Real(0), Real(0)});
  b.points_buffer().push_back({Real(8), Real(0), Real(0)});
  b.points_buffer().push_back({Real(0), Real(0), Real(8)});
  b.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(2), fan_bits_index_t(1)});
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(std::move(a));
  meshes.push_back(std::move(b));
  return meshes;
}

/// One form folded on a shared original edge, cut by a second form so the
/// fold reaches the arrangement at all.
template <typename Real> auto crease() -> std::vector<fan_bits_mesh_t<Real>> {
  fan_bits_mesh_t<Real> fold;
  fold.points_buffer().push_back({Real(0), Real(0), Real(0)});
  fold.points_buffer().push_back({Real(8), Real(0), Real(0)});
  fold.points_buffer().push_back({Real(0), Real(8), Real(0)});
  fold.points_buffer().push_back({Real(0), Real(0), Real(8)});
  fold.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  fold.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(3), fan_bits_index_t(1)});
  fan_bits_mesh_t<Real> cutter;
  cutter.points_buffer().push_back({Real(4), Real(-5), Real(-5)});
  cutter.points_buffer().push_back({Real(4), Real(10), Real(-5)});
  cutter.points_buffer().push_back({Real(4), Real(-5), Real(10)});
  cutter.faces_buffer().push_back(
      {fan_bits_index_t(0), fan_bits_index_t(1), fan_bits_index_t(2)});
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(std::move(fold));
  meshes.push_back(std::move(cutter));
  return meshes;
}

/// Three coplanar plates in z = 0, staggered so the cover runs 3 / 2 / 1.
/// `one_form` merges them into a single tag.
template <typename Real>
auto coplanar_stack(bool one_form) -> std::vector<fan_bits_mesh_t<Real>> {
  std::vector<fan_bits_mesh_t<Real>> plates;
  for (int plate = 0; plate < 3; ++plate)
    plates.push_back(fan_bits_translated<Real>(
        tf::make_plane_mesh<fan_bits_index_t, Real>(6, 6, 1, 1),
        Real(2 * plate), Real(0)));
  std::vector<fan_bits_mesh_t<Real>> meshes;
  if (one_form) {
    meshes.push_back(fan_bits_merged<Real>(std::move(plates)));
    return meshes;
  }
  for (auto &plate : plates)
    meshes.push_back(std::move(plate));
  return meshes;
}

/// A 2x2-tiled host with a smaller coplanar plate inside it.
template <typename Real>
auto coplanar_tiled() -> std::vector<fan_bits_mesh_t<Real>> {
  std::vector<fan_bits_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<fan_bits_index_t, Real>(6, 6, 2, 2));
  meshes.push_back(tf::make_plane_mesh<fan_bits_index_t, Real>(2, 2, 1, 1));
  return meshes;
}

} // namespace

TEMPLATE_TEST_CASE("plane graph fan bits: the exact-edge merge states its "
                   "intersection on every face's own side",
                   "[cut][planes][fan-bits]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename fan_bits_real_of<Int>::type;

  // Four faces, one canonical edge, and NO interior cut anywhere: the whole
  // contact is each face's own original side. Every instance is nevertheless
  // record-born, so every instance carries the fan flag and its pair's
  // radial orientation — and keeps the ordinal and side the loop would have
  // given it.
  const auto four =
      fan_bits_measure<Int, Real>(fan_bits_exact_edge_merge<Real>(2));
  const auto shared = widest_shared(four, 2);
  CHECK(shared.instances == 4);
  CHECK(shared.fans == 4);
  CHECK(shared.with_ordinal == 4);
  CHECK(four.duplicate_sides == 0);

  // The three-face variant states the same thing one wing short.
  const auto three =
      fan_bits_measure<Int, Real>(fan_bits_exact_edge_merge<Real>(1));
  const auto three_shared = widest_shared(three, 2);
  CHECK(three_shared.instances == 3);
  CHECK(three_shared.fans == 3);
  CHECK(three.duplicate_sides == 0);
}

TEMPLATE_TEST_CASE("plane graph fan bits: a box sheet landing on a tiled "
                   "plane's edges marks them",
                   "[cut][planes][fan-bits]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename fan_bits_real_of<Int>::type;

  // The box's own tessellation edge lies exactly ON the plane's edge, so the
  // contact is a side of BOTH forms and no interior cut is minted for it.
  const auto through =
      fan_bits_measure<Int, Real>(tiled_plane_and_box<Real>(true));
  const auto crossed = widest_shared(through, 2);
  CHECK(crossed.instances == 4);
  CHECK(crossed.fans == 4);
  CHECK(crossed.with_ordinal == 4);
  CHECK(through.fans > 0);
  CHECK(through.duplicate_sides == 0);

  // The same rim, resting instead of continuing: a record is a record.
  const auto resting =
      fan_bits_measure<Int, Real>(tiled_plane_and_box<Real>(false));
  const auto rested = widest_shared(resting, 2);
  CHECK(rested.instances >= 2);
  CHECK(rested.fans == rested.instances);
  CHECK(resting.duplicate_sides == 0);
}

TEMPLATE_TEST_CASE("plane graph fan bits: an edge on a face marks the cut "
                   "and the blade's own side alike",
                   "[cut][planes][fan-bits]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename fan_bits_real_of<Int>::type;

  // One span, two instances: the cut face's interior cut (no ordinal) and the
  // blade's own side (its ordinal), both intersection edges.
  const auto plain = fan_bits_measure<Int, Real>(edge_on_face<Real>(false));
  const auto contact = widest_shared(plain, 2);
  CHECK(contact.instances == 2);
  CHECK(contact.fans == 2);
  CHECK(contact.with_ordinal == 1);
  CHECK(plain.duplicate_sides == 0);

  // The same contact with form 1's second face laid flat: that face is
  // coplanar with form 0's, so it POOLS instead of emitting and its own side
  // joins the same span unmarked. One span, both laws: the cross-plane
  // contact marked, the within-plane member's side not.
  const auto pooled = fan_bits_measure<Int, Real>(edge_on_face<Real>(true));
  const auto pooled_contact = widest_shared(pooled, 2);
  CHECK(pooled_contact.instances == 3);
  CHECK(pooled_contact.fans == 2);
  CHECK(pooled_contact.with_ordinal == 2);
  CHECK(pooled.duplicate_sides == 0);
}

TEMPLATE_TEST_CASE("plane graph fan bits: a bare touch is marked and a "
                   "crease is not",
                   "[cut][planes][fan-bits]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename fan_bits_real_of<Int>::type;

  // Two forms meeting along one shared edge without crossing. Records exist,
  // so the span is marked: extraction states what the records state, and no
  // touch-versus-crossing verdict is taken here.
  const auto touch = fan_bits_measure<Int, Real>(fan_bits_bare_touch<Real>());
  const auto shared = widest_shared(touch, 2);
  CHECK(shared.instances == 2);
  CHECK(shared.fans == 2);
  CHECK(shared.with_ordinal == 2);
  CHECK(touch.duplicate_sides == 0);

  // The fold of ONE form: its two faces state the edge as their own side and
  // no record names it, so nothing marks it. The cutter's chords elsewhere in
  // the same scene ARE marked, which is what makes this a red-flippable test.
  const auto fold = fan_bits_measure<Int, Real>(crease<Real>());
  fan_bits_index_t self_spans = 0, marked_self_spans = 0;
  for (const auto &span : fold.spans)
    if (span.tags == 1 && span.instances == 2 && span.with_ordinal == 2) {
      ++self_spans;
      marked_self_spans += fan_bits_index_t(span.fans != 0);
    }
  CHECK(self_spans > 0);
  CHECK(marked_self_spans == 0);
  CHECK(fold.fans > 0);
  CHECK(fold.duplicate_sides == 0);
}

TEMPLATE_TEST_CASE("plane graph fan bits: coplanar contacts pool and mark "
                   "nothing",
                   "[cut][planes][fan-bits]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename fan_bits_real_of<Int>::type;

  // A coplanar pair pools its faces into one plane instead of emitting, so
  // the whole in-plane fabric — every stacked plate's every side — stays
  // free of marks. The stack's own reduction is not extraction's business.
  const auto staggered =
      fan_bits_measure<Int, Real>(coplanar_stack<Real>(false));
  CHECK(staggered.n_planes == 1);
  CHECK(staggered.fans == 0);
  CHECK(staggered.duplicate_sides == 0);

  const auto single_tag =
      fan_bits_measure<Int, Real>(coplanar_stack<Real>(true));
  CHECK(single_tag.n_planes == 1);
  CHECK(single_tag.fans == 0);
  CHECK(single_tag.duplicate_sides == 0);

  const auto tiled = fan_bits_measure<Int, Real>(coplanar_tiled<Real>());
  CHECK(tiled.n_planes == 1);
  CHECK(tiled.fans == 0);
  CHECK(tiled.duplicate_sides == 0);
}
