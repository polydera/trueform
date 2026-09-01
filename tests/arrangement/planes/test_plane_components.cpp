/**
 * @file test_plane_components.cpp
 * @brief Tests for the plane arrangement's component tier
 *
 * The carrier is the CELL, and a piece fences for exactly three reasons:
 * FAN — more than two live cell incidences meet at the piece; NON-MANIFOLD —
 * one original mesh edge carries more than two faces; DEPTH BORDER — the
 * number of sheets covering the ground changes across the piece. Nothing else
 * fences, and equal depth is ground continuing. Each scene states one producer
 * and the counterfactual that makes it visible:
 *
 *   TILED PAIR      two overlapping tiled plates — only the depth border
 *                   fences, and refining either tessellation moves nothing;
 *   TILED IN TILED  a fine insert inside a coarse host — the border fences
 *                   and the insert's interior tessellation does not;
 *   INTERIOR CUT    two plates crossing inside ONE form, where the depth is
 *                   one everywhere and only the crossing's four sheets fence;
 *   FAN JUNCTION    four pages of one form on one edge — four sheets on one
 *                   line is a fan whether the form states the coincidence
 *                   itself or a second form's cut brings the pages in;
 *   SHARED SPINE    four coplanar plates on ONE mesh edge — non-manifold by
 *                   face count alone — beside the same plates welded merely
 *                   coincident, where the ground continues.
 *
 * The pooling family states the same three producers where several plates
 * share ONE geometric plane:
 *
 *   OVERLAPPING PLATES  a coarse host, a tiled insert inside one of its
 *                       tiles, and a third plate overlapping both — one
 *                       plane however the plates are tagged, and coverage
 *                       is the only thing that fences in it;
 *   DUPLICATE STACK     k exact copies, whose members share their cells, so
 *                       no depth of stack fences and no diagonal of it does;
 *   STANDING PLATE      a plate standing on a host face, where the slit's own
 *                       cell reaches the seam TWICE and the count is
 *                       occurrences;
 *   CROSSED PAGES       one mesh's pages on one edge, single and doubled —
 *                       the form's own edge count fences what two live
 *                       incidences and equal depth cannot see;
 *   OPPOSED WINDING     the same overlap wound the other way, where the fence
 *                       tier reads coverage and not orientation;
 *   BARE TOUCH          two forms on one shared edge — stated, and fencing at
 *                       incidence two.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"
#include "input_lattice_for.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/graph/label_plane_arrangement_components.hpp>
#include <trueform/arrangement/planes/make_plane_arrangement_cells.hpp>
#include <trueform/arrangement/planes/make_plane_piece_fences.hpp>
#include <trueform/arrangement/planes/make_plane_piece_incidence.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

using components_index_t = tf::test::plane_index_t;

/// The real type each lattice width is quantized from.
template <typename Int> struct components_real_of;
template <> struct components_real_of<tf::exact::int32> {
  using type = float;
};
template <> struct components_real_of<tf::exact::int64> {
  using type = double;
};

template <typename Real>
using components_mesh_t = tf::polygons_buffer<components_index_t, Real, 3, 3>;

template <typename Real>
auto components_translated(components_mesh_t<Real> mesh, Real dx, Real dy)
    -> components_mesh_t<Real> {
  for (std::size_t point = 0; point < mesh.points().size(); ++point) {
    mesh.points()[point][0] += dx;
    mesh.points()[point][1] += dy;
  }
  return mesh;
}

/// One mesh out of several, so every cut of the result is a same-tag cut.
template <typename Real>
auto components_merged(std::vector<components_mesh_t<Real>> parts)
    -> components_mesh_t<Real> {
  components_mesh_t<Real> out;
  components_index_t base = 0;
  for (auto &part : parts) {
    for (std::size_t point = 0; point < part.points().size(); ++point)
      out.points_buffer().push_back(part.points()[point]);
    for (std::size_t face = 0; face < part.faces().size(); ++face) {
      const auto corners = part.faces()[face];
      out.faces_buffer().push_back(
          {corners[0] + base, corners[1] + base, corners[2] + base});
    }
    base += components_index_t(part.points().size());
  }
  return out;
}

template <typename Real>
auto stood_up(components_mesh_t<Real> mesh) -> components_mesh_t<Real> {
  for (std::size_t point = 0; point < mesh.points().size(); ++point)
    std::swap(mesh.points()[point][1], mesh.points()[point][2]);
  return mesh;
}

/// Everything a scene's assertions read: the partition under the law, the
/// partitions under the two counterfactuals, and the sizes of the arrangement
/// the instrument ran on.
struct components_t {
  components_index_t n_components = 0;
  components_index_t unfenced = 0; // the fence switched off entirely
  components_index_t fans_only =
      0; // the stated bit alone, quiet fences switched off
  components_index_t n_cells = 0;
  components_index_t n_fans = 0;
  components_index_t n_quiet =
      0; // fenced with no fan: non-manifold or depth border
  components_index_t n_live = 0;
  // The count the fan clause makes and the count it refuses to make: live
  // OCCURRENCES, where a cell reaching the piece from both sides of a slit
  // states two, against the distinct cells those occurrences come from.
  components_index_t n_crowded = 0;
  components_index_t n_crowded_cells = 0;
  std::vector<components_index_t>
      sizes; // live triangles per component, ascending
};

/// The partition, as sizes: labels are storage, the partition is the fact.
template <typename Labels, typename Live>
auto partition_sizes(const Labels &labels, const Live &live,
                     components_index_t extent)
    -> std::vector<components_index_t> {
  std::vector<components_index_t> counts(std::size_t(extent),
                                         components_index_t(0));
  for (std::size_t t = 0; t < labels.size(); ++t)
    if (live[t] != char(0))
      ++counts[std::size_t(labels[t])];
  counts.erase(std::remove(counts.begin(), counts.end(), components_index_t(0)),
               counts.end());
  std::sort(counts.begin(), counts.end());
  return counts;
}

template <typename Int, typename Real>
auto components_measure(std::vector<components_mesh_t<Real>> meshes,
                        bool within) -> components_t {
  std::vector<
      std::unique_ptr<tf::test::tagged_operand<components_index_t, Real>>>
      scenes;
  for (auto &mesh : meshes)
    scenes.push_back(
        std::make_unique<tf::test::tagged_operand<components_index_t, Real>>(
            std::move(mesh)));
  std::vector<decltype(scenes[0]->form())> forms;
  for (auto &scene : scenes)
    forms.push_back(scene->form());
  const auto form_range =
      tf::make_range(forms.data(), forms.data() + forms.size());
  const auto apply_to_face = [&](int tag, components_index_t object,
                                 const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[object]);
  };
  const auto apply_to_form = [&](components_index_t tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<components_index_t> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + components_index_t(forms[tag].faces().size());

  const auto mode = within ? tf::intersect_mode::primitives |
                                 tf::intersect_mode::resolve_crossing_contours |
                                 tf::intersect_mode::within
                           : tf::intersect_mode::primitives |
                                 tf::intersect_mode::resolve_crossing_contours;
  tf::polygon_intersections<components_index_t, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(form_range, 0.0);
  intersections.build(form_range, intersections_lattice, tf::intersect_config{mode, 0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_mesh_point = [&](int tag,
                                  components_index_t id) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[id]);
  };

  tf::arrangement::plane_arrangement<components_index_t, Int> arrangement;
  arrangement.record_triangle_cells();
  arrangement.record_triangle_arrangement();
  tf::intersect::graph::local_arrangement<components_index_t, Real, Int> world;
  world.build(std::move(intersections), get_mesh_point, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), false, false);
  arrangement.build(world, get_mesh_point);

  components_t out;
  REQUIRE(arrangement.failed().size() == 0);
  REQUIRE(arrangement.triangle_cells().size() ==
          arrangement.triangles().size());

  const auto cells = tf::arrangement::make_plane_arrangement_cells(arrangement);
  const auto incidence =
      tf::arrangement::make_plane_piece_incidence(arrangement, cells);
  const auto fences = tf::arrangement::make_plane_piece_fences(
      arrangement, world, incidence, cells);

  const auto coplanar_of = arrangement.coplanar_of();
  tf::buffer<char> live;
  live.allocate(arrangement.triangles().size());
  for (std::size_t t = 0; t < live.size(); ++t) {
    live[t] = char(coplanar_of[t] == components_index_t(-1));
    out.n_live += components_index_t(live[t] != char(0));
  }

  // Two counterfactuals over the same product: no fence at all, and the
  // stated bit alone (the quiet fences switched off).
  tf::buffer<char> no_fence;
  tf::buffer<char> fans_only;
  no_fence.allocate(fences.fan.size());
  fans_only.allocate(fences.fan.size());
  for (std::size_t piece = 0; piece < fences.fan.size(); ++piece) {
    no_fence[piece] = char(1);
    fans_only[piece] = char(fences.fan[piece] == char(0));
  }

  const auto labels = tf::csg::graph::label_plane_arrangement_components(
      incidence, fences.crossable, cells);
  const auto open = tf::csg::graph::label_plane_arrangement_components(
      incidence, no_fence, cells);
  const auto by_fans = tf::csg::graph::label_plane_arrangement_components(
      incidence, fans_only, cells);

  out.sizes = partition_sizes(labels.labels, live, labels.n_components);
  out.n_components = components_index_t(out.sizes.size());
  out.unfenced = components_index_t(
      partition_sizes(open.labels, live, open.n_components).size());
  out.fans_only = components_index_t(
      partition_sizes(by_fans.labels, live, by_fans.n_components).size());
  out.n_cells = cells.n_cells;
  std::vector<components_index_t> piece_cells;
  for (std::size_t piece = 0; piece < fences.fan.size(); ++piece) {
    out.n_fans += components_index_t(fences.fan[piece] != char(0));
    out.n_quiet += components_index_t(fences.fan[piece] == char(0) &&
                                      fences.crossable[piece] == char(0));
    piece_cells.clear();
    for (const auto row : incidence.rows_of_piece[piece]) {
      const auto triangle = row / components_index_t(3);
      if (coplanar_of[std::size_t(triangle)] == components_index_t(-1))
        piece_cells.push_back(cells.cell_of[std::size_t(triangle)]);
    }
    out.n_crowded += components_index_t(piece_cells.size() > std::size_t(2));
    std::sort(piece_cells.begin(), piece_cells.end());
    out.n_crowded_cells += components_index_t(
        std::size_t(std::unique(piece_cells.begin(), piece_cells.end()) -
                    piece_cells.begin()) > std::size_t(2));
  }
  return out;
}

/// Arms the law says are ONE arrangement, checked field by field so a failure
/// names the arm and the field that moved.
auto check_same_arrangement(const char *arm, const components_t &measured,
                            const components_t &reference) -> void {
  INFO(arm);
  CHECK(measured.n_components == reference.n_components);
  CHECK(measured.sizes == reference.sizes);
  CHECK(measured.n_cells == reference.n_cells);
  CHECK(measured.n_live == reference.n_live);
  CHECK(measured.n_fans == reference.n_fans);
  CHECK(measured.n_quiet == reference.n_quiet);
  CHECK(measured.fans_only == reference.fans_only);
}

/// Two coplanar plates, the second offset so each owns an interior cut of the
/// other. `ticks` states how finely both are tiled.
template <typename Real>
auto tiled_pair(components_index_t ticks)
    -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(
      tf::make_plane_mesh<components_index_t, Real>(6, 6, ticks, ticks));
  meshes.push_back(components_translated<Real>(
      tf::make_plane_mesh<components_index_t, Real>(4, 4, ticks, ticks),
      Real(3), Real(1)));
  return meshes;
}

/// Four plates of ONE form sharing one edge: the form's own non-manifold
/// junction, with no interior cut and one sheet on every cell.
template <typename Real>
auto same_form_fan() -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> parts;
  for (int page = 0; page < 4; ++page) {
    components_mesh_t<Real> plate;
    const Real dy = page == 0 ? Real(1) : (page == 2 ? Real(-1) : Real(0));
    const Real dz = page == 1 ? Real(1) : (page == 3 ? Real(-1) : Real(0));
    plate.points_buffer().push_back({Real(0), Real(0), Real(0)});
    plate.points_buffer().push_back({Real(3), Real(0), Real(0)});
    plate.points_buffer().push_back({Real(3), Real(3) * dy, Real(3) * dz});
    plate.points_buffer().push_back({Real(0), Real(3) * dy, Real(3) * dz});
    plate.faces_buffer().push_back(
        {components_index_t(0), components_index_t(1), components_index_t(2)});
    plate.faces_buffer().push_back(
        {components_index_t(0), components_index_t(2), components_index_t(3)});
    parts.push_back(std::move(plate));
  }
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(components_merged<Real>(std::move(parts)));
  return meshes;
}

/// A tiled insert strictly inside one tile of a coarse host.
template <typename Real>
auto tiled_in_tiled(components_index_t ticks)
    -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<components_index_t, Real>(6, 6, 3, 3));
  meshes.push_back(
      tf::make_plane_mesh<components_index_t, Real>(2, 2, ticks, ticks));
  return meshes;
}

/// Four coplanar plates of ONE form on one shared spine edge — the same two
/// vertices in every plate — two ending at it and two beginning: one mesh
/// edge, four faces, equal depth on both sides.
template <typename Real>
auto shared_spine() -> std::vector<components_mesh_t<Real>> {
  components_mesh_t<Real> out;
  out.points_buffer().push_back({Real(0), Real(-1), Real(0)});
  out.points_buffer().push_back({Real(0), Real(1), Real(0)});
  for (int side = 0; side < 2; ++side)
    for (int copy = 0; copy < 2; ++copy) {
      const Real x = side == 0 ? Real(-2) : Real(2);
      const auto base = components_index_t(out.points().size());
      out.points_buffer().push_back({x, Real(-1), Real(0)});
      out.points_buffer().push_back({x, Real(1), Real(0)});
      out.faces_buffer().push_back(
          {components_index_t(0), components_index_t(1), base + 1});
      out.faces_buffer().push_back({components_index_t(0), base + 1, base});
    }
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(std::move(out));
  return meshes;
}

/// The same plates as ONE mesh, so every contact between them is same-tag.
template <typename Real>
auto as_one_mesh(std::vector<components_mesh_t<Real>> parts)
    -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(components_merged<Real>(std::move(parts)));
  return meshes;
}

/// A coarse host, a tiled insert strictly inside one of its tiles, and a
/// third plate covering one corner of the insert and host ground elsewhere:
/// three mutually overlapping plates of one geometric plane. `ticks` states
/// how finely the insert is tiled.
template <typename Real>
auto overlapping_plates(components_index_t ticks)
    -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<components_index_t, Real>(6, 6, 3, 3));
  meshes.push_back(
      tf::make_plane_mesh<components_index_t, Real>(1, 1, ticks, ticks));
  meshes.push_back(components_translated<Real>(
      tf::make_plane_mesh<components_index_t, Real>(1, 1, 1, 1), Real(0.25),
      Real(0.25)));
  return meshes;
}

/// `copies` exact duplicates of one plate.
template <typename Real>
auto duplicate_stack(int copies) -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  for (int copy = 0; copy < copies; ++copy)
    meshes.push_back(tf::make_plane_mesh<components_index_t, Real>(2, 2, 1, 1));
  return meshes;
}

/// A host face and a plate standing on it, the plate's base edge strictly
/// inside the host: the seam is a slit that dead-ends in the host's own cell
/// and a whole boundary side of the plate.
template <typename Real>
auto standing_plate() -> std::vector<components_mesh_t<Real>> {
  components_mesh_t<Real> host;
  host.points_buffer().push_back({Real(-4), Real(-4), Real(0)});
  host.points_buffer().push_back({Real(4), Real(-4), Real(0)});
  host.points_buffer().push_back({Real(0), Real(4), Real(0)});
  host.faces_buffer().push_back(
      {components_index_t(0), components_index_t(1), components_index_t(2)});
  components_mesh_t<Real> standing;
  standing.points_buffer().push_back({Real(-1), Real(0), Real(0)});
  standing.points_buffer().push_back({Real(1), Real(0), Real(0)});
  standing.points_buffer().push_back({Real(0), Real(0), Real(2)});
  standing.faces_buffer().push_back(
      {components_index_t(0), components_index_t(1), components_index_t(2)});
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(std::move(host));
  meshes.push_back(std::move(standing));
  return meshes;
}

/// ONE mesh whose pages meet on the edge (0,0,0)-(6,0,0), one page in z = 0
/// and one in y = 0, beside the plate that cuts both into the arrangement —
/// it lies in y + z = 1, so it crosses each page and misses their edge.
/// `doubled` gives each plane a second, larger page on that same edge, so the
/// pages stack where they reach it and the mesh edge carries four faces.
template <typename Real>
auto crossed_pages(bool doubled) -> std::vector<components_mesh_t<Real>> {
  components_mesh_t<Real> pages;
  pages.points_buffer().push_back({Real(0), Real(0), Real(0)});
  pages.points_buffer().push_back({Real(6), Real(0), Real(0)});
  pages.points_buffer().push_back({Real(3), Real(2), Real(0)});
  pages.points_buffer().push_back({Real(3), Real(0), Real(2)});
  pages.faces_buffer().push_back(
      {components_index_t(0), components_index_t(1), components_index_t(2)});
  pages.faces_buffer().push_back(
      {components_index_t(0), components_index_t(3), components_index_t(1)});
  if (doubled) {
    pages.points_buffer().push_back({Real(3), Real(4), Real(0)});
    pages.points_buffer().push_back({Real(3), Real(0), Real(4)});
    pages.faces_buffer().push_back(
        {components_index_t(0), components_index_t(1), components_index_t(4)});
    pages.faces_buffer().push_back(
        {components_index_t(0), components_index_t(5), components_index_t(1)});
  }
  components_mesh_t<Real> cutter;
  cutter.points_buffer().push_back({Real(-4), Real(6), Real(-5)});
  cutter.points_buffer().push_back({Real(10), Real(6), Real(-5)});
  cutter.points_buffer().push_back({Real(3), Real(-8), Real(9)});
  cutter.faces_buffer().push_back(
      {components_index_t(0), components_index_t(1), components_index_t(2)});
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(std::move(pages));
  meshes.push_back(std::move(cutter));
  return meshes;
}

/// Two coplanar plates overlapping in one quarter; `reversed` winds the
/// second one the other way.
template <typename Real>
auto quarter_overlap(bool reversed) -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(tf::make_plane_mesh<components_index_t, Real>(2, 2, 1, 1));
  auto other = components_translated<Real>(
      tf::make_plane_mesh<components_index_t, Real>(2, 2, 1, 1), Real(1),
      Real(1));
  if (reversed)
    for (std::size_t face = 0; face < other.faces().size(); ++face)
      std::swap(other.faces()[face][1], other.faces()[face][2]);
  meshes.push_back(std::move(other));
  return meshes;
}

/// Two forms meeting on one shared edge and nowhere else.
template <typename Real>
auto components_bare_touch() -> std::vector<components_mesh_t<Real>> {
  components_mesh_t<Real> plate;
  plate.points_buffer().push_back({Real(0), Real(0), Real(0)});
  plate.points_buffer().push_back({Real(8), Real(0), Real(0)});
  plate.points_buffer().push_back({Real(0), Real(8), Real(0)});
  plate.faces_buffer().push_back(
      {components_index_t(0), components_index_t(1), components_index_t(2)});
  components_mesh_t<Real> standing;
  standing.points_buffer().push_back({Real(0), Real(0), Real(0)});
  standing.points_buffer().push_back({Real(8), Real(0), Real(0)});
  standing.points_buffer().push_back({Real(0), Real(0), Real(8)});
  standing.faces_buffer().push_back(
      {components_index_t(0), components_index_t(2), components_index_t(1)});
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(std::move(plate));
  meshes.push_back(std::move(standing));
  return meshes;
}

/// The same four plates without the shared vertices: coincident boundaries,
/// each plate its own points.
template <typename Real>
auto balanced_copies() -> std::vector<components_mesh_t<Real>> {
  std::vector<components_mesh_t<Real>> parts;
  for (int side = 0; side < 2; ++side)
    for (int copy = 0; copy < 2; ++copy)
      parts.push_back(components_translated<Real>(
          tf::make_plane_mesh<components_index_t, Real>(2, 2, 1, 1),
          side == 0 ? Real(-1) : Real(1), Real(0)));
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(components_merged<Real>(std::move(parts)));
  return meshes;
}

} // namespace

TEMPLATE_TEST_CASE("plane components: the tiled pair fences its mutual "
                   "boundary and nothing else",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // Three sheets of surface: what only the host covers, what both cover, and
  // what only the insert covers. The overlap is bordered by depth changes and must be
  // its own component; the two single-sheet regions are one each.
  const auto coarse = components_measure<Int, Real>(tiled_pair<Real>(2), false);
  CHECK(coarse.n_components == 3);
  CHECK(coarse.unfenced == 1);
  // Purely coplanar contact states no records, so no bit and no fan stands
  // anywhere: the rim fences quietly, and the bit alone fences nothing.
  CHECK(coarse.n_fans == 0);
  CHECK(coarse.n_quiet > 0);
  CHECK(coarse.fans_only == 1);

  // The same three sheets under a finer tessellation of both plates. The cells
  // multiply and the partition does not: a tessellation edge carries the same
  // sheets on both sides, so nothing fences it.
  const auto fine = components_measure<Int, Real>(tiled_pair<Real>(6), false);
  CHECK(fine.n_components == 3);
  CHECK(fine.unfenced == 1);
  CHECK(fine.n_cells > coarse.n_cells);
  CHECK(fine.n_live > coarse.n_live);
}

TEMPLATE_TEST_CASE("plane components: a tiled insert borders on its rim, not on "
                   "its tessellation",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  const auto coarse =
      components_measure<Int, Real>(tiled_in_tiled<Real>(1), false);
  CHECK(coarse.n_components == 2);
  CHECK(coarse.sizes == std::vector<components_index_t>{2, 14});
  CHECK(coarse.unfenced == 1);
  CHECK(coarse.n_fans == 0);
  CHECK(coarse.n_quiet > 0);
  CHECK(coarse.fans_only == 1);

  // A 16-times finer insert inside the same host tile: 4 rim sides' worth of
  // borders become more pieces, but the pad stays ONE component and the ring
  // stays one.
  const auto fine =
      components_measure<Int, Real>(tiled_in_tiled<Real>(4), false);
  CHECK(fine.n_components == 2);
  CHECK(fine.unfenced == 1);
  CHECK(fine.n_cells > coarse.n_cells);
  CHECK(fine.n_live > coarse.n_live);
  // The pad is the covered part of the host and the ring is the rest; refining
  // the insert changes neither.
  CHECK(fine.sizes.size() == 2);
  CHECK(fine.sizes[0] > coarse.sizes[0]);

}

TEMPLATE_TEST_CASE("plane components: an interior cut fences a same-tag "
                   "crossing",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // ONE form holding two crossing plates: every cell carries the same single
  // sheet, so the depth clause has nothing to say anywhere and the four halves
  // stay apart on the crossing alone — four cells meet on each cut span.
  std::vector<components_mesh_t<Real>> parts;
  parts.push_back(tf::make_plane_mesh<components_index_t, Real>(6, 6, 1, 1));
  parts.push_back(stood_up<Real>(
      tf::make_plane_mesh<components_index_t, Real>(6, 6, 1, 1)));
  std::vector<components_mesh_t<Real>> meshes;
  meshes.push_back(components_merged<Real>(std::move(parts)));

  const auto crossing = components_measure<Int, Real>(std::move(meshes), true);
  CHECK(crossing.n_components == 4);
  CHECK(crossing.sizes == std::vector<components_index_t>{3, 3, 3, 3});
  CHECK(crossing.n_fans == 2);
  CHECK(crossing.n_quiet == 0);
  CHECK(crossing.unfenced == 1);
}

TEMPLATE_TEST_CASE("plane components: a stated junction is a fan, an "
                   "unstated one never welds",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // Four plates of one form on one shared, stated edge: planes meet there and
  // domains are bound by pairing radially around it, so the merged edge is a
  // FAN — four live incidences on one piece — and the fan alone keeps the
  // pages apart.
  const auto fan = components_measure<Int, Real>(same_form_fan<Real>(), true);
  CHECK(fan.n_components == 4);
  CHECK(fan.sizes == std::vector<components_index_t>{2, 2, 2, 2});
  CHECK(fan.n_fans == 1);
  CHECK(fan.n_quiet == 0);
  CHECK(fan.fans_only == 4);
  CHECK(fan.unfenced == 1);

  // The same four pages with no within mode: the form states no records about
  // itself, so a second form's plate at x = 1.5 is what brings the pages into
  // the arrangement. SIX pieces carry more than two live incidences — the four
  // cut spans, each bordering its page's cell on both sides of the slit plus
  // the plate's two cells, and the two spine pieces the two coplanar page
  // pairs pool into one identity, each bordering all four page cells. Every
  // page is one component (its slit dead-ends inside it) and the plate's four
  // quadrants are four more.
  auto meshes = same_form_fan<Real>();
  auto plate = tf::make_plane_mesh<components_index_t, Real>(2, 2, 1, 1);
  for (std::size_t point = 0; point < plate.points().size(); ++point) {
    const auto x = plate.points()[point][0];
    plate.points()[point][0] = Real(1.5);
    plate.points()[point][2] = x;
  }
  meshes.push_back(std::move(plate));
  const auto quiet = components_measure<Int, Real>(std::move(meshes), false);
  CHECK(quiet.n_components == 8);
  // Žiga ruled this number twice, and both rulings are the same law read
  // at two states of the identity.
  //
  // While a same-tag plane was a CONNECTED one -- the exact-only
  // interregnum, taken for cost until a plane lattice existed -- pooling
  // was a connectivity fact, and these coplanar page pairs share no
  // vertex identity, so quiet mode did not unite them: "six to four is
  // right because the within flag exists" (2026-08-26).
  //
  // Identity is now the NAME the kernel states, which is exactly the
  // "until that lattice exists" the placeholder promised, and a name
  // knows nothing of connectivity: SAME CELL, SAME FEATURE, in quiet
  // mode too. So the two coplanar page pairs reunite without the within
  // flag and the two spine pieces are fans again -- "yes, they reunite,
  // the test requirement changes, nothing surprising" (2026-08-26).
  CHECK(quiet.n_fans == 6);
  CHECK(quiet.n_quiet == 0);
  CHECK(quiet.fans_only == 8);
  CHECK(quiet.unfenced == 1);
}

TEMPLATE_TEST_CASE("plane components: one shared spine fences, equal depth "
                   "continues",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // ONE mesh edge carrying four faces: the form's own non-manifold edge. The
  // four coplanar plates pool into two cells, so only TWO live incidences meet
  // there and no fan stands; the depth reads two on both sides — the edge's
  // own face count is the only fence.
  const auto spine = components_measure<Int, Real>(shared_spine<Real>(), true);
  CHECK(spine.n_components == 2);
  CHECK(spine.sizes == std::vector<components_index_t>{2, 2});
  CHECK(spine.n_fans == 0);
  CHECK(spine.n_quiet == 1);
  // RED FIRST: fans alone fence nothing here.
  CHECK(spine.fans_only == 1);
  CHECK(spine.unfenced == 1);

  // The same four plates welded merely coincident: every mesh edge keeps two
  // faces or one, the depth reads two everywhere, and the ground continues.
  const auto continues =
      components_measure<Int, Real>(balanced_copies<Real>(), true);
  CHECK(continues.n_components == 1);
  CHECK(continues.n_fans == 0);
  CHECK(continues.n_quiet == 0);
  CHECK(continues.unfenced == 1);
}

TEMPLATE_TEST_CASE("plane components: mutually overlapping plates are one "
                   "plane, however they are tagged",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // A coarse host, a tiled insert strictly inside one of its tiles, and a
  // third plate covering one corner of the insert and host ground beyond it.
  // One geometric plane is one plane, so all three pool into a single
  // triangulation and coverage is the only thing that varies across it: the
  // host alone, the insert without the plate, the plate without the insert,
  // and the corner all three cover — FOUR depth classes, each connected, each
  // bordered by depth changes. Purely coplanar contact states no seam, and
  // inside one plane an edge carries one live triangle per side, so no piece
  // can be crowded and no fan stands anywhere.
  const auto three =
      components_measure<Int, Real>(overlapping_plates<Real>(2), false);
  CHECK(three.n_components == 4);
  CHECK(three.n_fans == 0);
  CHECK(three.n_crowded == 0);
  CHECK(three.n_quiet > 0);
  CHECK(three.fans_only == 1);
  CHECK(three.unfenced == 1);

  // The same three plates as ONE mesh, and either way with the within flag
  // on. Same-tag coplanar faces pool whatever the flag says, and in-plane
  // structure resolves always: the flag governs transversal self-structure,
  // which a flat pack has none of. A lone form's own records are implied by
  // its arity — there is nothing else for its build to state — so the
  // one-mesh arms answer with the flag off too, and every arm is the same
  // arrangement down to the partition sizes.
  check_same_arrangement(
      "three tags, within on",
      components_measure<Int, Real>(overlapping_plates<Real>(2), true), three);
  check_same_arrangement("one tag, within off",
                         components_measure<Int, Real>(
                             as_one_mesh(overlapping_plates<Real>(2)), false),
                         three);
  check_same_arrangement("one tag, within on",
                         components_measure<Int, Real>(
                             as_one_mesh(overlapping_plates<Real>(2)), true),
                         three);

  // The insert's own tessellation is not a border: refine it and the cells
  // and the triangles multiply while the four depth classes stand.
  const auto fine =
      components_measure<Int, Real>(overlapping_plates<Real>(4), false);
  CHECK(fine.n_components == 4);
  CHECK(fine.n_fans == 0);
  CHECK(fine.n_cells > three.n_cells);
  CHECK(fine.n_live > three.n_live);
  const auto plain =
      components_measure<Int, Real>(overlapping_plates<Real>(1), false);
  CHECK(plain.n_components == 4);
  CHECK(plain.n_cells < three.n_cells);
}

TEMPLATE_TEST_CASE("plane components: a stack of duplicates is one ground",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // Exact copies of one plate pool into ONE triangulation, so its two regions
  // are the plane's two cells; each elects one survivor and the other copies
  // emit duplicates carrying that survivor's cell. The diagonal the copies
  // share therefore has k rows on each side and exactly TWO live ones — dead
  // members add no live incidence — so no count fences it and the equal depth
  // crosses it. The rim's k rows all name one cell, so no depth border stands
  // there either: two live triangles, one component, whatever k is.
  const auto pair =
      components_measure<Int, Real>(duplicate_stack<Real>(2), false);
  CHECK(pair.n_components == 1);
  CHECK(pair.sizes == std::vector<components_index_t>{2});
  CHECK(pair.n_cells == 2);
  CHECK(pair.n_live == 2);
  CHECK(pair.n_fans == 0);
  CHECK(pair.n_crowded == 0);
  CHECK(pair.n_quiet == 0);
  CHECK(pair.unfenced == 1);

  check_same_arrangement(
      "three copies",
      components_measure<Int, Real>(duplicate_stack<Real>(3), false), pair);
  check_same_arrangement("two copies, one tag",
                         components_measure<Int, Real>(
                             as_one_mesh(duplicate_stack<Real>(2)), true),
                         pair);
  check_same_arrangement("three copies, one tag",
                         components_measure<Int, Real>(
                             as_one_mesh(duplicate_stack<Real>(3)), true),
                         pair);
}

TEMPLATE_TEST_CASE("plane components: a slit's own cell reaches its seam "
                   "twice",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // A plate standing ON a host face, its base edge strictly inside it. The
  // seam is a slit in the host that dead-ends at both tips, so a walk goes
  // around them and the host holds ONE cell — which reaches the seam from
  // both sides. The standing plate reaches it once, from a whole boundary
  // side. THREE live occurrences over TWO cells: the verdict counts
  // occurrences, so the piece fences and the forms stay apart; counting
  // distinct cells would read two and cross it.
  const auto slit =
      components_measure<Int, Real>(standing_plate<Real>(), false);
  CHECK(slit.n_crowded == 1);
  CHECK(slit.n_crowded_cells == 0);
  CHECK(slit.n_cells == 2);
  CHECK(slit.n_components == 2);
  CHECK(slit.n_fans == 1);
  CHECK(slit.n_quiet == 0);
  CHECK(slit.fans_only == 2);
  CHECK(slit.unfenced == 1);
  // The standing plate is uncut: one triangle, one component of its own.
  REQUIRE(slit.sizes.size() == 2);
  CHECK(slit.sizes[0] == 1);
}

TEMPLATE_TEST_CASE("plane components: the mesh's own edge count fences what "
                   "the arrangement cannot see",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // ONE mesh whose two pages meet on one edge, cut into the arrangement by a
  // plate crossing both and missing the edge. The edge is an ordinary
  // two-face crease: one live incidence from each page, equal depth, nothing
  // to decide — CROSSABLE, and the two pages' near sides are one component.
  // Each page's cut is a fan (the page on both sides plus the cutter on both
  // sides), the cutter's two slits leave it one cell, and five cells make
  // four components.
  const auto crease =
      components_measure<Int, Real>(crossed_pages<Real>(false), false);
  CHECK(crease.n_cells == 5);
  CHECK(crease.n_components == 4);
  CHECK(crease.n_fans == 2);
  CHECK(crease.n_crowded == 2);
  CHECK(crease.n_quiet == 0);
  CHECK(crease.fans_only == 4);
  CHECK(crease.unfenced == 1);

  // The same edge with a SECOND, larger page on each of its two planes. Each
  // pair pools, so the cell reaching the edge is covered twice on both sides:
  // the edge reads two live incidences and two equal depths, and both the
  // count and the depth are blind. Only the form's own edge link, which
  // states FOUR faces on that edge, fences it. The smaller pages' rims are
  // the depth borders — two sides each, each split by the cut, eight pieces —
  // and with the edge they are the nine quiet fences; the six cut pieces are
  // the fans; nothing joins.
  const auto doubled =
      components_measure<Int, Real>(crossed_pages<Real>(true), false);
  CHECK(doubled.n_cells == 11);
  CHECK(doubled.n_components == 11);
  CHECK(doubled.n_fans == 6);
  CHECK(doubled.n_crowded == 6);
  CHECK(doubled.n_quiet == 9);
  // With the quiet fences switched off the edge crosses again, exactly as the
  // crease does, and the two planes' near ground is one.
  CHECK(doubled.fans_only == 4);
  CHECK(doubled.unfenced == 1);

  // Both scenes answer the same with the within flag on: the pages pool and
  // resolve in plane whatever the flag says.
  check_same_arrangement(
      "crease, within on",
      components_measure<Int, Real>(crossed_pages<Real>(false), true), crease);
  check_same_arrangement(
      "doubled, within on",
      components_measure<Int, Real>(crossed_pages<Real>(true), true), doubled);
}

TEMPLATE_TEST_CASE("plane components: winding is not coverage",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // Two coplanar plates overlapping in one quarter. Three depth classes —
  // each plate's own part and the quarter both cover — and the diagonal the
  // two plates lay on one line splits every one of them in two, which the
  // equal depth joins straight back: six cells, three components. The four
  // rim sides that run through the other plate's interior are the depth
  // borders, and nothing else fences.
  const auto aligned =
      components_measure<Int, Real>(quarter_overlap<Real>(false), false);
  CHECK(aligned.n_components == 3);
  CHECK(aligned.n_cells == 6);
  CHECK(aligned.n_fans == 0);
  CHECK(aligned.n_crowded == 0);
  CHECK(aligned.n_quiet == 4);
  CHECK(aligned.fans_only == 1);
  CHECK(aligned.unfenced == 1);

  // The second plate wound the other way: opposing plates are still one
  // geometric plane, and this tier reads how many members cover the ground,
  // never which way they face. Which member's labels take which side of the
  // surviving wall is the csg tier's question, and the domain gates carry
  // that half.
  check_same_arrangement(
      "two tags, reversed",
      components_measure<Int, Real>(quarter_overlap<Real>(true), false),
      aligned);
  check_same_arrangement(
      "two tags, reversed, within on",
      components_measure<Int, Real>(quarter_overlap<Real>(true), true),
      aligned);
  check_same_arrangement("one tag, aligned",
                         components_measure<Int, Real>(
                             as_one_mesh(quarter_overlap<Real>(false)), false),
                         aligned);
  check_same_arrangement("one tag, reversed",
                         components_measure<Int, Real>(
                             as_one_mesh(quarter_overlap<Real>(true)), false),
                         aligned);
  check_same_arrangement("one tag, reversed, within on",
                         components_measure<Int, Real>(
                             as_one_mesh(quarter_overlap<Real>(true)), true),
                         aligned);
}

TEMPLATE_TEST_CASE("plane components: a bare touch fences at incidence two",
                   "[cut][planes][components]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using Real = typename components_real_of<Int>::type;

  // Two forms meeting on one shared edge and nowhere else. One cell reaches
  // the piece from each side, so exactly TWO live incidences meet there and
  // no count can see anything — but the contact is STATED: a second operand
  // touches here, its two pages divide trivially, and the piece is a fan on
  // the strength of that bit alone. Neither form is cut, and the two stay two.
  const auto touch =
      components_measure<Int, Real>(components_bare_touch<Real>(), false);
  CHECK(touch.n_components == 2);
  CHECK(touch.sizes == std::vector<components_index_t>{1, 1});
  CHECK(touch.n_cells == 2);
  CHECK(touch.n_fans == 1);
  CHECK(touch.n_crowded == 0);
  CHECK(touch.n_quiet == 0);
  CHECK(touch.fans_only == 2);
  CHECK(touch.unfenced == 1);
}
