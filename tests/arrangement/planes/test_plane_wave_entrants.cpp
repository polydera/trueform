/**
 * @file test_plane_wave_entrants.cpp
 * @brief The entrance discovery, asked directly, with each tier's own mask
 *
 * A weld retires an original vertex, and every source face incident to it
 * holds an instance of that vertex. The faces the asking world does not name
 * are exactly the ones the identity leaves the cut world to reach, so the
 * discovery must answer the vertex's ring MINUS what the mask already holds —
 * no more, because promoting a face the world names states one (tag, object)
 * twice, and no less, because a face left behind names an identity nothing
 * moved for it.
 *
 * Both tiers ask the one discovery, so both masks are driven here: the cut
 * world's own descriptors, and the plane wave's dense answered set. The
 * wave's mask also carries the entrance's termination law — a face it has
 * answered for is never offered again — and that is asserted by answering one
 * and asking again.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "input_lattice_for.hpp"
#include "tagged_operand.hpp"

#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/planes/discover_plane_wave_entrants.hpp>
#include <trueform/arrangement/planes/plane_wave_answered.hpp>
#include <trueform/arrangement/planes/plane_world.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/graph/plane_uncut_entrants.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

using entrants_index_t = int;
using real_t = float;
using int_t = tf::exact::int32;
using entrants_mesh_t = tf::polygons_buffer<entrants_index_t, real_t, 3, 3>;
using operand_t = tf::test::tagged_operand<entrants_index_t, real_t>;

auto translated(entrants_mesh_t mesh, real_t dx, real_t dy, real_t dz)
    -> entrants_mesh_t {
  for (std::size_t i = 0; i < mesh.points_buffer().size(); ++i) {
    mesh.points_buffer()[i][0] += dx;
    mesh.points_buffer()[i][1] += dy;
    mesh.points_buffer()[i][2] += dz;
  }
  return mesh;
}

/// Two unit boxes overlapping at one corner: a few faces of each are cut and
/// most are not, which is the only shape this discovery has anything to say
/// about.
struct scene_t {
  std::vector<std::unique_ptr<operand_t>> operands;
  std::vector<decltype(std::declval<operand_t &>().form())> forms;
  tf::buffer<entrants_index_t> face_offsets;
  tf::intersect::graph::local_arrangement<entrants_index_t, real_t, int_t>
      world;

  auto apply_to_face() const {
    const auto *scene = this;
    return [scene](int tag, entrants_index_t object, const auto &apply) {
      apply(scene->forms[std::size_t(tag)].faces()[object]);
    };
  }
  auto apply_to_form() const {
    const auto *scene = this;
    return [scene](entrants_index_t tag, const auto &apply) {
      apply(scene->forms[std::size_t(tag)]);
    };
  }

  scene_t() {
    operands.push_back(std::make_unique<operand_t>(
        tf::make_box_mesh<entrants_index_t>(real_t(1), real_t(1), real_t(1))));
    operands.push_back(std::make_unique<operand_t>(translated(
        tf::make_box_mesh<entrants_index_t>(real_t(1), real_t(1), real_t(1)),
        real_t(0.6), real_t(0.6), real_t(0.6))));
    for (auto &operand : operands)
      forms.push_back(operand->form());
    const auto form_range =
        tf::make_range(forms.data(), forms.data() + forms.size());
    face_offsets.allocate(forms.size() + 1);
    face_offsets[0] = 0;
    for (std::size_t tag = 0; tag < forms.size(); ++tag)
      face_offsets[tag + 1] =
          face_offsets[tag] + entrants_index_t(forms[tag].faces().size());

    tf::polygon_intersections<entrants_index_t, real_t, int_t> intersections;
    intersections.with_edge_splits(false);
    const auto lattice = tf::test::input_lattice_for(form_range, 0.0);
    intersections.build(
        form_range, lattice,
        tf::intersect_config{tf::intersect_mode::primitives |
                                 tf::intersect_mode::resolve_crossing_contours,
                             0.0});
    const auto converter = lattice.converter();
    const auto get_mesh_point =
        [this, converter](int tag, entrants_index_t id) -> tf::point<int_t, 3> {
      return converter.convert(forms[std::size_t(tag)].points()[id]);
    };
    world.build(std::move(intersections), get_mesh_point, apply_to_face(),
                apply_to_form(), tf::make_range(face_offsets), false, false);
  }
};

/// The ring a mask leaves unnamed, stated the way the discovery must state
/// it: flat source faces, ascending, each once.
template <typename IsNamed>
auto unnamed_ring(const scene_t &scene, entrants_index_t tag,
                  entrants_index_t vertex, const IsNamed &is_named)
    -> std::vector<entrants_index_t> {
  std::vector<entrants_index_t> ring;
  scene.apply_to_form()(tag, [&](const auto &form) {
    for (const auto face : form.face_membership()[vertex])
      if (!is_named(tag, entrants_index_t(face)))
        ring.push_back(scene.face_offsets[std::size_t(tag)] +
                       entrants_index_t(face));
  });
  std::sort(ring.begin(), ring.end());
  ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
  return ring;
}

auto as_vector(const tf::buffer<entrants_index_t> &entrants)
    -> std::vector<entrants_index_t> {
  return std::vector<entrants_index_t>(entrants.begin(), entrants.end());
}

} // namespace

TEST_CASE("plane entrants: a retired original brings in the faces its own "
          "mask does not name",
          "[arrangement][planes][entrants]") {
  scene_t scene;
  const auto graph_descriptors = scene.world.graph().descriptors();
  const auto face_offsets = scene.world.face_offsets();
  const auto cut_world_names = [&](entrants_index_t tag,
                                   entrants_index_t object) {
    return tf::intersect::graph::plane_graph_names_face<entrants_index_t>(
        graph_descriptors, face_offsets, tag, object);
  };

  // The vertex to retire: one the cut world holds PART of the ring of, so
  // the answer is a proper subset of the ring and the mask is what decides
  // it. A vertex whose ring is wholly named or wholly unnamed would let a
  // discovery that ignores the mask pass.
  entrants_index_t retired_vertex = entrants_index_t(-1);
  std::size_t ring_size = 0;
  scene.apply_to_form()(entrants_index_t(0), [&](const auto &form) {
    for (entrants_index_t v = 0; v < entrants_index_t(form.points().size());
         ++v) {
      const auto ring =
          unnamed_ring(scene, entrants_index_t(0), v, cut_world_names);
      const auto whole = form.face_membership()[v].size();
      if (retired_vertex == entrants_index_t(-1) && !ring.empty() &&
          ring.size() < whole) {
        retired_vertex = v;
        ring_size = whole;
      }
    }
  });
  REQUIRE(retired_vertex != entrants_index_t(-1));
  REQUIRE(
      unnamed_ring(scene, entrants_index_t(0), retired_vertex, cut_world_names)
          .size() < ring_size);

  tf::buffer<entrants_index_t> retired;
  retired.push_back(scene.world.vertex_offsets()[0] + retired_vertex);
  const tf::buffer<entrants_index_t> no_roots;

  tf::buffer<entrants_index_t> entrants;
  tf::intersect::graph::discover_uncut_entrants(
      scene.world.graph(), tf::make_range(retired), tf::make_range(no_roots),
      scene.world.vertex_offsets(), face_offsets, scene.apply_to_form(),
      cut_world_names, entrants);
  REQUIRE(as_vector(entrants) == unnamed_ring(scene, entrants_index_t(0),
                                              retired_vertex, cut_world_names));

  // THE WAVE'S MASK answers the same question with its own authority, and a
  // face it has answered for is never offered again — which is what ends the
  // entrance.
  const auto plane_world = tf::arrangement::make_plane_world(scene.world);
  tf::arrangement::plane_wave_answered<entrants_index_t> answered;
  const auto wave_names = [&](entrants_index_t tag, entrants_index_t object) {
    return answered.answered(tag, object);
  };
  tf::buffer<entrants_index_t> wave_entrants;
  tf::arrangement::discover_plane_wave_entrants(
      plane_world, tf::make_range(retired), tf::make_range(no_roots), answered,
      plane_world.face_offsets(), scene.apply_to_form(), wave_entrants);
  const auto wave_ring =
      unnamed_ring(scene, entrants_index_t(0), retired_vertex, wave_names);
  REQUIRE(!wave_ring.empty());
  REQUIRE(as_vector(wave_entrants) == wave_ring);

  const auto before = answered.count;
  answered.answer(plane_world.face_offsets(), wave_entrants[0]);
  REQUIRE(answered.count == before + 1);
  tf::arrangement::discover_plane_wave_entrants(
      plane_world, tf::make_range(retired), tf::make_range(no_roots), answered,
      plane_world.face_offsets(), scene.apply_to_form(), wave_entrants);
  REQUIRE(
      as_vector(wave_entrants) ==
      std::vector<entrants_index_t>(wave_ring.begin() + 1, wave_ring.end()));

  // nothing retired and nothing split reaches nothing
  const tf::buffer<entrants_index_t> nothing;
  tf::buffer<entrants_index_t> empty;
  tf::intersect::graph::discover_uncut_entrants(
      scene.world.graph(), tf::make_range(nothing), tf::make_range(no_roots),
      scene.world.vertex_offsets(), face_offsets, scene.apply_to_form(),
      cut_world_names, empty);
  REQUIRE(empty.size() == 0u);
}
