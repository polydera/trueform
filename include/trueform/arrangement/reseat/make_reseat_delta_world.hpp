/*
 * Copyright (c) 2026 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../exact/input_lattice.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../intersect/graph/local_arrangement.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../../intersect/polygon_intersections.hpp"
#include "../identity/identity_substrate.hpp"
#include "../operands/boolean.hpp"
#include "../policy/arrangement_range_policy.hpp"
#include "./make_reseat_delta_forms.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::arrangement::reseat {

/// The world of the delta, stopped where the local arrangement ends: its
/// records, planes, definitions and identities are what the full build
/// would state for these faces, on these points, its identities in its own
/// substrate. Every piece enters the cut world whether or not a record names
/// it, as the carve's own faces do. Its points stand on the world's lattice,
/// so its converter is the identity carrying the world's `quantum`.
template <typename Index, typename Int> struct reseat_delta_world {
  using form_t = std::decay_t<
      decltype(std::declval<const tf::polygons_buffer<Index, Int, 3, 3> &>()
                   .polygons())>;
  using structs_t = decltype(tf::arrangement::dispatch::make_missing_structures(
      std::declval<form_t>()));
  using policy_t =
      tf::arrangement::arrangement_range_policy<tf::core::std_vector<form_t>,
                                                structs_t>;
  using world_t = tf::intersect::graph::local_arrangement<Index, Int, Int>;

  policy_t policy;
  tf::exact::input_lattice<Index, Int, Int> lattice;
  tf::arrangement::identity_substrate<Index, Int> identities;
  world_t world;
  tf::buffer<Index> face_counts;

  auto apply_to_form() const { return policy.make_apply_to_form(); }
  auto reader() const { return lattice.reader(apply_to_form()); }
};

template <typename Index, typename Int>
auto make_reseat_delta_world(const reseat_delta_forms<Index, Int> &forms,
                             tf::intersect_config config, Int quantum)
    -> reseat_delta_world<Index, Int> {
  using delta_t = reseat_delta_world<Index, Int>;
  using form_t = typename delta_t::form_t;
  tf::core::std_vector<form_t> views;
  for (const auto &mesh : forms.meshes)
    views.push_back(mesh.polygons());
  tf::small_vector<typename delta_t::structs_t, 10> structs(views.size());
  for (std::size_t i = 0; i < views.size(); ++i)
    structs[i] = tf::arrangement::dispatch::make_missing_structures(views[i]);
  delta_t delta{
      typename delta_t::policy_t(std::move(views), std::move(structs)),
      {},
      {},
      {},
      {}};
  config.tolerance = 0.0;
  delta.lattice = delta.policy.template make_lattice<Int>(
      tf::exact::pt_converter<Int, Int, 3>{{quantum}}, 0.0);
  const auto n_tags = delta.policy.n_tags();
  if (n_tags == Index(1))
    config.mode = config.mode | tf::intersect_mode::within;
  tf::polygon_intersections<Index, Int, Int> intersections;
  intersections.with_edge_splits(false);
  delta.policy.build_intersections(intersections, delta.lattice, config);
  const auto apply_form = delta.policy.make_apply_to_form();
  auto apply_to_face = [apply_form](int tag, Index object, const auto &f) {
    apply_form(Index(tag), [&](const auto &form) { f(form.faces()[object]); });
  };
  auto get_mesh_point = delta.lattice.reader(apply_form);
  delta.face_counts.allocate(std::size_t(n_tags) + 1);
  delta.face_counts[0] = Index(0);
  for (Index t = 0; t < n_tags; ++t)
    apply_form(t, [&, t](const auto &form) {
      delta.face_counts[std::size_t(t) + 1] =
          delta.face_counts[std::size_t(t)] + Index(form.faces().size());
    });
  tf::buffer<Index> carved_faces;
  for (Index t = 0; t < n_tags; ++t) {
    const auto &faces = forms.faces[std::size_t(t)];
    for (std::size_t f = 0; f < faces.size(); ++f)
      if (faces[f].role != delta_role::neighbour)
        carved_faces.push_back(delta.face_counts[std::size_t(t)] + Index(f));
  }
  delta.world.build(
      delta.identities, std::move(intersections), get_mesh_point, apply_to_face,
      apply_form, tf::make_range(delta.face_counts),
      bool(config.mode & tf::intersect_mode::within),
      delta.lattice.moved_points().size() != 0, tf::make_range(carved_faces));
  return delta;
}

} // namespace tf::arrangement::reseat
